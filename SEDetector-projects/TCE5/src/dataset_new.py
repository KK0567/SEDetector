# dataset_new.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import GlobalHypergraph, k_hop_subhypergraph


@dataclass
class Sample:
    target_hid: int                # hyperedge_id (global within this split)
    hids_global: List[int]         # list of hyperedge_id (global), first is target
    nids_global: List[int]         # list of node_id
    sub_edges: List[List[int]]     # list of local node indices per hyperedge
    y: int                         # class id in train label space


class LRUCache:
    """key=target_hid -> (hids_global, nids_global, sub_edges)"""
    def __init__(self, max_size: int = 20000):
        self.max_size = int(max_size)
        self._od: "OrderedDict[int, Tuple[List[int], List[int], List[List[int]]]]" = OrderedDict()

    def get(self, k: int):
        if k not in self._od:
            return None
        v = self._od.pop(k)
        self._od[k] = v
        return v

    def put(self, k: int, v):
        if self.max_size <= 0:
            return
        if k in self._od:
            self._od.pop(k)
        self._od[k] = v
        if len(self._od) > self.max_size:
            self._od.popitem(last=False)


class HyperedgeSubgraphDataset(Dataset):
    """
    indices: 一定是 hyperedge_id 列表（不是 row_idx）
    """
    def __init__(
        self,
        g: GlobalHypergraph,
        indices: np.ndarray,            # hyperedge_id list
        k_hop: int = 1,
        max_edges: int = 64,
        max_nodes: int = 256,
        max_members_per_edge: int = 64,
        max_hes_per_node: int = 48,
        hub_degree_skip: int = 0,
        seed: int = 0,
        cache_size: int = 20000,
        cache: Optional[LRUCache] = None,
        label2cid: Optional[dict] = None,   # train label space mapping
    ):
        self.g = g
        self.indices = indices.astype(np.int64)  # hyperedge_id
        self.k_hop = int(k_hop)
        self.max_edges = int(max_edges)
        self.max_nodes = int(max_nodes)
        self.max_members_per_edge = int(max_members_per_edge)
        self.max_hes_per_node = int(max_hes_per_node)
        self.hub_degree_skip = int(hub_degree_skip)
        self.seed = int(seed)

        self.cache = cache if cache is not None else LRUCache(cache_size)
        self.label2cid = label2cid
        if self.label2cid is None:
            raise ValueError("label2cid must be provided (use train split mapping)!")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int) -> Sample:
        target_hid = int(self.indices[i])

        cached = self.cache.get(target_hid)
        if cached is None:
            hids_global, nids_global, sub_edges = k_hop_subhypergraph(
                target_hid=target_hid,
                node2hes=self.g.node2hes,
                he2nodes=self.g.he2nodes,
                k_hop=self.k_hop,
                max_edges=self.max_edges,
                max_nodes=self.max_nodes,
                max_members_per_edge=self.max_members_per_edge,
                max_hes_per_node=self.max_hes_per_node,
                hub_degree_skip=self.hub_degree_skip,
                seed=(self.seed + target_hid) % 1000003,
            )
            if not sub_edges:
                # 极端保护：中心超边无法构图
                return Sample(target_hid=target_hid, hids_global=[], nids_global=[], sub_edges=[], y=0)

            self.cache.put(target_hid, (hids_global, nids_global, sub_edges))
        else:
            hids_global, nids_global, sub_edges = cached

        # label 从 row_idx 获取：row_idx = hid2idx[target_hid]
        if target_hid not in self.g.hid2idx:
            raise KeyError(f"target_hid {target_hid} not in g.hid2idx (split mismatch).")
        ridx = self.g.hid2idx[target_hid]
        lb = str(self.g.labels[ridx])

        if lb not in self.label2cid:
            raise ValueError(
                f"Unknown label in this split: {lb}. "
                f"Train label space={list(self.label2cid.keys())}"
            )

        y = int(self.label2cid[lb])
        return Sample(
            target_hid=target_hid,
            hids_global=hids_global,
            nids_global=nids_global,
            sub_edges=sub_edges,
            y=y
        )


def collate_subgraph_ids(batch: List[Sample], device: Optional[torch.device] = None):
    """
    输出：
      H:        [B, Emax, Nmax]
      node_ids: [B, Nmax]  (node_id, pad=-1)
      edge_hids:[B, Emax]  (hyperedge_id, pad=-1)  <-- 关键：这里保留 hyperedge_id
      y:        [B]
    """
    # 过滤空样本
    batch = [s for s in batch if s.sub_edges and s.nids_global and s.hids_global]
    if len(batch) == 0:
        dev = device if device is not None else torch.device("cpu")
        return (
            torch.zeros((0, 1, 1), device=dev),
            torch.full((0, 1), -1, dtype=torch.long, device=dev),
            torch.full((0, 1), -1, dtype=torch.long, device=dev),
            torch.zeros((0, 1), device=dev),
            torch.zeros((0, 1), device=dev),
            torch.zeros((0,), dtype=torch.long, device=dev),
        )

    B = len(batch)
    Emax = max(len(s.sub_edges) for s in batch)
    Nmax = max(len(s.nids_global) for s in batch)

    dev = device if device is not None else torch.device("cpu")
    H = torch.zeros((B, Emax, Nmax), dtype=torch.float32, device=dev)
    node_mask = torch.zeros((B, Nmax), dtype=torch.float32, device=dev)
    edge_mask = torch.zeros((B, Emax), dtype=torch.float32, device=dev)
    y = torch.zeros((B,), dtype=torch.long, device=dev)

    node_ids = torch.full((B, Nmax), -1, dtype=torch.long, device=dev)
    edge_hids = torch.full((B, Emax), -1, dtype=torch.long, device=dev)

    for bi, s in enumerate(batch):
        n_nodes = len(s.nids_global)
        n_edges = len(s.sub_edges)

        node_mask[bi, :n_nodes] = 1.0
        edge_mask[bi, :n_edges] = 1.0
        y[bi] = int(s.y)

        node_ids[bi, :n_nodes] = torch.as_tensor(s.nids_global, dtype=torch.long, device=dev)
        edge_hids[bi, :n_edges] = torch.as_tensor(s.hids_global, dtype=torch.long, device=dev)

        for e_idx, members in enumerate(s.sub_edges[:Emax]):
            if not members:
                continue
            m = torch.as_tensor(members, dtype=torch.long, device=dev)
            m = m[(m >= 0) & (m < n_nodes)]
            if m.numel() > 0:
                H[bi, e_idx, m] = 1.0

    return H, node_ids, edge_hids, node_mask, edge_mask, y
