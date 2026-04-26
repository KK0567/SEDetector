# utils.py
# -*- coding: utf-8 -*-
import json
import math
import random
import hashlib
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from collections import defaultdict, deque

import numpy as np
import torch


# ---------------------------
# basic
# ---------------------------
def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_hash(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()
    return int(h[:8], 16)


def hash_bow(counter: Dict[str, float], dim: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    if not counter:
        return v
    for k, val in counter.items():
        idx = stable_hash(str(k)) % dim
        v[idx] += float(val)
    return v


def hash_string_to_vec(s: str, dim: int) -> np.ndarray:
    tokens = []
    cur = []
    for ch in str(s).lower():
        if ch.isalnum() or ch in "._-:":
            cur.append(ch)
        else:
            if cur:
                tokens.append("".join(cur))
                cur = []
    if cur:
        tokens.append("".join(cur))

    cnt = {}
    for tk in tokens[:300]:
        cnt[tk] = cnt.get(tk, 0.0) + 1.0
    return hash_bow(cnt, dim)


# ---------------------------
# type features
# ---------------------------
def type_list_default() -> List[str]:
    return ["entity", "event"]


def onehot_type(t: str, type_list: List[str]) -> np.ndarray:
    v = np.zeros(len(type_list), dtype=np.float32)
    if t in type_list:
        v[type_list.index(t)] = 1.0
    else:
        v[0] = 1.0
    return v


@dataclass
class HypergraphMeta:
    json_path: str
    N: int
    E: int
    type_list: List[str]


@dataclass
class GlobalHypergraph:
    # row-indexed tensors (row_idx = 0..E-1)
    node_feats: torch.Tensor      # [N, Dn]  (node_id通常是0..N-1)
    edge_feats: torch.Tensor      # [E, De]  (row_idx)
    labels: List[str]             # len=E    (row_idx)
    label_ids: torch.Tensor       # [E]      (row_idx)
    label2id: Dict[str, int]
    id2label: List[str]

    # incidence indexes keyed by hyperedge_id (hid)
    node2hes: Dict[int, List[int]]   # node_id -> list of hyperedge_id
    he2nodes: Dict[int, List[int]]   # hyperedge_id -> list of node_id

    # mapping between hid and row_idx
    hid2idx: Dict[int, int]          # hyperedge_id -> row_idx
    idx2hid: List[int]               # row_idx -> hyperedge_id

    meta: HypergraphMeta


def build_incidence_index(hyperedges: List[dict]) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    注意：这里的 key 一律使用 hyperedge_id（hid），不是 row_idx。
    """
    node2hes = defaultdict(list)
    he2nodes = {}
    for he in hyperedges:
        hid = int(he["hyperedge_id"])
        members = [int(x) for x in he.get("members", [])]
        he2nodes[hid] = members
        for nid in members:
            node2hes[nid].append(hid)
    return node2hes, he2nodes


def load_global_hypergraph_from_json(
    json_path: str,
    node_hash_dim: int = 64,
    event_hash_dim: int = 128,
    tech_hash_dim: int = 128,
    device: str = "cpu",
) -> GlobalHypergraph:
    """
    读取 split 级超图 JSON：
      - node_feats: 以 node_id 为索引 (通常连续)
      - edge_feats/labels: 以 row_idx(0..E-1) 为索引
      - node2hes/he2nodes: 以 hyperedge_id 为 key（因为数据里 hyperedge_id 可能不连续）
      - hid2idx/idx2hid: 提供 hyperedge_id <-> row_idx 的映射（关键修复点）
    """
    with open(json_path, "r", encoding="utf-8") as f:
        hg = json.load(f)

    nodes = hg.get("nodes", [])
    hyperedges = hg.get("hyperedges", [])

    N = len(nodes)
    E = len(hyperedges)

    tlist = type_list_default()

    # --------- hid <-> row_idx mapping (关键) ----------
    idx2hid = [int(he["hyperedge_id"]) for he in hyperedges]
    hid2idx = {hid: i for i, hid in enumerate(idx2hid)}

    # ---------- incidence index keyed by hid ----------
    node2hes, he2nodes = build_incidence_index(hyperedges)

    # ---------- node degree (by hid incidence) ----------
    deg = np.zeros(N, dtype=np.float32)
    for he in hyperedges:
        for nid in he.get("members", []):
            nid = int(nid)
            if 0 <= nid < N:
                deg[nid] += 1.0

    # ---------- node feats ----------
    node_feats = []
    for n in nodes:
        nid = int(n.get("node_id", 0))
        tok = str(n.get("token", ""))
        nt = str(n.get("type", "entity")).lower()

        t_one = onehot_type(nt, tlist)

        stats = np.zeros(3, dtype=np.float32)
        d = deg[nid] if 0 <= nid < N else 0.0
        stats[0] = math.log1p(d)
        stats[1] = d / max(N, 1)
        stats[2] = 1.0 if d > 0 else 0.0

        h_tok = hash_string_to_vec(tok, node_hash_dim)

        feat = np.concatenate([t_one, stats, h_tok], axis=0)
        node_feats.append(feat)

    node_feats = torch.from_numpy(np.stack(node_feats).astype(np.float32)).to(device)

    # ---------- edge feats + labels (row_idx order) ----------
    labels = []
    edge_feats = []

    for he in hyperedges:
        labels.append(str(he.get("label", "UNK")))
        score = float(he.get("semantic_score", 0.0))

        feats = he.get("features", {}) or {}
        event_types = feats.get("event_types", {}) or {}
        techniques = feats.get("techniques", {}) or {}

        if isinstance(event_types, list):
            event_counter = {str(x): 1.0 for x in event_types}
        else:
            event_counter = {str(k): float(v) for k, v in dict(event_types).items()}

        if isinstance(techniques, list):
            tech_counter = {str(x): 1.0 for x in techniques}
        else:
            tech_counter = {str(k): float(v) for k, v in dict(techniques).items()}

        ev_hash = hash_bow(event_counter, event_hash_dim)
        te_hash = hash_bow(tech_counter, tech_hash_dim)

        num_members = float(len(he.get("members", [])))
        num_ev = float(len(event_counter))
        num_te = float(len(tech_counter))

        scalars = np.array([score, num_members, num_ev, num_te], dtype=np.float32)
        ef = np.concatenate([scalars, ev_hash, te_hash], axis=0)
        edge_feats.append(ef)

    edge_feats = torch.from_numpy(np.stack(edge_feats).astype(np.float32)).to(device)

    # ---------- label mapping (per file) ----------
    uniq = sorted(list(set(labels)))
    label2id = {lb: i for i, lb in enumerate(uniq)}
    id2label = [lb for lb, _ in sorted(label2id.items(), key=lambda kv: kv[1])]
    y = torch.tensor([label2id[lb] for lb in labels], dtype=torch.long, device=device)

    g = GlobalHypergraph(
        node_feats=node_feats,
        edge_feats=edge_feats,
        labels=labels,
        label_ids=y,
        label2id=label2id,
        id2label=id2label,
        node2hes=dict(node2hes),
        he2nodes=he2nodes,
        hid2idx=hid2idx,
        idx2hid=idx2hid,
        meta=HypergraphMeta(json_path=json_path, N=N, E=E, type_list=tlist),
    )
    return g


# ---------------------------
# k-hop subhypergraph (二部图 BFS) + hub 抑制
# ---------------------------
def k_hop_subhypergraph(
    target_hid: int,
    node2hes: Dict[int, List[int]],
    he2nodes: Dict[int, List[int]],
    k_hop: int = 1,
    max_edges: int = 64,
    max_nodes: int = 256,
    max_members_per_edge: int = 64,
    max_hes_per_node: int = 48,
    hub_degree_skip: int = 0,
    seed: int = 0,
) -> Tuple[List[int], List[int], List[List[int]]]:
    """
    返回：
      hids_global: 全局 hyperedge_id 列表（hids_global[0]==target_hid）
      nids_global: 全局 node_id 列表
      sub_edges:   list(list(local_node_id)) 与 hids_global 同序
    """
    rng = np.random.RandomState(seed)

    target_hid = int(target_hid)
    if target_hid not in he2nodes:
        return [], [], []

    visited_h = set([target_hid])
    visited_n = set()

    q = deque()
    q.append(("h", target_hid, 0))

    while q:
        typ, idx, d = q.popleft()
        if d >= 2 * k_hop:
            continue

        if typ == "h":
            members = he2nodes.get(int(idx), [])
            for nid in members:
                nid = int(nid)
                if nid not in visited_n:
                    visited_n.add(nid)
                    q.append(("n", nid, d + 1))
                    if len(visited_n) >= max_nodes:
                        break

        else:  # typ == "n"
            hes = node2hes.get(int(idx), [])
            deg = len(hes)

            if hub_degree_skip and deg >= int(hub_degree_skip):
                continue

            if deg > max_hes_per_node:
                choose = rng.choice(deg, size=max_hes_per_node, replace=False)
                hes = [hes[i] for i in choose.tolist()]

            for hid in hes:
                hid = int(hid)
                if hid not in visited_h:
                    visited_h.add(hid)
                    q.append(("h", hid, d + 1))
                    if len(visited_h) >= max_edges:
                        break

        if len(visited_h) >= max_edges and len(visited_n) >= max_nodes:
            break

    hids_global = [target_hid] + [h for h in visited_h if h != target_hid]
    nids_global = list(visited_n)

    hids_global = hids_global[:max_edges]
    nids_global = nids_global[:max_nodes]

    nid2local = {nid: i for i, nid in enumerate(nids_global)}

    sub_edges: List[List[int]] = []
    for hid in hids_global:
        members_global = [int(x) for x in he2nodes.get(int(hid), []) if int(x) in nid2local]
        if len(members_global) > max_members_per_edge:
            members_global = members_global[:max_members_per_edge]
        members_local = [nid2local[n] for n in members_global]
        sub_edges.append(members_local)

    if len(sub_edges) == 0 or len(sub_edges[0]) == 0:
        return [], [], []

    return hids_global, nids_global, sub_edges
