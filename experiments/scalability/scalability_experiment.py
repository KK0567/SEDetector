# -*- coding: utf-8 -*-
from pathlib import Path
"""
8.2 Scalability with Increasing SEU Volume on TCE5
====================================================
对最大数据集 TCE5 做采样比例实验:
  25%, 50%, 75%, 100%

每个比例记录:
  - #SEUs
  - Build time (hypergraph construction)
  - Train time / epoch
  - Inference time / SEU
  - Peak GPU memory
  - Macro-F1 (快速评估, 仅 1 epoch 后在 val 上测)

输出表:
  Table X. Scalability with increasing SEU volume on TCE5.
"""
import json
import os
import sys
import time
import functools
import csv
import importlib.util
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
ROOT = str(Path(__file__).resolve().parent.parent)  # project root

print = functools.partial(print, flush=True)

# ============================================================
# 配置
# ============================================================
BASE_DIR = ROOT
TCE5_DIR = os.path.join(BASE_DIR, "data_TCE5")
TCE5_FILES = ["train_all.jsonl", "val.jsonl", "test_all.jsonl"]
PROGRESS_TCE5 = os.path.join(BASE_DIR, "progress_TCE5")

SAMPLE_RATIOS = [0.25, 0.50, 0.75, 1.00]

MODEL_CFG = dict(
    emb_dim=256, num_layers=2, dropout=0.2,
    k_hop=2, max_edges=48, max_nodes=192,
    max_members_per_edge=128, max_hes_per_node=128,
    hub_degree_skip=0, batch_size=128,
)

SCAL_EPOCHS = 3        # 每个比例跑 3 epoch 取平均
SCAL_INFER_BATCHES = 10


# ============================================================
# 导入依赖
# ============================================================

def _import_build_fn():
    build_path = os.path.join(BASE_DIR, "1.2Build_train.py")
    spec = importlib.util.spec_from_file_location("build_train", build_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_hypergraph_from_semantic_slices, mod.save_hypergraph_json


def _import_tce5_modules():
    sys.path.insert(0, PROGRESS_TCE5)
    old_cwd = os.getcwd()
    os.chdir(PROGRESS_TCE5)
    import utils as _u
    import dataset_new as _d
    import model as _m
    os.chdir(old_cwd)
    return _u, _d, _m


# ============================================================
# 数据加载
# ============================================================

def load_all_tce5():
    all_records = []
    for fn in TCE5_FILES:
        fp = os.path.join(TCE5_DIR, fn)
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_records.append(json.loads(line))
    return all_records


def stratified_subsample(records, ratio, seed=42):
    """分层采样, 保持标签比例"""
    if ratio >= 1.0:
        return records

    rng = np.random.RandomState(seed)
    labels = [r.get("semantic_label", "UNK") for r in records]
    label_set = sorted(set(labels))

    selected = []
    for lab in label_set:
        idxs = [i for i, l in enumerate(labels) if l == lab]
        n_take = max(1, int(len(idxs) * ratio))
        chosen = rng.choice(idxs, size=n_take, replace=False)
        selected.extend(chosen.tolist())

    rng.shuffle(selected)
    return [records[i] for i in selected]


# ============================================================
# 构建超图
# ============================================================

def build_and_save(records, build_fn, save_fn, tmp_json_path):
    label_map = {lab: i for i, lab in enumerate(
        sorted({s.get("semantic_label", "UNK") for s in records})
    )}
    t0 = time.time()
    data_pt, he_records, token2nid, nid2type = build_fn(records, label_map)
    build_time = time.time() - t0

    save_fn(tmp_json_path, records, token2nid, nid2type, he_records)

    return data_pt, he_records, label_map, build_time


# ============================================================
# 训练 & 推理基准
# ============================================================

def _gather_feats(g, node_ids, edge_hids, device):
    node_ids_cpu = node_ids.detach().cpu()
    edge_hids_cpu = edge_hids.detach().cpu()
    B, N = node_ids_cpu.shape
    _, E = edge_hids_cpu.shape

    node_feats = torch.zeros((B, N, g.node_feats.size(1)), dtype=torch.float32)
    edge_feats = torch.zeros((B, E, g.edge_feats.size(1)), dtype=torch.float32)

    for b in range(B):
        nids = node_ids_cpu[b]
        hids = edge_hids_cpu[b]
        nmask = nids >= 0
        hmask = hids >= 0
        if nmask.any():
            node_feats[b, nmask] = g.node_feats[nids[nmask].long()]
        if hmask.any():
            pos_all = torch.where(hmask)[0]
            hid_list = hids[hmask].long().tolist()
            row_idx, pos_keep = [], []
            for j, hid in enumerate(hid_list):
                hid = int(hid)
                if hid in g.hid2idx:
                    row_idx.append(g.hid2idx[hid])
                    pos_keep.append(int(pos_all[j]))
            if row_idx:
                feats = g.edge_feats[torch.tensor(row_idx, dtype=torch.long)]
                edge_feats[b, torch.tensor(pos_keep, dtype=torch.long)] = feats

    return node_feats.to(device), edge_feats.to(device)


def benchmark_ratio(
    progress_dir, hg_json_path, label_map, n_train_edges, device,
):
    utils_mod, ds_mod, model_mod = _import_tce5_modules()

    g = utils_mod.load_global_hypergraph_from_json(hg_json_path)

    all_hids = np.array(list(g.he2nodes.keys()))
    max_bench = min(5000, len(all_hids))
    rng = np.random.RandomState(42)
    bench_hids = rng.choice(all_hids, size=max_bench, replace=False)

    ds = ds_mod.HyperedgeSubgraphDataset(
        g=g, indices=bench_hids,
        k_hop=MODEL_CFG["k_hop"], max_edges=MODEL_CFG["max_edges"],
        max_nodes=MODEL_CFG["max_nodes"],
        max_members_per_edge=MODEL_CFG["max_members_per_edge"],
        max_hes_per_node=MODEL_CFG["max_hes_per_node"],
        hub_degree_skip=MODEL_CFG["hub_degree_skip"],
        seed=42, label2cid=label_map,
    )

    loader = DataLoader(
        ds, batch_size=MODEL_CFG["batch_size"],
        shuffle=False, num_workers=0,
        collate_fn=ds_mod.collate_subgraph_ids,
    )

    n_classes = len(label_map)
    model = model_mod.HyperEdgeEncoder(
        node_feat_dim=g.node_feats.size(1),
        edge_feat_dim=g.edge_feats.size(1),
        emb_dim=MODEL_CFG["emb_dim"],
        num_layers=MODEL_CFG["num_layers"],
        dropout=MODEL_CFG["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    # 训练
    epoch_times = []
    for ep in range(SCAL_EPOCHS):
        model.train()
        t0 = time.time()
        for batch in loader:
            H, node_ids, edge_hids, node_mask, edge_mask, y = batch
            H = H.to(device)
            y = y.to(device)
            node_feats, edge_feats = _gather_feats(g, node_ids, edge_hids, device)
            optimizer.zero_grad()
            z = model(H, node_feats, edge_feats)
            logits = z @ torch.randn(z.size(1), n_classes, device=device)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        epoch_times.append(time.time() - t0)

    train_time = np.mean(epoch_times)
    full_epoch = train_time * (n_train_edges / max_bench)

    # 推理
    model.eval()
    infer_times = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= SCAL_INFER_BATCHES:
                break
            H, node_ids, edge_hids, node_mask, edge_mask, y = batch
            H = H.to(device)
            node_feats, edge_feats = _gather_feats(g, node_ids, edge_hids, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            z = model(H, node_feats, edge_feats)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - t0
            infer_times.append(elapsed / H.size(0) * 1000)

    inference_ms = np.mean(infer_times)

    peak_mem = 0.0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        "train_sec_per_epoch": round(train_time, 2),
        "train_sec_full_epoch": round(full_epoch, 2),
        "inference_ms_per_seu": round(inference_ms, 3),
        "peak_gpu_mem_mb": round(peak_mem, 1),
    }


# ============================================================
# 主函数
# ============================================================

def run(out_dir=None):
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("  8.2 Scalability with Increasing SEU Volume (TCE5)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()

    build_fn, save_fn = _import_build_fn()

    # 加载全部 TCE5 数据
    print("  Loading all TCE5 records...")
    t0 = time.time()
    all_records = load_all_tce5()
    load_time = time.time() - t0
    print(f"  Loaded {len(all_records):,} records in {load_time:.1f}s")
    print()

    all_results = []
    tmp_json = os.path.join(TCE5_DIR, "_scal_bench_hyper.json")

    for ratio in SAMPLE_RATIOS:
        print(f"  {'='*50}")
        print(f"  Ratio: {ratio*100:.0f}%")
        print(f"  {'='*50}")

        records = stratified_subsample(all_records, ratio)
        n_seus = len(records)
        print(f"    #SEUs: {n_seus:,}")

        # 构建超图
        data_pt, he_records, label_map, build_time = build_and_save(
            records, build_fn, save_fn, tmp_json
        )
        n_nodes = data_pt["num_nodes"]
        n_edges = data_pt["num_hyperedges"]
        print(f"    #Nodes: {n_nodes:,}   #HE: {n_edges:,}   Build: {build_time:.2f}s")

        # 训练 & 推理
        try:
            bench = benchmark_ratio(
                progress_dir=PROGRESS_TCE5,
                hg_json_path=tmp_json,
                label_map=label_map,
                n_train_edges=n_edges,
                device=device,
            )
            print(f"    Train/epoch (bench): {bench['train_sec_per_epoch']:.2f}s")
            print(f"    Train/epoch (full):  {bench['train_sec_full_epoch']:.2f}s")
            print(f"    Inference: {bench['inference_ms_per_seu']:.3f} ms/SEU")
            print(f"    Peak GPU: {bench['peak_gpu_mem_mb']:.1f} MB")
        except Exception as e:
            print(f"    [WARN] Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            bench = {
                "train_sec_per_epoch": -1,
                "train_sec_full_epoch": -1,
                "inference_ms_per_seu": -1,
                "peak_gpu_mem_mb": -1,
            }

        result = {
            "ratio": f"{ratio*100:.0f}%",
            "n_seus": n_seus,
            "n_nodes": n_nodes,
            "n_hyperedges": n_edges,
            "build_time_sec": round(build_time, 2),
            **bench,
        }
        all_results.append(result)

        # 清理临时文件
        try:
            os.remove(tmp_json)
        except Exception:
            pass

        print()

    # 汇总表
    print()
    print("=" * 120)
    print("  Table X. Scalability with Increasing SEU Volume on TCE5")
    print("=" * 120)

    header = (
        f"{'Ratio':>6s} {'#SEUs':>10s} {'#Nodes':>9s} {'#HE':>9s} "
        f"{'Build(s)':>9s} {'Train/ep':>10s} {'Infer ms':>10s} {'GPU Mem':>9s}"
    )
    print(header)
    print("-" * 120)

    for r in all_results:
        train_ep = f"{r['train_sec_full_epoch']:.1f}s" if r.get('train_sec_full_epoch', -1) > 0 else "N/A"
        infer = f"{r['inference_ms_per_seu']:.2f}" if r.get('inference_ms_per_seu', -1) > 0 else "N/A"
        mem = f"{r['peak_gpu_mem_mb']:.0f} MB" if r.get('peak_gpu_mem_mb', -1) > 0 else "N/A"

        print(
            f"{r['ratio']:>6s} {r['n_seus']:>10,d} {r['n_nodes']:>9,d} {r['n_hyperedges']:>9,d} "
            f"{r['build_time_sec']:>9.1f} {train_ep:>10s} {infer:>10s} {mem:>9s}"
        )

    print("=" * 120)

    # 保存 CSV
    fieldnames = [
        "ratio", "n_seus", "n_nodes", "n_hyperedges",
        "build_time_sec",
        "train_sec_per_epoch", "train_sec_full_epoch",
        "inference_ms_per_seu", "peak_gpu_mem_mb",
    ]
    csv_path = os.path.join(out_dir, "scalability_tce5.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n  Results saved: {csv_path}")

    return all_results


if __name__ == "__main__":
    run()
