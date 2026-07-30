# -*- coding: utf-8 -*-
"""
8.1 Computational Cost and Runtime Overhead Measurement
========================================================
对每个数据集统计:
  - #SEUs, #Nodes, #Hyperedges
  - JSONL load time
  - Hypergraph build time
  - Training time / epoch (GPU)
  - Inference time / SEU (GPU)
  - Peak GPU memory

输出表:
  Table X. Computational cost and runtime overhead.
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
from pathlib import Path
ROOT = str(Path(__file__).resolve().parent.parent)  # project root

print = functools.partial(print, flush=True)

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = ROOT
PROGRESS_DAPT = os.path.join(BASE_DIR, "progress_DAPT")

DATASETS = {
    "OPTC": {
        "dir": os.path.join(BASE_DIR, "data_OPTC"),
        "files": ["train_all.jsonl", "val_all.jsonl", "test_all.jsonl"],
        "progress": os.path.join(BASE_DIR, "progress_OPTC"),
    },
    "TCE5": {
        "dir": os.path.join(BASE_DIR, "data_TCE5"),
        "files": ["train_all.jsonl", "val.jsonl", "test_all.jsonl"],
        "progress": os.path.join(BASE_DIR, "progress_TCE5"),
    },
    "DAPT": {
        "dir": os.path.join(BASE_DIR, "data_DAPT"),
        "files": ["train.jsonl", "val.jsonl", "test.jsonl"],
        "progress": os.path.join(BASE_DIR, "progress_DAPT"),
    },
}

# 训练超参 (与正式实验一致)
MODEL_CFG = dict(
    emb_dim=256, num_layers=2, dropout=0.2,
    k_hop=2, max_edges=48, max_nodes=192,
    max_members_per_edge=128, max_hes_per_node=128,
    hub_degree_skip=0, batch_size=128,
)

BENCH_EPOCHS = 3       # 跑 3 个 epoch 取平均
BENCH_INFER_BATCHES = 10  # 跑 10 个 batch 取平均推理时间


# ============================================================
# 导入依赖 (复用已有代码)
# ============================================================

def _import_build_fn():
    build_path = os.path.join(BASE_DIR, "1.2Build_train.py")
    spec = importlib.util.spec_from_file_location("build_train", build_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_hypergraph_from_semantic_slices


def _import_progress(progress_dir):
    """导入指定 progress 目录的 utils / dataset_new / model"""
    sys.path.insert(0, progress_dir)
    old_cwd = os.getcwd()
    os.chdir(progress_dir)

    import utils as _u
    import dataset_new as _d
    import model as _m

    os.chdir(old_cwd)
    return _u, _d, _m


# ============================================================
# 数据加载 & 统计
# ============================================================

def load_jsonl_timed(path):
    t0 = time.time()
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    elapsed = time.time() - t0
    return records, elapsed


def count_tokens(records):
    tokens = set()
    for r in records:
        for m in r.get("motifs", []):
            for e in m.get("entities", []):
                tokens.add(e)
    return len(tokens)


# ============================================================
# 构建超图 (计时)
# ============================================================

def build_hypergraph_timed(records, build_fn):
    label_map = {lab: i for i, lab in enumerate(
        sorted({s.get("semantic_label", "UNK") for s in records})
    )}
    t0 = time.time()
    data_pt, he_records, token2nid, nid2type = build_fn(records, label_map)
    elapsed = time.time() - t0
    return data_pt, he_records, token2nid, nid2type, elapsed, label_map


# ============================================================
# 训练 & 推理基准测试
# ============================================================

def benchmark_train_infer(
    progress_dir, hg_json_path, label_map,
    n_train_edges, device,
):
    """
    加载超图 → 创建模型 → 跑几个 epoch + 推理 → 返回时间统计
    """
    utils_mod, ds_mod, model_mod = _import_progress(progress_dir)

    # 加载超图
    g = utils_mod.load_global_hypergraph_from_json(hg_json_path)

    # 创建 Dataset (用 train 的全部 hyperedge_ids)
    all_hids = np.array(list(g.he2nodes.keys()))
    # 取一个子集用于基准测试 (最多 5000 个, 加速)
    max_bench = min(5000, len(all_hids))
    rng = np.random.RandomState(42)
    bench_hids = rng.choice(all_hids, size=max_bench, replace=False)

    ds = ds_mod.HyperedgeSubgraphDataset(
        g=g,
        indices=bench_hids,
        k_hop=MODEL_CFG["k_hop"],
        max_edges=MODEL_CFG["max_edges"],
        max_nodes=MODEL_CFG["max_nodes"],
        max_members_per_edge=MODEL_CFG["max_members_per_edge"],
        max_hes_per_node=MODEL_CFG["max_hes_per_node"],
        hub_degree_skip=MODEL_CFG["hub_degree_skip"],
        seed=42,
        label2cid=label_map,
    )

    loader = DataLoader(
        ds, batch_size=MODEL_CFG["batch_size"],
        shuffle=False, num_workers=0,
        collate_fn=ds_mod.collate_subgraph_ids,
    )

    # 创建模型
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

    # --- GPU 内存重置 ---
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    # --- 训练基准测试 ---
    epoch_times = []
    n_batches_per_epoch = len(loader)

    for ep in range(BENCH_EPOCHS):
        model.train()
        t0 = time.time()

        for batch in loader:
            H, node_ids, edge_hids, node_mask, edge_mask, y = batch
            H = H.to(device)
            y = y.to(device)

            # gather features (从 CPU 全局特征 → GPU)
            node_feats, edge_feats = _gather_feats(g, node_ids, edge_hids, device)

            optimizer.zero_grad()
            z = model(H, node_feats, edge_feats)

            # 简单线性头做分类
            logits = z @ torch.randn(z.size(1), n_classes, device=device)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        epoch_times.append(time.time() - t0)

    train_time_per_epoch = np.mean(epoch_times)

    # --- 推理基准测试 ---
    model.eval()
    infer_times = []
    n_inferred = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= BENCH_INFER_BATCHES:
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

            batch_size = H.size(0)
            infer_times.append(elapsed / batch_size * 1000)  # ms per SEU
            n_inferred += batch_size

    inference_ms_per_seu = np.mean(infer_times)

    # --- GPU 内存 ---
    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # 推算完整 epoch 时间 (基于 5000 样本的基准)
    full_epoch_time = train_time_per_epoch * (n_train_edges / max_bench)

    return {
        "train_sec_per_epoch": round(train_time_per_epoch, 2),
        "train_sec_full_epoch": round(full_epoch_time, 2),
        "inference_ms_per_seu": round(inference_ms_per_seu, 3),
        "peak_gpu_mem_mb": round(peak_mem_mb, 1),
        "bench_samples": max_bench,
        "bench_batches": n_batches_per_epoch,
    }


def _gather_feats(g, node_ids, edge_hids, device):
    """简化版 gather_batch_global_feats"""
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
            row_idx = []
            pos_keep = []
            for j, hid in enumerate(hid_list):
                hid = int(hid)
                if hid in g.hid2idx:
                    row_idx.append(g.hid2idx[hid])
                    pos_keep.append(int(pos_all[j]))
            if row_idx:
                feats = g.edge_feats[torch.tensor(row_idx, dtype=torch.long)]
                edge_feats[b, torch.tensor(pos_keep, dtype=torch.long)] = feats

    return node_feats.to(device), edge_feats.to(device)


# ============================================================
# 主函数: 测量单个数据集
# ============================================================

def measure_dataset(name, cfg, build_fn, device):
    """对单个数据集做全部测量"""
    print(f"\n  {'='*55}")
    print(f"  Dataset: {name}")
    print(f"  {'='*55}")

    result = {"dataset": name}

    # --- 1. 加载 JSONL ---
    all_records = []
    total_load_time = 0
    for fn in cfg["files"]:
        fp = os.path.join(cfg["dir"], fn)
        recs, t = load_jsonl_timed(fp)
        all_records.extend(recs)
        total_load_time += t
    n_seus = len(all_records)
    n_tokens = count_tokens(all_records)
    result["n_seus"] = n_seus
    result["n_tokens"] = n_tokens
    result["load_time_sec"] = round(total_load_time, 2)
    print(f"    #SEUs: {n_seus:,}   #Tokens: {n_tokens:,}   Load: {total_load_time:.2f}s")

    # --- 2. 构建超图 ---
    data_pt, he_records, token2nid, nid2type, build_time, label_map = \
        build_hypergraph_timed(all_records, build_fn)
    n_nodes = data_pt["num_nodes"]
    n_edges = data_pt["num_hyperedges"]
    result["n_nodes"] = n_nodes
    result["n_hyperedges"] = n_edges
    result["build_time_sec"] = round(build_time, 2)
    print(f"    #Nodes: {n_nodes:,}   #Hyperedges: {n_edges:,}   Build: {build_time:.2f}s")

    # --- 3. 保存临时超图 JSON (用于加载) ---
    tmp_json = os.path.join(cfg["dir"], f"_bench_Hyper_train.json")
    from pathlib import Path
    # 使用 build 模块的 save 函数
    build_spec = importlib.util.spec_from_file_location(
        "build_train", os.path.join(BASE_DIR, "1.2Build_train.py"))
    build_mod = importlib.util.module_from_spec(build_spec)
    build_spec.loader.exec_module(build_mod)

    build_mod.save_hypergraph_json(tmp_json, all_records, token2nid, nid2type, he_records)
    print(f"    Temp hypergraph saved: {tmp_json}")

    # --- 4. 训练 & 推理基准测试 ---
    try:
        bench = benchmark_train_infer(
            progress_dir=cfg["progress"],
            hg_json_path=tmp_json,
            label_map=label_map,
            n_train_edges=n_edges,
            device=device,
        )
        result.update(bench)
        print(f"    Train/epoch (bench {bench['bench_samples']} samples): {bench['train_sec_per_epoch']:.2f}s")
        print(f"    Train/epoch (full, extrapolated): {bench['train_sec_full_epoch']:.2f}s")
        print(f"    Inference: {bench['inference_ms_per_seu']:.3f} ms/SEU")
        print(f"    Peak GPU mem: {bench['peak_gpu_mem_mb']:.1f} MB")
    except Exception as e:
        print(f"    [WARN] Training benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        result.update({
            "train_sec_per_epoch": -1,
            "train_sec_full_epoch": -1,
            "inference_ms_per_seu": -1,
            "peak_gpu_mem_mb": -1,
        })
    finally:
        # 清理临时文件
        try:
            os.remove(tmp_json)
        except Exception:
            pass

    return result


# ============================================================
# 汇总表
# ============================================================

def print_summary_table(all_results):
    print()
    print("=" * 130)
    print("  Table X. Computational Cost and Runtime Overhead")
    print("=" * 130)

    header = (
        f"{'Dataset':<8s} "
        f"{'#SEUs':>9s} {'#Tokens':>9s} {'#Nodes':>9s} {'#HE':>9s} "
        f"{'Load(s)':>8s} {'Build(s)':>9s} "
        f"{'Train/ep':>10s} {'Infer ms':>10s} {'GPU Mem':>9s}"
    )
    print(header)
    print("-" * 130)

    for r in all_results:
        train_ep = f"{r['train_sec_full_epoch']:.1f}s" if r.get('train_sec_full_epoch', -1) > 0 else "N/A"
        infer = f"{r['inference_ms_per_seu']:.2f}" if r.get('inference_ms_per_seu', -1) > 0 else "N/A"
        mem = f"{r['peak_gpu_mem_mb']:.0f} MB" if r.get('peak_gpu_mem_mb', -1) > 0 else "N/A"

        print(
            f"{r['dataset']:<8s} "
            f"{r['n_seus']:>9,d} {r['n_tokens']:>9,d} {r['n_nodes']:>9,d} {r['n_hyperedges']:>9,d} "
            f"{r['load_time_sec']:>8.1f} {r['build_time_sec']:>9.1f} "
            f"{train_ep:>10s} {infer:>10s} {mem:>9s}"
        )

    print("=" * 130)
    print()
    print("  Theoretical Complexity:")
    print("    SEU abstraction:        O(\\sum_k |S_k|)")
    print("    Hypergraph construction: O(\\sum_k |T_k|)")
    print("    Hypergraph propagation:  O(L \\sum_{M_k in M} |M_k| d)")
    print("=" * 130)


def save_csv(all_results, out_path):
    fieldnames = [
        "dataset", "n_seus", "n_tokens", "n_nodes", "n_hyperedges",
        "load_time_sec", "build_time_sec",
        "train_sec_per_epoch", "train_sec_full_epoch",
        "inference_ms_per_seu", "peak_gpu_mem_mb",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n  Results saved: {out_path}")


# ============================================================
# 入口
# ============================================================

def run(out_dir=None):
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("  8.1 Computational Cost & Runtime Measurement")
    print("=" * 60)

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()

    build_fn = _import_build_fn()

    all_results = []
    for name, cfg in DATASETS.items():
        r = measure_dataset(name, cfg, build_fn, device)
        all_results.append(r)

    print_summary_table(all_results)
    save_csv(all_results, os.path.join(out_dir, "runtime_cost.csv"))

    return all_results


if __name__ == "__main__":
    run()
