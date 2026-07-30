# -*- coding: utf-8 -*-
from pathlib import Path
"""
9. Top-k Perturbation Faithfulness Experiment
===============================================
对每个测试样本, 按重要性排名移除 top-k 和 random-k 组件,
测量预测概率下降、翻转率和忠实度差距.

三个解释粒度:
  - SEU (evidence): 超边特征掩码
  - Token: 节点结构+特征移除
  - Neighbor HE: 邻居超边结构+特征移除

k 值: 1, 3, 5
Random baseline: 20 次重复取平均

输出表:
  Table X. Top-k perturbation faithfulness comparison.
"""
import json
import os
import sys
import time
import math
import random
import functools
import csv
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
ROOT = str(Path(__file__).resolve().parent.parent)  # project root

print = functools.partial(print, flush=True)

# ============================================================
# 数据集配置
# ============================================================
BASE_DIR = ROOT

DATASETS = {
    "OPTC": {
        "progress": os.path.join(BASE_DIR, "progress_OPTC"),
        "train_hg": os.path.join(BASE_DIR, "data_OPTC", "Hyper_train.json"),
        "test_hg": os.path.join(BASE_DIR, "data_OPTC", "Hyper_test.json"),
        "ckpt": os.path.join(BASE_DIR, "progress_OPTC", "outputs_OPTC", "BEST", "best.pt"),
        "run_script": "run_OPTC",
        "k_hop": 2, "max_edges": 48, "max_nodes": 192,
        "max_members_per_edge": 48, "max_hes_per_node": 32,
        "hub_degree_skip": 0,
        "mode": "head_kd", "emb_dim": 256, "num_layers": 2, "dropout": 0.2,
        "tau": 0.07, "head_tau": 0.05, "logit_adj": 0.08, "logit_adj_mode": "sub",
    },
    "TCE5": {
        "progress": os.path.join(BASE_DIR, "progress_TCE5"),
        "train_hg": os.path.join(BASE_DIR, "data_TCE5", "Hyper_train.json"),
        "test_hg": os.path.join(BASE_DIR, "data_TCE5", "Hyper_test.json"),
        "ckpt": os.path.join(BASE_DIR, "progress_TCE5", "outputs_TCE5", "BEST", "best.pt"),
        "run_script": "run_TCE5",
        "k_hop": 1, "max_edges": 48, "max_nodes": 192,
        "max_members_per_edge": 128, "max_hes_per_node": 128,
        "hub_degree_skip": 3,
        "mode": "head_kd", "emb_dim": 256, "num_layers": 2, "dropout": 0.2,
        "tau": 0.10, "head_tau": 0.05, "logit_adj": 0.15, "logit_adj_mode": "sub",
    },
    "DAPT": {
        "progress": os.path.join(BASE_DIR, "progress_DAPT"),
        "train_hg": os.path.join(BASE_DIR, "data_DAPT", "Hyper_train.json"),
        "test_hg": os.path.join(BASE_DIR, "data_DAPT", "Hyper_test.json"),
        "ckpt": os.path.join(BASE_DIR, "progress_DAPT", "outputs_DAPT", "BEST", "best.pt"),
        "run_script": "run_DAPT",
        "k_hop": 1, "max_edges": 48, "max_nodes": 192,
        "max_members_per_edge": 128, "max_hes_per_node": 128,
        "hub_degree_skip": 3,
        "mode": "head_kd", "emb_dim": 256, "num_layers": 2, "dropout": 0.2,
        "tau": 0.10, "head_tau": 0.05, "logit_adj": 0.15, "logit_adj_mode": "sub",
    },
}

TOPK_VALUES = [1, 3, 5]
RANDOM_REPEATS = 20
MAX_TEST_SAMPLES = 500  # 最多测 500 个样本


# ============================================================
# 导入 progress 目录的模块
# ============================================================

_loaded_modules = {}

def load_progress_modules(ds_name):
    """加载指定数据集的 progress 目录模块 (每次都重新加载以避免冲突)"""
    cfg = DATASETS[ds_name]
    progress_dir = cfg["progress"]

    # 清理旧的 progress 路径
    for p in list(sys.path):
        if "progress_" in p and "experiments" not in p:
            sys.path.remove(p)

    sys.path.insert(0, progress_dir)
    old_cwd = os.getcwd()
    os.chdir(progress_dir)

    # 清除所有可能的缓存
    for mod_name in ["utils", "dataset_new", "model", "layers1", "layers",
                     "run_DAPT", "run_new_2", "run_OPTC", "run_TCE5"]:
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    import importlib
    utils = importlib.import_module("utils")
    dataset_new = importlib.import_module("dataset_new")
    model = importlib.import_module("model")

    run_mod_name = cfg["run_script"]
    run_mod = importlib.import_module(run_mod_name)

    os.chdir(old_cwd)

    mods = {
        "utils": utils,
        "dataset_new": dataset_new,
        "model": model,
        "run": run_mod,
    }
    return mods


# ============================================================
# 模型加载
# ============================================================

def setup_model(ds_name, device):
    """加载训练好的模型"""
    cfg = DATASETS[ds_name]
    mods = load_progress_modules(ds_name)

    # 加载训练超图 (获取特征维度和标签映射)
    g_train = mods["utils"].load_global_hypergraph_from_json(cfg["train_hg"])
    label2cid = g_train.label2id
    n_classes = len(label2cid)

    # 创建模型
    model = mods["model"].HyperEdgeEncoder(
        node_feat_dim=g_train.node_feats.size(1),
        edge_feat_dim=g_train.edge_feats.size(1),
        emb_dim=cfg["emb_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    head = mods["run"].CosineHead(cfg["emb_dim"], n_classes).to(device)

    # 加载 checkpoint
    ckpt = torch.load(cfg["ckpt"], map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    if ckpt.get("head", None) is not None:
        head.load_state_dict(ckpt["head"])
        head.to(device)
        head.eval()

    prototypes = ckpt.get("prototypes", None)
    if prototypes is not None:
        if not isinstance(prototypes, torch.Tensor):
            prototypes = torch.tensor(prototypes)
        prototypes = prototypes.to(device)
        prototypes = F.normalize(prototypes, dim=-1)

    # log_prior
    cnt = Counter()
    for lb in g_train.labels:
        if lb in label2cid:
            cnt[label2cid[lb]] += 1
    sizes = [cnt.get(i, 1) for i in range(n_classes)]
    freq = torch.tensor(sizes, dtype=torch.float32, device=device)
    prior = freq / (freq.sum() + 1e-12)
    log_prior = torch.log(prior + 1e-12)

    return model, head, prototypes, g_train, label2cid, log_prior, n_classes


# ============================================================
# 前向推理
# ============================================================

@torch.no_grad()
def forward_pass(model, head, prototypes, mode, H, node_feats, edge_feats,
                 tau, head_tau, log_prior, logit_adj, logit_adj_mode):
    """单次前向推理, 返回 logits, prob, pred"""
    z = model(H, node_feats, edge_feats)
    z = F.normalize(z, dim=1)

    if mode == "proto_only":
        run_mod = list(_loaded_modules.values())[0]["run"]
        logits = run_mod.logits_from_multi_prototypes(z, prototypes, tau=tau, reduce="logsumexp")
    elif mode in ("head_only", "head_kd"):
        logits = head(z, tau=head_tau)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # logit adjustment
    if logit_adj > 0 and log_prior is not None:
        if logit_adj_mode == "sub":
            logits = logits - logit_adj * log_prior.unsqueeze(0)
        else:
            logits = logits + logit_adj * log_prior.unsqueeze(0)

    prob = F.softmax(logits, dim=1)
    pred = logits.argmax(dim=1)
    return logits, prob, pred


def gather_feats(g, node_ids, edge_hids, device):
    """从全局超图收集节点/边特征"""
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


# ============================================================
# 重要性排名 (leave-one-out)
# ============================================================

def compute_importance(model, head, prototypes, cfg, H, node_feats, edge_feats,
                       node_ids, edge_hids, target_cid, base_prob, device,
                       log_prior=None):
    """计算 token 和 hyperedge 的 leave-one-out 重要性"""
    mode = cfg["mode"]
    tau, head_tau = cfg["tau"], cfg["head_tau"]
    la, lam = cfg["logit_adj"], cfg["logit_adj_mode"]

    # Token importance
    N = H.size(2)
    token_importance = []
    inc = (H[0].sum(dim=0) > 0).cpu().numpy()
    active_nodes = [i for i in range(N) if node_ids[0, i].item() >= 0 and inc[i]]

    for n in active_nodes:
        Hm = H.clone()
        nfm = node_feats.clone()
        Hm[:, :, n] = 0.0
        nfm[:, n, :] = 0.0
        _, prob, _ = forward_pass(model, head, prototypes, mode, Hm, nfm, edge_feats,
                                   tau, head_tau, log_prior, la, lam)
        drop = base_prob - float(prob[0, target_cid].item())
        token_importance.append({"idx": n, "drop": drop})

    token_importance.sort(key=lambda x: x["drop"], reverse=True)

    # Hyperedge importance (neighboring HE, exclude center e=0)
    E = H.size(1)
    he_importance = []
    for e in range(1, E):  # skip center (e=0)
        if edge_hids[0, e].item() < 0:
            continue
        Hm = H.clone()
        efm = edge_feats.clone()
        Hm[:, e, :] = 0.0
        efm[:, e, :] = 0.0
        _, prob, _ = forward_pass(model, head, prototypes, mode, Hm, node_feats, efm,
                                   tau, head_tau, log_prior, la, lam)
        drop = base_prob - float(prob[0, target_cid].item())
        he_importance.append({"idx": e, "drop": drop})

    he_importance.sort(key=lambda x: x["drop"], reverse=True)

    return token_importance, he_importance


# ============================================================
# Top-k / Random-k Perturbation
# ============================================================

def perturb_and_measure(model, head, prototypes, cfg, H, node_feats, edge_feats,
                        target_cid, base_prob, base_pred,
                        importance_list, remove_type, k, rng,
                        log_prior=None):
    """
    对单个样本做 top-k 和 random-k 扰动, 返回结果字典.
    remove_type: "token" 或 "hyperedge"
    """
    mode = cfg["mode"]
    tau, head_tau = cfg["tau"], cfg["head_tau"]
    la, lam = cfg["logit_adj"], cfg["logit_adj_mode"]

    actual_k = min(k, len(importance_list))
    if actual_k == 0:
        return {
            "top_prob_drop": 0.0, "top_flip": 0,
            "rand_prob_drop": 0.0, "rand_flip_rate": 0.0,
            "faithfulness_gap": 0.0, "n_available": 0,
        }

    # --- Top-k removal ---
    top_indices = [x["idx"] for x in importance_list[:actual_k]]
    Hm = H.clone()
    nfm = node_feats.clone()
    efm = edge_feats.clone()

    for idx in top_indices:
        if remove_type == "token":
            Hm[:, :, idx] = 0.0
            nfm[:, idx, :] = 0.0
        elif remove_type == "hyperedge":
            Hm[:, idx, :] = 0.0
            efm[:, idx, :] = 0.0

    _, prob, pred = forward_pass(model, head, prototypes, mode, Hm, nfm, efm,
                                  tau, head_tau, log_prior, la, lam)
    top_prob = float(prob[0, target_cid].item())
    top_drop = base_prob - top_prob
    top_flip = int(pred[0].item() != base_pred)

    # --- Random-k removal (多次取平均) ---
    all_indices = [x["idx"] for x in importance_list]
    rand_drops = []
    rand_flips = []

    for _ in range(RANDOM_REPEATS):
        if actual_k >= len(all_indices):
            rand_idx = list(all_indices)
        else:
            rand_idx = rng.sample(all_indices, actual_k)

        Hm = H.clone()
        nfm = node_feats.clone()
        efm = edge_feats.clone()

        for idx in rand_idx:
            if remove_type == "token":
                Hm[:, :, idx] = 0.0
                nfm[:, idx, :] = 0.0
            elif remove_type == "hyperedge":
                Hm[:, idx, :] = 0.0
                efm[:, idx, :] = 0.0

        _, prob, pred = forward_pass(model, head, prototypes, mode, Hm, nfm, efm,
                                      tau, head_tau, log_prior, la, lam)
        rp = float(prob[0, target_cid].item())
        rand_drops.append(base_prob - rp)
        rand_flips.append(int(pred[0].item() != base_pred))

    rand_drop_mean = float(np.mean(rand_drops))
    rand_flip_rate = float(np.mean(rand_flips))
    faithfulness_gap = top_drop - rand_drop_mean

    return {
        "top_prob_drop": round(top_drop, 4),
        "top_flip": top_flip,
        "rand_prob_drop": round(rand_drop_mean, 4),
        "rand_flip_rate": round(rand_flip_rate, 4),
        "faithfulness_gap": round(faithfulness_gap, 4),
        "n_available": len(all_indices),
    }


# ============================================================
# 单数据集主函数
# ============================================================

def run_dataset(ds_name, device):
    """对单个数据集运行 perturbation 实验"""
    cfg = DATASETS[ds_name]
    mods = load_progress_modules(ds_name)

    print(f"\n  Loading model from {cfg['ckpt']} ...")
    model, head, prototypes, g_train, label2cid, log_prior, n_classes = \
        setup_model(ds_name, device)

    print(f"  Loading test hypergraph from {cfg['test_hg']} ...")
    g_test = mods["utils"].load_global_hypergraph_from_json(cfg["test_hg"])

    # 创建 test dataset
    all_test_hids = np.array(list(g_test.he2nodes.keys()))
    n_test = min(MAX_TEST_SAMPLES, len(all_test_hids))
    rng_np = np.random.RandomState(42)
    test_hids = rng_np.choice(all_test_hids, size=n_test, replace=False)

    ds = mods["dataset_new"].HyperedgeSubgraphDataset(
        g=g_test, indices=test_hids,
        k_hop=cfg["k_hop"], max_edges=cfg["max_edges"],
        max_nodes=cfg["max_nodes"],
        max_members_per_edge=cfg["max_members_per_edge"],
        max_hes_per_node=cfg["max_hes_per_node"],
        hub_degree_skip=cfg["hub_degree_skip"],
        seed=42, label2cid=label2cid,
    )

    print(f"  Test samples: {n_test}")
    print(f"  Running perturbation analysis (k={TOPK_VALUES})...")

    rng = random.Random(42)
    all_results = []
    n_processed = 0
    n_skipped = 0

    for i in range(len(ds)):
        sample = ds[i]
        if not sample.sub_edges or not sample.nids_global:
            n_skipped += 1
            continue

        # 构建单样本张量
        batch = mods["dataset_new"].collate_subgraph_ids([sample])
        H, node_ids, edge_hids, node_mask, edge_mask, y = batch
        H = H.to(device)
        node_ids = node_ids.to(device)
        edge_hids = edge_hids.to(device)

        node_feats, edge_feats = gather_feats(g_test, node_ids, edge_hids, device)

        # 基线推理
        _, prob, pred = forward_pass(model, head, prototypes, cfg["mode"],
                                      H, node_feats, edge_feats,
                                      cfg["tau"], cfg["head_tau"],
                                      log_prior, cfg["logit_adj"], cfg["logit_adj_mode"])
        target_cid = int(pred[0].item())
        base_prob = float(prob[0, target_cid].item())

        if base_prob < 0.3:
            n_skipped += 1
            continue

        # 计算重要性
        token_imp, he_imp = compute_importance(
            model, head, prototypes, cfg,
            H, node_feats, edge_feats, node_ids, edge_hids,
            target_cid, base_prob, device, log_prior=log_prior,
        )

        # 对每个 k 值做扰动
        for k in TOPK_VALUES:
            # Token perturbation
            tok_res = perturb_and_measure(
                model, head, prototypes, cfg,
                H, node_feats, edge_feats,
                target_cid, base_prob, target_cid,
                token_imp, "token", k, rng, log_prior=log_prior,
            )
            tok_res.update({"dataset": ds_name, "level": "Token", "k": k,
                            "sample_idx": i})
            all_results.append(tok_res)

            # Hyperedge perturbation
            he_res = perturb_and_measure(
                model, head, prototypes, cfg,
                H, node_feats, edge_feats,
                target_cid, base_prob, target_cid,
                he_imp, "hyperedge", k, rng, log_prior=log_prior,
            )
            he_res.update({"dataset": ds_name, "level": "Neighbor HE", "k": k,
                            "sample_idx": i})
            all_results.append(he_res)

        n_processed += 1
        if n_processed % 50 == 0:
            print(f"    Processed {n_processed}/{n_test} samples...")

    print(f"  Done: {n_processed} processed, {n_skipped} skipped")
    return all_results, n_processed


# ============================================================
# 汇总输出
# ============================================================

def print_summary_table(all_results, sample_counts):
    """打印汇总表"""
    print()
    print("=" * 130)
    print("  Table X. Top-k Perturbation Faithfulness Comparison")
    print("=" * 130)

    header = (
        f"{'Dataset':<8s} {'Level':<14s} "
        f"{'Top-1 drop':>11s} {'Top-3 drop':>11s} {'Top-5 drop':>11s} "
        f"{'Rand-1 drop':>12s} {'Rand-3 drop':>12s} {'Rand-5 drop':>12s} "
        f"{'Flip rate':>10s} {'Faith gap':>10s}"
    )
    print(header)
    print("-" * 130)

    for ds in ["OPTC", "TCE5", "DAPT"]:
        ds_results = [r for r in all_results if r["dataset"] == ds]
        n_samples = sample_counts.get(ds, 0)

        for level in ["Token", "Neighbor HE"]:
            row_data = []
            for k in TOPK_VALUES:
                subset = [r for r in ds_results
                          if r["level"] == level and r["k"] == k]
                if not subset:
                    row_data.append((0, 0, 0, 0, 0))
                    continue
                avg_top = np.mean([r["top_prob_drop"] for r in subset])
                avg_rand = np.mean([r["rand_prob_drop"] for r in subset])
                avg_flip = np.mean([r["top_flip"] for r in subset])
                avg_gap = np.mean([r["faithfulness_gap"] for r in subset])
                row_data.append((avg_top, avg_rand, avg_flip, avg_gap, len(subset)))

            if len(row_data) < 3:
                continue

            t1, r1, f1, g1, _ = row_data[0]
            t3, r3, f3, g3, _ = row_data[1]
            t5, r5, f5, g5, _ = row_data[2]

            avg_flip_rate = np.mean([f1, f3, f5])
            avg_gap = np.mean([g1, g3, g5])

            print(
                f"{ds:<8s} {level:<14s} "
                f"{t1:>11.4f} {t3:>11.4f} {t5:>11.4f} "
                f"{r1:>12.4f} {r3:>12.4f} {r5:>12.4f} "
                f"{avg_flip_rate:>9.1%} {avg_gap:>10.4f}"
            )

    print("=" * 130)

    # 分析说明
    print()
    print("  Notes:")
    print("  - Top-k drop: average probability drop when removing top-k most important components")
    print("  - Rand-k drop: average probability drop when removing k random components (20 repeats)")
    print("  - Flip rate: fraction of samples where prediction changes after top-k removal")
    print("  - Faithfulness gap: Top-k drop - Random-k drop (positive = faithful explanation)")
    print("  - Token: node-level perturbation (zeroes incidence + features)")
    print("  - Neighbor HE: neighboring hyperedge removal (excludes center SEU)")
    print("=" * 130)


def save_csv(all_results, out_path):
    """保存详细 CSV"""
    fieldnames = [
        "dataset", "level", "k", "sample_idx",
        "top_prob_drop", "top_flip",
        "rand_prob_drop", "rand_flip_rate",
        "faithfulness_gap", "n_available",
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
    print("  9. Top-k Perturbation Faithfulness Experiment")
    print("  (R2/R4: Explainability Ranking Validation)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Top-k values: {TOPK_VALUES}")
    print(f"  Random repeats: {RANDOM_REPEATS}")
    print(f"  Max test samples: {MAX_TEST_SAMPLES}")
    print()

    all_results = []
    sample_counts = {}

    for ds_name in ["OPTC", "TCE5", "DAPT"]:
        print(f"\n  {'='*55}")
        print(f"  Dataset: {ds_name}")
        print(f"  {'='*55}")

        try:
            results, n = run_dataset(ds_name, device)
            all_results.extend(results)
            sample_counts[ds_name] = n
        except Exception as e:
            print(f"  [ERROR] {ds_name} failed: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        print_summary_table(all_results, sample_counts)
        save_csv(all_results, os.path.join(out_dir, "perturbation_faithfulness.csv"))

    return all_results


if __name__ == "__main__":
    run()
