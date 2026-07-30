# -*- coding: utf-8 -*-
from pathlib import Path
"""
Modality-Dependent SEU and Hypergraph Statistics
=================================================
统计 host-log (OPTC, TCE5) vs network-flow (DAPT) 的 SEU 和超图结构差异.

输出表:
  Table X. Modality-dependent SEU and hypergraph statistics.

统计指标:
  1. Avg. tokens / SEU              — 每个 SEU 的平均实体令牌数
  2. Avg. role relations / SEU      — 每个 SEU 的平均角色关系数
  3. Avg. hyperedge size            — 平均超边大小 (实体 + 事件节点)
  4. % process/file tokens          — 进程/文件类令牌占比
  5. % endpoint/protocol/service    — 端点/协议/服务类令牌占比
  6. Token entropy                  — 令牌分布的信息熵
  7. Average degree                 — 超图中节点的平均度
"""
import json
import os
import sys
import math
import functools
import csv
from collections import Counter, defaultdict

import numpy as np
ROOT = str(Path(__file__).resolve().parent.parent)  # project root

print = functools.partial(print, flush=True)


# ============================================================
# 数据加载
# ============================================================

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_dataset(name):
    base = {
        "OPTC": os.path.join(ROOT, "data_OPTC"),
        "TCE5": os.path.join(ROOT, "data_TCE5"),
        "DAPT": os.path.join(ROOT, "data_DAPT"),
    }[name]
    files = {
        "OPTC": ["train_all.jsonl", "val_all.jsonl", "test_all.jsonl"],
        "TCE5": ["train_all.jsonl", "val.jsonl", "test_all.jsonl"],
        "DAPT": ["train.jsonl", "val.jsonl", "test.jsonl"],
    }[name]
    all_recs = []
    for fn in files:
        all_recs.extend(load_jsonl(os.path.join(base, fn)))
    return all_recs


# ============================================================
# 令牌分类
# ============================================================

# Host-log 令牌: 裸 hex hash (e.g., "231e05a2") → 通常是 process/file 节点
# DAPT 令牌: 带前缀 (HOST_*, PEER_*, PORT_*, P_*, SVC_*)

PROCESS_FILE_PREFIXES = ("HOST_",)  # DAPT 中的主机/进程令牌
ENDPOINT_PROTO_PREFIXES = ("PEER_", "PORT_", "P_", "SVC_")  # 端点/协议/服务


def classify_token(token):
    """
    分类令牌:
    - 'process_file': 进程/文件类 (host-log hex hash, DAPT HOST_*)
    - 'endpoint_proto': 端点/协议/服务类 (DAPT PEER_*, PORT_*, P_*, SVC_*)
    - 'other': 其他
    """
    for prefix in PROCESS_FILE_PREFIXES:
        if token.startswith(prefix):
            return "process_file"
    for prefix in ENDPOINT_PROTO_PREFIXES:
        if token.startswith(prefix):
            return "endpoint_proto"
    # 裸 hex hash (OPTC/TCE5) → process_file
    if all(c in "0123456789abcdef" for c in token) and len(token) >= 6:
        return "process_file"
    return "other"


# ============================================================
# 统计计算
# ============================================================

def compute_modality_stats(records, name):
    """计算单个数据集的模态统计"""
    n_seus = len(records)

    tokens_per_seu = []
    roles_per_seu = []
    hyperedge_sizes = []
    all_tokens = []
    token_type_counts = Counter()

    # SEU-level 统计
    for rec in records:
        motifs = rec.get("motifs", [])
        seu_tokens = set()
        seu_roles = set()
        he_size = 0

        for m in motifs:
            entities = m.get("entities", [])
            event_type = m.get("event_type", "")

            # 实体令牌
            for e in entities:
                seu_tokens.add(e)
                all_tokens.append(e)
                token_type_counts[classify_token(e)] += 1

            # 事件类型也计入超边大小
            if event_type:
                he_size += 1  # EVT: node

            # 角色关系: 每个 motif 的实体类别组合
            cats = set()
            for e in entities:
                for prefix in ("HOST_", "PEER_", "PORT_", "P_", "SVC_"):
                    if e.startswith(prefix):
                        cats.add(prefix.rstrip("_"))
                        break
                else:
                    cats.add("ENTITY")
            role = "-".join(sorted(cats))
            seu_roles.add(role)

            he_size += len(entities)

        tokens_per_seu.append(len(seu_tokens))
        roles_per_seu.append(len(seu_roles))
        hyperedge_sizes.append(he_size)

    # Token entropy
    token_counter = Counter(all_tokens)
    n_total_tokens = len(all_tokens)
    n_unique_tokens = len(token_counter)
    if n_total_tokens > 0:
        probs = np.array(list(token_counter.values())) / n_total_tokens
        token_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
    else:
        token_entropy = 0.0

    # Average degree: 需要构建超图来计算
    # 简化: 使用 "每个令牌出现在多少个 SEU 中" 作为度的近似
    token_seu_count = defaultdict(set)
    for idx, rec in enumerate(records):
        for m in rec.get("motifs", []):
            for e in m.get("entities", []):
                token_seu_count[e].add(idx)
    if token_seu_count:
        avg_degree = np.mean([len(s) for s in token_seu_count.values()])
    else:
        avg_degree = 0.0

    # Token type percentages
    n_pf = token_type_counts.get("process_file", 0)
    n_ep = token_type_counts.get("endpoint_proto", 0)
    pct_pf = n_pf / n_total_tokens * 100 if n_total_tokens > 0 else 0
    pct_ep = n_ep / n_total_tokens * 100 if n_total_tokens > 0 else 0

    stats = {
        "dataset": name,
        "n_seus": n_seus,
        "avg_tokens_per_seu": float(np.mean(tokens_per_seu)),
        "avg_roles_per_seu": float(np.mean(roles_per_seu)),
        "avg_hyperedge_size": float(np.mean(hyperedge_sizes)),
        "pct_process_file": pct_pf,
        "pct_endpoint_proto": pct_ep,
        "n_unique_tokens": n_unique_tokens,
        "token_entropy": token_entropy,
        "avg_degree": float(avg_degree),
    }
    return stats


# ============================================================
# 汇总输出
# ============================================================

def print_summary_table(all_stats):
    """打印模态差异汇总表"""
    print()
    print("=" * 120)
    print("  Table X. Modality-dependent SEU and Hypergraph Statistics")
    print("=" * 120)

    header = (
        f"{'Dataset':<8s} {'Type':<12s} "
        f"{'N_SEU':>8s} "
        f"{'Tok/SEU':>8s} "
        f"{'Role/SEU':>9s} "
        f"{'HE_size':>8s} "
        f"{'%Proc/File':>11s} "
        f"{'%Endpt/Svc':>11s} "
        f"{'TokEnt':>8s} "
        f"{'AvgDeg':>8s}"
    )
    print(header)
    print("-" * 120)

    type_map = {"OPTC": "Host-log", "TCE5": "Host-log", "DAPT": "Network-flow"}

    for s in all_stats:
        ds = s["dataset"]
        dtype = type_map.get(ds, "?")
        print(
            f"{ds:<8s} {dtype:<12s} "
            f"{s['n_seus']:>8d} "
            f"{s['avg_tokens_per_seu']:>8.1f} "
            f"{s['avg_roles_per_seu']:>9.2f} "
            f"{s['avg_hyperedge_size']:>8.1f} "
            f"{s['pct_process_file']:>10.1f}% "
            f"{s['pct_endpoint_proto']:>10.1f}% "
            f"{s['token_entropy']:>8.2f} "
            f"{s['avg_degree']:>8.2f}"
        )

    print("=" * 120)
    print()
    print("  Notes:")
    print("  - Tok/SEU: average unique entity tokens per SEU")
    print("  - Role/SEU: average unique role-relation patterns per SEU")
    print("  - HE_size: average hyperedge size (entity nodes + event nodes)")
    print("  - %Proc/File: proportion of process/file tokens (host-log hex hash, DAPT HOST_*)")
    print("  - %Endpt/Svc: proportion of endpoint/protocol/service tokens (DAPT PEER_*, PORT_*, SVC_*)")
    print("  - TokEnt: Shannon entropy of token distribution (higher = more diverse)")
    print("  - AvgDeg: average number of SEUs each token participates in")
    print("=" * 120)


def save_csv(all_stats, out_path):
    """保存 CSV"""
    fieldnames = [
        "dataset", "n_seus",
        "avg_tokens_per_seu", "avg_roles_per_seu", "avg_hyperedge_size",
        "pct_process_file", "pct_endpoint_proto",
        "n_unique_tokens", "token_entropy", "avg_degree",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_stats)
    print(f"\n  Results saved: {out_path}")


# ============================================================
# 入口
# ============================================================

def run(out_dir=None):
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("  Modality-Dependent SEU & Hypergraph Statistics")
    print("  (Host-log vs Network-flow comparison)")
    print("=" * 60)
    print()

    all_stats = []
    for ds in ["OPTC", "TCE5", "DAPT"]:
        print(f"  Loading {ds} ...")
        records = load_dataset(ds)
        print(f"  {ds}: {len(records)} SEUs")
        stats = compute_modality_stats(records, ds)
        all_stats.append(stats)
        print(f"  {ds}: avg_tok/seu={stats['avg_tokens_per_seu']:.1f}, "
              f"avg_he_size={stats['avg_hyperedge_size']:.1f}, "
              f"tok_entropy={stats['token_entropy']:.2f}")
        print()

    print_summary_table(all_stats)
    save_csv(all_stats, os.path.join(out_dir, "modality_stats.csv"))

    return all_stats


if __name__ == "__main__":
    run()
