# -*- coding: utf-8 -*-
from pathlib import Path
"""
k-Anonymity-Inspired Analysis of SEU Equivalence Classes
SEU 等价类 k-匿名分析

回应 Reviewer #3: "Add theoretical leakage bounds or compare against
standard privacy frameworks such as differential privacy or k-anonymity."

原理:
  将每个 SEU 映射为准标识符 (QID), 统计共享相同 QID 的等价类大小 k.
  k 越大, 匿名性越强; k=1 (unique) 表示该 SEU 可被唯一识别.

QID 设置:
  1. Token-only:           frozenset(event_types)
  2. Token+Role+Context:   (event_types, entity_categories, role_pattern, context_bucket)
"""
import json
import os
import sys
import math
import functools
import csv
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, Optional

import numpy as np
ROOT = str(Path(__file__).resolve().parent.parent)  # project root

# 修复输出缓冲
print = functools.partial(print, flush=True)


# ============================================================
# 数据加载
# ============================================================

def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_dataset(name: str) -> List[Dict]:
    base = {
        "OPTC":  os.path.join(ROOT, "data_OPTC"),
        "TCE5":  os.path.join(ROOT, "data_TCE5"),
        "DAPT":  os.path.join(ROOT, "data_DAPT"),
    }[name]
    files = {
        "OPTC":  ["train_all.jsonl", "val_all.jsonl", "test_all.jsonl"],
        "TCE5":  ["train_all.jsonl", "val.jsonl", "test_all.jsonl"],
        "DAPT":  ["train.jsonl", "val.jsonl", "test.jsonl"],
    }[name]
    all_recs = []
    for fn in files:
        all_recs.extend(load_jsonl(os.path.join(base, fn)))
    return all_recs


# ============================================================
# QID 提取
# ============================================================

def _entity_category(token: str) -> str:
    """
    从实体令牌中提取类别前缀.
    DAPT: HOST_xxx -> HOST, PEER_xxx -> PEER, etc.
    OPTC/TCE5: 裸 hex hash -> ENTITY
    """
    for prefix in ("HOST_", "PEER_", "PORT_", "P_", "SVC_"):
        if token.startswith(prefix):
            return prefix.rstrip("_")
    return "ENTITY"


def _role_pattern(motif: Dict) -> str:
    """
    从 motif 中提取角色关系模式 (实体类别的有序组合).
    例如: "HOST-PEER-PORT" 表示该 motif 包含 HOST, PEER, PORT 三类实体.
    """
    cats = sorted(set(_entity_category(e) for e in motif.get("entities", [])))
    return "-".join(cats) if cats else "EMPTY"


def _context_bucket(rec: Dict) -> str:
    """
    上下文桶: 基于 motif 数 + 事件类型数 + 实体总数 的组合.
    用粗粒度桶避免 QID 过于精细.
    """
    n_motifs = len(rec.get("motifs", []))
    n_events = len(set(m.get("event_type", "") for m in rec.get("motifs", [])))
    n_entities = sum(len(m.get("entities", [])) for m in rec.get("motifs", []))

    # 粗粒度桶
    motif_b = "m1" if n_motifs == 1 else ("m2" if n_motifs == 2 else f"m{n_motifs}")
    event_b = f"e{min(n_events, 5)}"
    ent_b = "ent1" if n_entities <= 3 else ("ent2" if n_entities <= 6 else "ent3+")
    return f"{motif_b}_{event_b}_{ent_b}"


def qid_token_only(rec: Dict) -> tuple:
    """QID 设置 1: 仅事件类型集合"""
    events = frozenset(m.get("event_type", "") for m in rec.get("motifs", []))
    return (events,)


def qid_token_role_context(rec: Dict) -> tuple:
    """QID 设置 2: 事件类型 + 实体类别 + 角色模式 + 上下文桶"""
    events = frozenset(m.get("event_type", "") for m in rec.get("motifs", []))

    # 实体类别集合
    ent_cats = frozenset()
    for m in rec.get("motifs", []):
        ent_cats = ent_cats | frozenset(_entity_category(e) for e in m.get("entities", []))

    # 角色模式 (所有 motif 的 role pattern 的有序组合)
    roles = tuple(sorted(set(_role_pattern(m) for m in rec.get("motifs", []))))

    # 上下文桶
    ctx = _context_bucket(rec)

    return (events, ent_cats, roles, ctx)


# ============================================================
# k-匿名统计
# ============================================================

def compute_k_anonymity_stats(qids: List[tuple]) -> Dict:
    """
    给定所有 SEU 的 QID 列表, 计算等价类统计.
    """
    # 统计每个等价类的大小
    class_counts = Counter(qids)
    sizes = list(class_counts.values())
    n_total = len(qids)

    if not sizes:
        return {"error": "no data"}

    sizes_arr = np.array(sizes)
    n_classes = len(sizes)
    n_unique = sum(1 for s in sizes if s == 1)
    n_below5 = sum(1 for s in sizes if s < 5)
    n_below10 = sum(1 for s in sizes if s < 10)

    # Entropy: H = -sum(p * log2(p))
    probs = sizes_arr / n_total
    entropy = -np.sum(probs * np.log2(probs + 1e-12))

    stats = {
        "n_seus": n_total,
        "n_classes": n_classes,
        "min_k": int(sizes_arr.min()),
        "median_k": float(np.median(sizes_arr)),
        "mean_k": float(np.mean(sizes_arr)),
        "max_k": int(sizes_arr.max()),
        "pct_unique": n_unique / n_classes * 100,       # 占等价类的比例
        "pct_unique_of_total": n_unique / n_total * 100, # 占总 SEU 的比例
        "pct_k_lt_5": n_below5 / n_classes * 100,
        "pct_k_lt_10": n_below10 / n_classes * 100,
        "entropy": float(entropy),
        "max_entropy": float(np.log2(n_classes)) if n_classes > 1 else 0,
    }
    return stats


# ============================================================
# 运行分析
# ============================================================

def analyze_dataset(name: str, records: List[Dict]) -> List[Dict]:
    """对单个数据集做两种 QID 设置的 k-匿名分析."""
    results = []
    qid_settings = {
        "Token-only": qid_token_only,
        "Token+Role+Context": qid_token_role_context,
    }

    for setting_name, qid_fn in qid_settings.items():
        qids = [qid_fn(r) for r in records]
        stats = compute_k_anonymity_stats(qids)
        stats["dataset"] = name
        stats["qid_setting"] = setting_name
        results.append(stats)

        print(f"  [{setting_name}]")
        print(f"    SEUs: {stats['n_seus']}, 等价类: {stats['n_classes']}")
        print(f"    min_k={stats['min_k']}, median_k={stats['median_k']:.0f}, "
              f"mean_k={stats['mean_k']:.1f}, max_k={stats['max_k']}")
        print(f"    %unique(类)={stats['pct_unique']:.1f}%, "
              f"%unique(总)={stats['pct_unique_of_total']:.1f}%, "
              f"%k<5={stats['pct_k_lt_5']:.1f}%, "
              f"%k<10={stats['pct_k_lt_10']:.1f}%")
        print(f"    Entropy={stats['entropy']:.2f} "
              f"(max={stats['max_entropy']:.2f})")

    return results


# ============================================================
# 汇总输出
# ============================================================

def print_summary_table(all_results: List[Dict]):
    """打印汇总表"""
    print()
    print("=" * 110)
    print("  Table: k-Anonymity-Inspired Analysis of Released SEU Equivalence Classes")
    print("=" * 110)

    header = (f"{'Dataset':<8s} {'QID Setting':<22s} "
              f"{'N_SEU':>7s} {'N_cls':>7s} "
              f"{'min_k':>6s} {'med_k':>7s} {'mean_k':>8s} "
              f"{'%unique':>8s} {'%k<5':>7s} {'%k<10':>7s} "
              f"{'Entropy':>8s}")
    print(header)
    print("-" * 110)

    for r in all_results:
        if "error" in r:
            continue
        print(f"{r['dataset']:<8s} {r['qid_setting']:<22s} "
              f"{r['n_seus']:>7d} {r['n_classes']:>7d} "
              f"{r['min_k']:>6d} {r['median_k']:>7.0f} {r['mean_k']:>8.1f} "
              f"{r['pct_unique_of_total']:>7.1f}% {r['pct_k_lt_5']:>6.1f}% {r['pct_k_lt_10']:>6.1f}% "
              f"{r['entropy']:>8.2f}")

    print("=" * 110)

    # 分析说明
    print()
    print("  说明:")
    print("  - Token-only: QID = frozenset(event_types)")
    print("    攻击者仅知道 SEU 中出现的事件类型集合")
    print("  - Token+Role+Context: QID = (event_types, entity_categories, role_patterns, context_bucket)")
    print("    攻击者额外拥有实体类别、角色关系模式和结构上下文信息 (stronger attacker)")
    print("  - %unique(总): 占总 SEU 数的比例, k=1 意味着该 SEU 可被唯一识别")
    print("  - Entropy: 等价类分布的信息熵, 越高表示匿名性越好")
    print("=" * 110)


def save_csv(all_results: List[Dict], out_path: str):
    """保存 CSV"""
    fieldnames = [
        "dataset", "qid_setting", "n_seus", "n_classes",
        "min_k", "median_k", "mean_k", "max_k",
        "pct_unique", "pct_unique_of_total",
        "pct_k_lt_5", "pct_k_lt_10",
        "entropy", "max_entropy",
    ]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n结果已保存: {out_path}")


# ============================================================
# 入口
# ============================================================

def run_all(datasets=None, out_dir=None):
    if datasets is None:
        datasets = ["OPTC", "TCE5", "DAPT"]
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))

    all_results = []

    for ds in datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds}")
        print(f"{'='*60}")
        records = load_dataset(ds)
        print(f"  加载 {len(records)} 条 SEU")
        results = analyze_dataset(ds, records)
        all_results.extend(results)

    print_summary_table(all_results)
    save_csv(all_results, os.path.join(out_dir, "k_anonymity_results.csv"))
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="k-Anonymity Analysis")
    parser.add_argument("--optc", action="store_true")
    parser.add_argument("--tce5", action="store_true")
    parser.add_argument("--dapt", action="store_true")
    args = parser.parse_args()

    datasets = []
    if args.optc:
        datasets.append("OPTC")
    if args.tce5:
        datasets.append("TCE5")
    if args.dapt:
        datasets.append("DAPT")
    if not datasets:
        datasets = ["OPTC", "TCE5", "DAPT"]

    run_all(datasets=datasets)
