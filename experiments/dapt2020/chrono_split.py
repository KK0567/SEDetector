# -*- coding: utf-8 -*-
from pathlib import Path
"""
DAPT2020 Chronological Split
=============================
按 day 字段做时间划分，回应审稿人关于 temporal split 和泄漏风险的质疑.

DAPT2020 day 顺序: Monday → Tuesday → Wednesday → Thursday → Friday
划分方案:
  Train: Monday + Tuesday + Wednesday  (~54%)
  Val:   Thursday                       (~28%)
  Test:  Friday                         (~18%)

输出到 data_DAPT_chrono/ 目录，供后续超图构建与模型训练使用.
"""
import json
import os
import sys
import functools
from collections import Counter
ROOT = str(Path(__file__).resolve().parent.parent)  # project root

print = functools.partial(print, flush=True)

# ============================================================
# 配置
# ============================================================
DAPT_DIR = os.path.join(ROOT, "data_DAPT")
OUTPUT_DIR = os.path.join(ROOT, "data_DAPT_chrono")

DAY_ORDER = ["monday", "tuesday", "wednesday", "thursday", "friday"]
TRAIN_DAYS = {"monday", "tuesday", "wednesday"}
VAL_DAYS = {"thursday"}
TEST_DAYS = {"friday"}


def load_all_records():
    """加载所有 split 的 JSONL 数据"""
    all_records = []
    for split in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        path = os.path.join(DAPT_DIR, split)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_records.append(json.loads(line))
    return all_records


def split_records(records):
    """按 day 字段划分"""
    train, val, test = [], [], []
    for rec in records:
        day = rec.get("day", "").lower()
        if day in TRAIN_DAYS:
            train.append(rec)
        elif day in VAL_DAYS:
            val.append(rec)
        elif day in TEST_DAYS:
            test.append(rec)
        else:
            # 未知 day 放入 train
            train.append(rec)
    return train, val, test


def save_jsonl(records, path):
    """保存 JSONL"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def print_split_stats(name, records):
    """打印 split 统计信息"""
    n = len(records)
    days = Counter(r.get("day", "unknown") for r in records)
    labels = Counter(r.get("semantic_label", "unknown") for r in records)
    print(f"  {name}: {n:>7d} records")
    print(f"    Days:   {dict(sorted(days.items(), key=lambda x: DAY_ORDER.index(x[0]) if x[0] in DAY_ORDER else 99))}")
    print(f"    Labels: {dict(sorted(labels.items()))}")


def run():
    print("=" * 60)
    print("  DAPT2020 Chronological Split")
    print("=" * 60)
    print()

    # 加载
    print("  Loading all DAPT2020 records...")
    records = load_all_records()
    print(f"  Total: {len(records)} records")
    print()

    # 原始 split 统计
    print("  Original split distribution:")
    for split in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        path = os.path.join(DAPT_DIR, split)
        recs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    recs.append(json.loads(line))
        days = Counter(r.get("day", "?") for r in recs)
        print(f"    {split}: {len(recs):>7d}  days={dict(sorted(days.items()))}")
    print()

    # 时间划分
    print("  Chronological split:")
    print(f"    Train: {sorted(TRAIN_DAYS)} → Mon+Tue+Wed")
    print(f"    Val:   {sorted(VAL_DAYS)} → Thu")
    print(f"    Test:  {sorted(TEST_DAYS)} → Fri")
    print()

    train, val, test = split_records(records)
    total = len(train) + len(val) + len(test)

    print_split_stats("Train", train)
    print_split_stats("Val", val)
    print_split_stats("Test", test)
    print()

    print(f"  Split ratios:")
    print(f"    Train: {len(train)/total*100:.1f}%")
    print(f"    Val:   {len(val)/total*100:.1f}%")
    print(f"    Test:  {len(test)/total*100:.1f}%")
    print()

    # 保存
    save_jsonl(train, os.path.join(OUTPUT_DIR, "train.jsonl"))
    save_jsonl(val, os.path.join(OUTPUT_DIR, "val.jsonl"))
    save_jsonl(test, os.path.join(OUTPUT_DIR, "test.jsonl"))

    print(f"  Saved to: {OUTPUT_DIR}")
    print(f"    train.jsonl  ({len(train)} records)")
    print(f"    val.jsonl    ({len(val)} records)")
    print(f"    test.jsonl   ({len(test)} records)")
    print()

    return OUTPUT_DIR


if __name__ == "__main__":
    run()
