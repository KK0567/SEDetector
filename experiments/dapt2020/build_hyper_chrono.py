# -*- coding: utf-8 -*-
from pathlib import Path
"""
Build Hypergraphs for DAPT2020 Chronological Split
===================================================
读取 data_DAPT_chrono/ 下的 train/val/test.jsonl,
构建超图并保存 Hyper_*.json / Hyper_*.pt
"""
import json
import os
import sys
import functools

import torch
ROOT = str(Path(__file__).resolve().parent.parent)  # project root

print = functools.partial(print, flush=True)

# 复用已有的超图构建函数
BASE_DIR = ROOT
sys.path.insert(0, BASE_DIR)

# 导入构建函数 (1.2Build_train.py 不是一个合法 module name, 用 importlib)
import importlib.util

_build_spec = importlib.util.spec_from_file_location(
    "build_train", os.path.join(BASE_DIR, "1.2Build_train.py")
)
_build_mod = importlib.util.module_from_spec(_build_spec)
_build_spec.loader.exec_module(_build_mod)

build_hypergraph = _build_mod.build_hypergraph_from_semantic_slices
save_hypergraph_json = _build_mod.save_hypergraph_json

# ============================================================
# 配置
# ============================================================
DATA_DIR = os.path.join(ROOT, "data_DAPT_chrono")


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_one(split_name, in_path, out_dir):
    """构建单个 split 的超图"""
    print(f"  [{split_name}] Loading {in_path} ...")
    samples = load_jsonl(in_path)
    print(f"  [{split_name}] {len(samples)} samples loaded")

    # 创建 label_map: 使用 DAPT 的固定 5 类映射
    # 必须与原始训练保持一致
    all_labels = sorted({s.get("semantic_label", "UNK") for s in samples})
    label_map = {lab: i for i, lab in enumerate(all_labels)}
    print(f"  [{split_name}] label_map = {label_map}")

    # 构建超图
    data_pt, hyperedge_records, token2nid, nid2type = build_hypergraph(samples, label_map)

    # 保存 .pt
    pt_path = os.path.join(out_dir, f"Hyper_{split_name}.pt")
    torch.save(data_pt, pt_path)
    print(f"  [{split_name}] Saved {pt_path}")

    # 保存 .json
    json_path = os.path.join(out_dir, f"Hyper_{split_name}.json")
    save_hypergraph_json(json_path, samples, token2nid, nid2type, hyperedge_records)
    print(f"  [{split_name}] Saved {json_path}")

    print(f"  [{split_name}] nodes={data_pt['num_nodes']}, hyperedges={data_pt['num_hyperedges']}")

    return label_map


def run():
    print("=" * 60)
    print("  Build Hypergraphs for DAPT2020 Chrono Split")
    print("=" * 60)
    print()

    out_dir = DATA_DIR
    os.makedirs(out_dir, exist_ok=True)

    # 构建 train 时记录 label_map, val/test 使用相同的映射
    # 先扫描所有数据获取统一 label_map
    all_samples = []
    for split in ["train", "val", "test"]:
        path = os.path.join(DATA_DIR, f"{split}.jsonl")
        all_samples.extend(load_jsonl(path))

    unified_label_map = {
        lab: i for i, lab in enumerate(sorted({s.get("semantic_label", "UNK") for s in all_samples}))
    }
    print(f"  Unified label_map: {unified_label_map}")
    print()

    # 重新构建函数: 使用统一 label_map
    for split in ["train", "val", "test"]:
        in_path = os.path.join(DATA_DIR, f"{split}.jsonl")
        samples = load_jsonl(in_path)

        data_pt, hyperedge_records, token2nid, nid2type = build_hypergraph(samples, unified_label_map)

        pt_path = os.path.join(out_dir, f"Hyper_{split}.pt")
        torch.save(data_pt, pt_path)

        json_path = os.path.join(out_dir, f"Hyper_{split}.json")
        save_hypergraph_json(json_path, samples, token2nid, nid2type, hyperedge_records)

        print(f"  [{split:>5s}] {len(samples):>7d} samples, "
              f"nodes={data_pt['num_nodes']:>7d}, "
              f"hyperedges={data_pt['num_hyperedges']:>7d}")

    print()
    print(f"  All hypergraphs saved to: {out_dir}")
    print()


if __name__ == "__main__":
    run()
