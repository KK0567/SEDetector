# -*- coding: utf-8 -*-
"""
Raw+Hypergraph v2 — 真正的 Raw+HG 消融变体
==============================================
设计原则:
  1. 保留与主模型完全相同的超图结构 (incidence matrix H)
  2. 用原始事件/会话字段构造特征 (event_types, techniques, counts)
  3. 不经过 SEU abstraction (移除 semantic_score)
  4. 不使用随机噪声
  5. 输出维度、训练预算、分类头与主模型完全一致

特征差异:
  Main  edge_feats = [semantic_score, total_count, num_entities, num_events,
                      hash_bow(event_types, 128), hash_bow(techniques, 128)]
                      → 4 + 128 + 128 = 260 dims

  RawHG edge_feats = [total_count, num_entities, num_events,
                      hash_bow(event_types, 128), hash_bow(techniques, 128)]
                      → 3 + 128 + 128 = 259 dims  (去掉 semantic_score)

  Node features 保持完全相同 (type onehot + degree stats + token hash).

运行:
  python run_rawhg_v2.py --dataset OPTC --seed 2021
  python run_rawhg_v2.py --dataset TCE5 --seed 2021
  python run_rawhg_v2.py --dataset DAPT --seed 2021
"""
import sys
import os
import argparse

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = ROOT

DATASETS = {
    "OPTC": {
        "progress_dir": os.path.join(PROJECT_ROOT, "progress_OPTC"),
        "run_script": "run_OPTC.py",
        "data_prefix": "../data_OPTC",
        "args": {
            "emb_dim": "256", "epochs": "8", "episodes_per_epoch": "120",
            "lr": "0.0005", "k_hop": "2", "proto_k": "800", "proto_m": "3",
            "proto_interval": "2", "proto_ema": "0.9",
            "class_weight_mode": "inv_sqrt", "logit_adj": "0.08",
            "focal_gamma": "1.0", "hard_pairs": "",
            "min_quota_labels": "", "hub_degree_skip": "0",
            "tau": "0.07", "head_tau": "0.05",
            "kd_alpha": "0.5", "kd_T": "2.0",
            "mode": "head_kd",
        },
    },
    "TCE5": {
        "progress_dir": os.path.join(PROJECT_ROOT, "progress_TCE5"),
        "run_script": "run_TCE5.py",
        "data_prefix": "../data_TCE5",
        "args": {
            "emb_dim": "256", "epochs": "50", "episodes_per_epoch": "120",
            "lr": "0.0005", "k_hop": "1", "proto_k": "300", "proto_m": "3",
            "proto_interval": "2", "proto_ema": "0.9",
            "class_weight_mode": "inv_sqrt", "logit_adj": "0.08",
            "focal_gamma": "1.0", "hard_pairs": "",
            "min_quota_labels": "", "hub_degree_skip": "3",
            "tau": "0.07", "head_tau": "0.05",
            "kd_alpha": "0.3", "kd_T": "3.0",
            "mode": "head_kd",
        },
    },
    "DAPT": {
        "progress_dir": os.path.join(PROJECT_ROOT, "progress_DAPT"),
        "run_script": "run_DAPT.py",
        "data_prefix": "../data_DAPT",
        "args": {
            "emb_dim": "256", "epochs": "50", "episodes_per_epoch": "256",
            "lr": "0.0003", "k_hop": "2", "proto_k": "1024", "proto_m": "3",
            "proto_interval": "2", "proto_ema": "0.9",
            "class_weight_mode": "effective", "logit_adj": "0.05",
            "focal_gamma": "1.2",
            "hard_pairs": "LateralMovement:Exfiltration,Exfiltration:LateralMovement",
            "min_quota_labels": "Exfiltration:8,LateralMovement:8",
            "hub_degree_skip": "0",
            "tau": "0.05", "head_tau": "0.05",
            "kd_alpha": "0.2", "kd_T": "3.0",
            "mode": "head_kd",
        },
    },
}


# ============================================================
# 解析参数
# ============================================================
parser = argparse.ArgumentParser(description="Raw+HG v2 Ablation")
parser.add_argument("--dataset", required=True, choices=["OPTC", "TCE5", "DAPT"])
parser.add_argument("--seed", type=int, default=2021)
cli_args = parser.parse_args()

ds_name = cli_args.dataset
seed = cli_args.seed
ds_cfg = DATASETS[ds_name]
progress_dir = ds_cfg["progress_dir"]
run_script = ds_cfg["run_script"]
module_name = run_script.replace(".py", "")

# ============================================================
# sys.path 设置
# ============================================================
sys.path.insert(0, progress_dir)
os.chdir(progress_dir)

# ============================================================
# Monkey-patch: 移除 semantic_score (Raw+HG 核心修改)
# ============================================================
import utils as _utils

_original_load = _utils.load_global_hypergraph_from_json

def _raw_load(*args, **kwargs):
    """Load hypergraph then strip semantic_score from edge features."""
    g = _original_load(*args, **kwargs)
    # edge_feats[:, 0] = semantic_score → 删除该列
    g.edge_feats = g.edge_feats[:, 1:]
    return g

_utils.load_global_hypergraph_from_json = _raw_load

# ============================================================
# 构造 sys.argv (与主模型完全相同的训练配置)
# ============================================================
out_dir = f"./outputs_{ds_name}_abl_RawHG_v2"

sys.argv = [os.path.join(progress_dir, run_script)]
sys.argv += [
    "--train_hg", f"{ds_cfg['data_prefix']}/Hyper_train.json",
    "--val_hg",    f"{ds_cfg['data_prefix']}/Hyper_val.json",
    "--test_hg",   f"{ds_cfg['data_prefix']}/Hyper_test.json",
]

# 数据集参数 + seed + 输出目录
for k, v in ds_cfg["args"].items():
    sys.argv += [f"--{k}", str(v)]
sys.argv += ["--seed", str(seed)]
sys.argv += ["--out_dir", out_dir]

# ============================================================
# 启动训练
# ============================================================
print(f"\n{'='*60}")
print(f"  Raw+HG v2  |  Dataset: {ds_name}  |  Seed: {seed}")
print(f"  Output: {out_dir}")
print(f"  Feature change: remove semantic_score from edge_feats")
print(f"{'='*60}\n")

from importlib import import_module
mod = import_module(module_name)
mod.main()
