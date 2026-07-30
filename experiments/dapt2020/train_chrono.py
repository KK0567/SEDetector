# -*- coding: utf-8 -*-
from pathlib import Path
"""
Train SEDetector + SEU+MLP on DAPT2020 Chronological Split
============================================================
使用 subprocess 依次调用:
  1. SEDetector (full model, head_kd mode)
  2. SEU+MLP (ablation, no hypergraph message passing)

结果输出到 dapt2020/results/ 目录.

预计时间:
  - SEDetector: ~2-3 小时
  - SEU+MLP:    ~30-60 分钟
"""
import os
import sys
import subprocess
import functools
import time
import json
ROOT = str(Path(__file__).resolve().parent.parent)  # project root

print = functools.partial(print, flush=True)

# ============================================================
# 路径配置
# ============================================================
HERE = os.path.dirname(os.path.abspath(__file__))
PROGRESS_DIR = os.path.join(ROOT, "progress_DAPT")
SEU_MLP_DIR = os.path.join(ROOT, "experiments", "ablation", "Abl_SEU_MLP")
DATA_DIR = os.path.join(ROOT, "data_DAPT_chrono")
RESULTS_DIR = os.path.join(HERE, "results")
PYTHON = "python"

# 超图路径 (绝对路径，避免 cwd 问题)
TRAIN_HG = os.path.join(DATA_DIR, "Hyper_train.json")
VAL_HG = os.path.join(DATA_DIR, "Hyper_val.json")
TEST_HG = os.path.join(DATA_DIR, "Hyper_test.json")


def run_sedetector():
    """运行 SEDetector (full model) on chrono split"""
    print("  [1/2] SEDetector (full model, head_kd)")
    print(f"    Train HG: {TRAIN_HG}")
    print(f"    Val HG:   {VAL_HG}")
    print(f"    Test HG:  {TEST_HG}")

    out_dir = os.path.join(RESULTS_DIR, "SEDetector_chrono")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        PYTHON,
        os.path.join(PROGRESS_DIR, "run_DAPT.py"),
        "--train_hg", TRAIN_HG,
        "--val_hg", VAL_HG,
        "--test_hg", TEST_HG,
        "--mode", "head_kd",
        "--out_dir", out_dir,
        "--seed", "32",
    ]

    print(f"    Command: {' '.join(cmd)}")
    print(f"    Working dir: {PROGRESS_DIR}")
    print()

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=PROGRESS_DIR)
    elapsed = time.time() - t0

    if proc.returncode == 0:
        print(f"    SEDetector completed in {elapsed/60:.1f} min")
    else:
        print(f"    SEDetector FAILED (exit code {proc.returncode})")
    print()
    return proc.returncode


def run_seu_mlp():
    """运行 SEU+MLP (ablation) on chrono split"""
    print("  [2/2] SEU+MLP (no hypergraph message passing)")

    out_dir = os.path.join(RESULTS_DIR, "SEU_MLP_chrono")
    os.makedirs(out_dir, exist_ok=True)

    # 创建临时启动脚本
    launcher = os.path.join(HERE, "_tmp_seu_mlp_launcher.py")
    launcher_code = f'''# -*- coding: utf-8 -*-
"""Auto-generated launcher for SEU+MLP on chrono split"""
import sys, os

ABLA_DIR = r"{SEU_MLP_DIR}"
PROGRESS_DIR = r"{PROGRESS_DIR}"
sys.path.insert(0, PROGRESS_DIR)
os.chdir(PROGRESS_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F

class _AblationEncoder(nn.Module):
    """Ablation: remove HypergraphMessagePassing, use MLP on center edge only."""
    def __init__(self, node_feat_dim, edge_feat_dim, emb_dim=128,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )
        self.out_ln = nn.LayerNorm(emb_dim)

    def forward(self, H, node_feats, edge_feats):
        center_edge = edge_feats[:, 0, :]
        z = self.edge_encoder(center_edge)
        z = self.out_ln(z)
        z = F.normalize(z, p=2, dim=-1)
        return z

import model as _model_mod
_model_mod.HyperEdgeEncoder = _AblationEncoder

sys.argv = [os.path.join(PROGRESS_DIR, "run_DAPT.py")]
sys.argv += [
    "--train_hg", r"{TRAIN_HG}",
    "--val_hg", r"{VAL_HG}",
    "--test_hg", r"{TEST_HG}",
    "--mode", "head_kd",
    "--out_dir", r"{out_dir}",
    "--seed", "32",
]

from run_DAPT import main
main()
'''
    with open(launcher, "w", encoding="utf-8") as f:
        f.write(launcher_code)

    print(f"    Launcher: {launcher}")
    print(f"    Working dir: {PROGRESS_DIR}")
    print()

    t0 = time.time()
    proc = subprocess.run([PYTHON, launcher], cwd=PROGRESS_DIR)
    elapsed = time.time() - t0

    # 清理临时文件
    try:
        os.remove(launcher)
    except Exception:
        pass

    if proc.returncode == 0:
        print(f"    SEU+MLP completed in {elapsed/60:.1f} min")
    else:
        print(f"    SEU+MLP FAILED (exit code {proc.returncode})")
    print()
    return proc.returncode


def collect_results():
    """收集训练结果，打印汇总表"""
    print("=" * 90)
    print("  Table X. Robustness under chronological split on DAPT2020")
    print("=" * 90)
    print()

    header = f"{'Split':<12s} {'Method':<15s} {'Macro-F1':>10s} {'PR-AUC':>10s} {'ROC-AUC':>10s} {'Accuracy':>10s}"
    print(header)
    print("-" * 90)

    # 从结果目录中搜索 metrics 文件
    for model_name, sub_dir in [("SEDetector", "SEDetector_chrono"), ("SEU+MLP", "SEU_MLP_chrono")]:
        model_dir = os.path.join(RESULTS_DIR, sub_dir)
        if not os.path.isdir(model_dir):
            print(f"  {sub_dir}: not found")
            continue

        # 查找最新的 test_metrics.json 或类似文件
        found = False
        for root, dirs, files in os.walk(model_dir):
            for fn in files:
                if "test" in fn.lower() and fn.endswith(".json"):
                    fp = os.path.join(root, fn)
                    try:
                        with open(fp, "r", encoding="utf-8") as f:
                            metrics = json.load(f)
                        mf1 = metrics.get("macro_f1", metrics.get("macro-F1", -1))
                        prauc = metrics.get("pr_auc", metrics.get("PR-AUC", -1))
                        roauc = metrics.get("roc_auc", metrics.get("ROC-AUC", -1))
                        acc = metrics.get("accuracy", metrics.get("acc", -1))
                        split_label = "Chrono"
                        print(f"{split_label:<12s} {model_name:<15s} {mf1:>10.4f} {prauc:>10.4f} {roauc:>10.4f} {acc:>10.4f}")
                        found = True
                    except Exception:
                        pass

        if not found:
            print(f"  {model_name}: results not found in {model_dir}")
            print(f"    (check sub-directories for metrics files)")

    print()
    print("=" * 90)

    # 对比原始 random split 结果 (如果有)
    print()
    print("  Note: Compare with original random split results in the main paper.")
    print("  Expected: chronological split shows moderate decrease,")
    print("  but SEDetector still outperforms SEU+MLP baseline.")
    print("=" * 90)


def run():
    print("=" * 60)
    print("  Train Models on DAPT2020 Chronological Split")
    print("=" * 60)
    print()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 检查超图文件是否存在
    for hg in [TRAIN_HG, VAL_HG, TEST_HG]:
        if not os.path.isfile(hg):
            print(f"  ERROR: Hypergraph not found: {hg}")
            print(f"  Please run build_hyper_chrono.py first.")
            return

    print(f"  Hypergraphs found in: {DATA_DIR}")
    print(f"  Results will be saved to: {RESULTS_DIR}")
    print()

    # 运行 SEDetector
    rc1 = run_sedetector()

    # 运行 SEU+MLP
    rc2 = run_seu_mlp()

    # 汇总结果
    collect_results()

    return rc1, rc2


if __name__ == "__main__":
    run()
