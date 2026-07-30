# -*- coding: utf-8 -*-
from pathlib import Path
"""
SEDetector 三数据集 5-seed 统一实验脚本
=======================================
修复 DAPT 跨种子不稳定问题（原型 EMA、重建频率、子图采样等）。

在 IDE 中点击 Run 即可依次运行 OPTC → TCE5 → DAPT，每个数据集 5 个种子。

主要修改 (vs 之前的 run_DAPT_5seeds.py):
  - proto_ema:       0.5 → 0.85  (教师信号更稳定)
  - proto_interval:  1   → 3     (减少重建频率)
  - episodes_per_epoch: 256 → 160 (减少随机累积)
  - hub_degree_skip: 0   → 2     (减少高度节点子图方差)
  - proto_m:         5   → 3     (更少的聚类中心, 更稳定)
"""

import subprocess
import sys
import os
import time
from datetime import datetime
ROOT = str(Path(__file__).resolve().parent.parent)  # project root

# ============================================================
# 通用配置
# ============================================================
PYTHON = sys.executable
SEEDS = [2021, 2022, 2023, 2024, 2025]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, "all_3datasets_log.txt")


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = "[{}] {}".format(ts, msg)
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_seeds(dataset_label, progress_dir, base_args):
    """对指定数据集依次运行 5 个种子"""
    log("")
    log("=" * 60)
    log("{}  5-seed experiments".format(dataset_label))
    log("Seeds: {}".format(SEEDS))
    log("Working dir: {}".format(progress_dir))
    log("=" * 60)

    ok, fail = 0, 0
    for i, seed in enumerate(SEEDS, 1):
        cmd = [PYTHON] + base_args + ["--seed", str(seed)]
        log("[{}/5] {} seed={} STARTING".format(i, dataset_label, seed))

        t0 = time.time()
        try:
            result = subprocess.run(cmd, cwd=progress_dir)
            elapsed = time.time() - t0

            if result.returncode == 0:
                log("[{}/5] {} seed={} OK ({:.1f} min)".format(
                    i, dataset_label, seed, elapsed / 60))
                ok += 1
            else:
                log("[{}/5] {} seed={} FAIL exit={} ({:.1f} min)".format(
                    i, dataset_label, seed, result.returncode, elapsed / 60))
                fail += 1
        except Exception as e:
            elapsed = time.time() - t0
            log("[{}/5] {} seed={} EXCEPTION: {} ({:.1f} min)".format(
                i, dataset_label, seed, str(e), elapsed / 60))
            fail += 1

    log("{} done: OK={}  FAIL={}".format(dataset_label, ok, fail))
    return ok, fail


# ============================================================
# OPTC 配置 (稳定, 快速)
# ============================================================
OPTC_DIR = os.path.join(ROOT, "progress_OPTC")
OPTC_ARGS = [
    "run_OPTC.py",
    "--mode",              "head_kd",
    "--emb_dim",           "256",
    "--num_layers",        "2",
    "--dropout",           "0.2",
    "--epochs",            "8",
    "--episodes_per_epoch","120",
    "--batch_size",        "128",
    "--lr",                "0.0005",
    "--weight_decay",      "0.0001",
    "--proto_k",           "800",
    "--proto_bs",          "128",
    "--proto_m",           "3",
    "--proto_reduce",      "logsumexp",
    "--kmeans_iters",      "20",
    "--proto_interval",    "2",
    "--proto_ema",         "0.9",
    "--proto_source",      "trainval",
    "--tau",               "0.07",
    "--k_hop",             "2",
    "--max_edges",         "48",
    "--max_nodes",         "192",
    "--max_members_per_edge", "128",
    "--max_hes_per_node",  "128",
    "--hub_degree_skip",   "0",
    "--use_amp",
    "--grad_clip",         "1.0",
    "--out_dir",           "./outputs_OPTC",
    "--warmup_ratio",      "0.1",
    "--class_weight_mode", "inv_sqrt",
    "--logit_adj",         "0.08",
    "--logit_adj_mode",    "sub",
    "--focal_gamma",       "1.0",
    "--ladj_ramp_epochs",  "5",
    "--focal_ramp_epochs", "5",
    "--head_tau",          "0.05",
    "--head_wd",           "0.0",
    "--kd_alpha",          "0.5",
    "--kd_T",              "2.0",
    "--hard_pairs",        "",
    "--pair_margin",       "2.0",
    "--pair_weight",       "0.0",
    "--supcon_w",          "0.0",
    "--supcon_temp",       "0.2",
    "--train_eval_max",    "5000",
    "--anom_tau",          "0.5",
    "--auto_tau",
    "--tau_grid",          "1001",
    "--min_per_class",     "1",
    "--min_quota_labels",  "",
    "--train_hg",          "../data_OPTC/Hyper_train.json",
    "--val_hg",            "../data_OPTC/Hyper_val.json",
    "--test_hg",           "../data_OPTC/Hyper_test.json",
]

# ============================================================
# TCE5 配置 (稳定)
# ============================================================
TCE5_DIR = os.path.join(ROOT, "progress_TCE5")
TCE5_ARGS = [
    "run_TCE5.py",
    "--mode",              "head_kd",
    "--emb_dim",           "256",
    "--num_layers",        "2",
    "--dropout",           "0.2",
    "--epochs",            "50",
    "--episodes_per_epoch","120",
    "--batch_size",        "128",
    "--lr",                "0.0005",
    "--weight_decay",      "0.0001",
    "--proto_k",           "300",
    "--proto_bs",          "128",
    "--proto_m",           "3",
    "--proto_reduce",      "logsumexp",
    "--kmeans_iters",      "20",
    "--proto_interval",    "2",
    "--proto_ema",         "0.9",
    "--proto_source",      "trainval",
    "--tau",               "0.07",
    "--k_hop",             "1",
    "--max_edges",         "48",
    "--max_nodes",         "192",
    "--max_members_per_edge", "128",
    "--max_hes_per_node",  "128",
    "--hub_degree_skip",   "3",
    "--use_amp",
    "--grad_clip",         "1.0",
    "--out_dir",           "./outputs_TCE5",
    "--warmup_ratio",      "0.1",
    "--class_weight_mode", "inv_sqrt",
    "--logit_adj",         "0.08",
    "--logit_adj_mode",    "sub",
    "--focal_gamma",       "1.0",
    "--ladj_ramp_epochs",  "10",
    "--focal_ramp_epochs", "10",
    "--head_tau",          "0.05",
    "--head_wd",           "0.0",
    "--kd_alpha",          "0.3",
    "--kd_T",              "3.0",
    "--hard_pairs",        "",
    "--pair_margin",       "2.0",
    "--pair_weight",       "0.0",
    "--supcon_w",          "0.0",
    "--supcon_temp",       "0.2",
    "--train_eval_max",    "5000",
    "--anom_tau",          "0.5",
    "--auto_tau",
    "--tau_grid",          "1001",
    "--min_per_class",     "1",
    "--min_quota_labels",  "",
    "--train_hg",          "../data_TCE5/Hyper_train.json",
    "--val_hg",            "../data_TCE5/Hyper_val.json",
    "--test_hg",           "../data_TCE5/Hyper_test.json",
]

# ============================================================
# DAPT2020 配置 (修复稳定性)
# ============================================================
# 关键修改 (vs 之前的 run_DAPT_5seeds.py):
#   proto_ema:          0.5 → 0.85   教师信号更平滑
#   proto_interval:     1   → 3      减少重建频率
#   episodes_per_epoch: 256 → 160    减少每轮随机累积
#   hub_degree_skip:    0   → 2      减少高度节点子图方差
#   proto_m:            5   → 3      更少的聚类中心
#   kmeans_iters:       20  → 30     更充分的聚类收敛
# ============================================================
DAPT_DIR = os.path.join(ROOT, "progress_DAPT")
DAPT_ARGS = [
    "run_DAPT.py",
    "--mode",              "head_kd",
    "--emb_dim",           "256",
    "--num_layers",        "2",
    "--dropout",           "0.2",
    "--epochs",            "50",
    "--episodes_per_epoch","160",
    "--batch_size",        "128",
    "--lr",                "0.0003",
    "--weight_decay",      "0.0001",
    "--proto_k",           "1024",
    "--proto_bs",          "128",
    "--proto_m",           "3",
    "--proto_reduce",      "logsumexp",
    "--kmeans_iters",      "30",
    "--proto_interval",    "3",
    "--proto_ema",         "0.85",
    "--proto_source",      "trainval",
    "--tau",               "0.05",
    "--k_hop",             "2",
    "--max_edges",         "48",
    "--max_nodes",         "192",
    "--max_members_per_edge", "128",
    "--max_hes_per_node",  "128",
    "--hub_degree_skip",   "2",
    "--use_amp",
    "--grad_clip",         "1.0",
    "--out_dir",           "./outputs_DAPT",
    "--warmup_ratio",      "0.1",
    "--class_weight_mode", "effective",
    "--logit_adj",         "0.05",
    "--logit_adj_mode",    "sub",
    "--focal_gamma",       "1.2",
    "--ladj_ramp_epochs",  "10",
    "--focal_ramp_epochs", "10",
    "--head_tau",          "0.05",
    "--head_wd",           "0.0",
    "--kd_alpha",          "0.2",
    "--kd_T",              "3.0",
    "--hard_pairs",        "LateralMovement:Exfiltration,Exfiltration:LateralMovement",
    "--pair_margin",       "2.0",
    "--pair_weight",       "0.0",
    "--supcon_w",          "0.0",
    "--supcon_temp",       "0.2",
    "--train_eval_max",    "5000",
    "--anom_tau",          "0.5",
    "--auto_tau",
    "--tau_grid",          "1001",
    "--min_per_class",     "1",
    "--min_quota_labels",  "Exfiltration:6,CommandAndControl:4,Discovery:4",
    "--train_hg",          "../data_DAPT/Hyper_train.json",
    "--val_hg",            "../data_DAPT/Hyper_val.json",
    "--test_hg",           "../data_DAPT/Hyper_test.json",
]


def main():
    log("=" * 70)
    log("  SEDetector  3-dataset  5-seed  unified runner")
    log("  Total: 3 datasets x 5 seeds = 15 experiments")
    log("=" * 70)

    t_start = time.time()
    total_ok, total_fail = 0, 0

    # --- OPTC (fastest, ~2min/seed) ---
    ok, fail = run_seeds("OPTC", OPTC_DIR, OPTC_ARGS)
    total_ok += ok
    total_fail += fail

    # --- TCE5 (medium, ~15min/seed) ---
    ok, fail = run_seeds("TCE5", TCE5_DIR, TCE5_ARGS)
    total_ok += ok
    total_fail += fail

    # --- DAPT2020 (slowest, ~30min/seed) ---
    ok, fail = run_seeds("DAPT2020", DAPT_DIR, DAPT_ARGS)
    total_ok += ok
    total_fail += fail

    total_time = time.time() - t_start

    log("")
    log("=" * 70)
    log("  ALL DONE")
    log("  Total: 15  |  OK: {}  |  FAIL: {}".format(total_ok, total_fail))
    log("  Time:  {:.1f} min ({:.1f} hours)".format(total_time / 60, total_time / 3600))
    log("=" * 70)

    if total_fail == 0:
        log("")
        log("All experiments succeeded!")
        log("Run aggregate_results.py to generate summary tables.")


if __name__ == "__main__":
    main()
