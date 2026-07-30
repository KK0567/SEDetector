# -*- coding: utf-8 -*-
from pathlib import Path
"""
SEDetector 消融实验 一键运行器
==============================
依次运行全部 5 个消融变体 x 3 个数据集 x 5 个种子 = 75 次实验。
也可以只跑指定的变体或数据集。

用法:
  python run_all.py                    # 跑全部
  python run_all.py --variants Abl_SEU_MLP Abl_NoKD   # 只跑两个变体
  python run_all.py --datasets OPTC                   # 只跑 OPTC
  python run_all.py --dry_run                         # 预览不执行
"""

import subprocess
import sys
import os
import time
import argparse
from datetime import datetime
ROOT = str(Path(__file__).resolve().parent.parent)  # project root

# ============================================================
# 配置
# ============================================================
SEED_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
SEEDS = [2021, 2022, 2023, 2024, 2025]

ALL_VARIANTS = [
    "Abl_SEU_MLP",
    "Abl_SEU_GCN",
    "Abl_RawHG",
    "Abl_NoKD",
    "Abl_NoSEU",
]

ALL_DATASETS = ["OPTC", "TCE5", "DAPT"]

DATASET_PROGRESS = {
    "OPTC": os.path.join(ROOT, "progress_OPTC"),
    "TCE5": os.path.join(ROOT, "progress_TCE5"),
    "DAPT": os.path.join(ROOT, "progress_DAPT"),
}

LOG_FILE = os.path.join(SEED_DIR, "run_all_log.txt")


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = "[{}] {}".format(ts, msg)
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_experiment(variant, dataset, seed, dry_run=False):
    """运行单次实验"""
    script = os.path.join(SEED_DIR, variant, "run_{}.py".format(dataset))
    cwd = DATASET_PROGRESS.get(dataset, SEED_DIR)

    if not os.path.isfile(script):
        log("  [SKIP] {} not found".format(script))
        return False

    cmd = [PYTHON, script, "--seed", str(seed)]

    if dry_run:
        log("  [DRY] {} / {} / seed={}".format(variant, dataset, seed))
        log("         cmd: {}".format(" ".join(cmd)))
        log("         cwd: {}".format(cwd))
        return True

    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=cwd)
        elapsed = time.time() - t0
        if result.returncode == 0:
            log("  [OK]   {} / {} / seed={}  ({:.1f} min)".format(
                variant, dataset, seed, elapsed / 60))
            return True
        else:
            log("  [FAIL] {} / {} / seed={}  exit={}  ({:.1f} min)".format(
                variant, dataset, seed, result.returncode, elapsed / 60))
            return False
    except Exception as e:
        elapsed = time.time() - t0
        log("  [ERR]  {} / {} / seed={}  {}  ({:.1f} min)".format(
            variant, dataset, seed, str(e), elapsed / 60))
        return False


def main():
    parser = argparse.ArgumentParser(description="SEDetector Ablation Runner")
    parser.add_argument("--variants", nargs="+", default=ALL_VARIANTS,
                        help="Which variants to run (default: all)")
    parser.add_argument("--datasets", nargs="+", default=ALL_DATASETS,
                        help="Which datasets to run (default: all)")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                        help="Which seeds to run (default: 2021-2025)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    total = len(args.variants) * len(args.datasets) * len(args.seeds)

    log("=" * 70)
    log("  SEDetector Ablation Experiment Runner")
    log("=" * 70)
    log("  Variants: {} ({})".format(args.variants, len(args.variants)))
    log("  Datasets: {} ({})".format(args.datasets, len(args.datasets)))
    log("  Seeds:    {} ({})".format(args.seeds, len(args.seeds)))
    log("  Total experiments: {}".format(total))
    log("  Dry run: {}".format(args.dry_run))
    log("=" * 70)

    ok_count = 0
    fail_count = 0
    exp_num = 0
    t_start = time.time()

    for variant in args.variants:
        for dataset in args.datasets:
            log("\n--- {} / {} ---".format(variant, dataset))
            for seed in args.seeds:
                exp_num += 1
                log("[{}/{}] {} / {} / seed={}".format(
                    exp_num, total, variant, dataset, seed))

                success = run_experiment(variant, dataset, seed, args.dry_run)
                if success:
                    ok_count += 1
                else:
                    fail_count += 1

    total_time = time.time() - t_start

    log("\n" + "=" * 70)
    log("  ALL DONE")
    log("  Total: {}  |  OK: {}  |  FAIL: {}".format(total, ok_count, fail_count))
    log("  Time:  {:.1f} min".format(total_time / 60))
    log("=" * 70)

    # 提示运行汇总脚本
    if not args.dry_run and fail_count == 0:
        log("\nAll experiments succeeded! Run aggregate_results.py to summarize.")
        log("  python {}".format(os.path.join(SEED_DIR, "aggregate_results.py")))


if __name__ == "__main__":
    main()
