# -*- coding: utf-8 -*-
from pathlib import Path
"""
SEDetector 消融实验 缺失项补跑
5 变体 x 3 数据集 x 5 seeds = 75 次实验
  1. SEU+MLP   2. SEU+GCN   3. w/o OpCatAgg
  4. w/o TemplAbs   5. w/o RoleTok
用法: python run_missing.py [--variants SEU_MLP NoOpCat] [--datasets OPTC] [--dry_run]
"""
import subprocess, sys, os, time, argparse
from datetime import datetime
ROOT = str(Path(__file__).resolve().parent.parent)  # project root

PYTHON = sys.executable
SEED_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SEED_DIR, "run_missing_log.txt")
SEEDS = [2021, 2022, 2023, 2024, 2025]
DIRS = {
    "OPTC": os.path.join(ROOT, "progress_OPTC"),
    "TCE5": os.path.join(ROOT, "progress_TCE5"),
    "DAPT": os.path.join(ROOT, "progress_DAPT"),
}

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = "[{}] {}".format(ts, msg)
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

ALL_EXPERIMENTS = [
    # (variant, dataset, script_path)
    ("SEU_MLP", "OPTC", os.path.join(ROOT, "experiments", "ablation", "data_ablations", "SEU_MLP_OPTC.py"),
    ("SEU_MLP", "TCE5", os.path.join(ROOT, "experiments", "ablation", "data_ablations", "SEU_MLP_TCE5.py"),
    ("SEU_MLP", "DAPT", os.path.join(ROOT, "experiments", "ablation", "data_ablations", "SEU_MLP_DAPT.py"),
    ("SEU_GCN", "OPTC", os.path.join(ROOT, "experiments", "ablation", "data_ablations", "SEU_GCN_OPTC.py"),
    ("SEU_GCN", "TCE5", os.path.join(ROOT, "experiments", "ablation", "data_ablations", "SEU_GCN_TCE5.py"),
    ("SEU_GCN", "DAPT", os.path.join(ROOT, "experiments", "ablation", "data_ablations", "SEU_GCN_DAPT.py"),
    ("NoOpCat", "OPTC", os.path.join(ROOT, "experiments", "ablation", "data_ablations", "NoOpCat_OPTC.py"),
    ("NoOpCat", "TCE5", os.path.join(ROOT, "experiments", "ablation", "data_ablations", "NoOpCat_TCE5.py"),
    ("NoOpCat", "DAPT", os.path.join(ROOT, "experiments", "ablation", "data_ablations", "NoOpCat_DAPT.py"),
    ("NoTemplAbs", "OPTC", os.path.join(ROOT, "experiments", "ablation", "data_ablations", "NoTemplAbs_OPTC.py"),
    ("NoTemplAbs", "TCE5", os.path.join(ROOT, "experiments", "ablation", "data_ablations", "NoTemplAbs_TCE5.py"),
    ("NoTemplAbs", "DAPT", os.path.join(ROOT, "experiments", "ablation", "data_ablations", "NoTemplAbs_DAPT.py"),
    ("NoRole", "OPTC", os.path.join(ROOT, "experiments", "ablation", "data_ablations", "NoRole_OPTC.py"),
    ("NoRole", "TCE5", os.path.join(ROOT, "experiments", "ablation", "data_ablations", "NoRole_TCE5.py"),
    ("NoRole", "DAPT", os.path.join(ROOT, "experiments", "ablation", "data_ablations", "NoRole_DAPT.py"),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+",
        default=["SEU_MLP","SEU_GCN","NoOpCat","NoTemplAbs","NoRole"])
    parser.add_argument("--datasets", nargs="+", default=["OPTC","TCE5","DAPT"])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    exps = [(v,d,s) for v,d,s in ALL_EXPERIMENTS
            if v in args.variants and d in args.datasets]
    total = len(exps) * len(args.seeds)

    log("=" * 60)
    log("  消融实验缺失项补跑")
    log("  变体: {} ({})".format(args.variants, len(args.variants)))
    log("  数据集: {} ({})".format(args.datasets, len(args.datasets)))
    log("  Seeds: {} ({})".format(args.seeds, len(args.seeds)))
    log("  总计: {} 次实验".format(total))
    log("=" * 60)

    ok_count, fail_count, exp_num = 0, 0, 0
    prev_vd = None
    for vname, ds, script_path in exps:
        vd = (vname, ds)
        if vd != prev_vd:
            log("\n--- {} / {} ---".format(vname, ds))
            prev_vd = vd
        cwd = DIRS[ds]
        for seed in args.seeds:
            exp_num += 1
            log("[{}/{}] {} / {} / seed={}".format(exp_num, total, vname, ds, seed))
            if args.dry_run:
                log("  [DRY] " + script_path)
                ok_count += 1
                continue
            cmd = [PYTHON, script_path, "--seed", str(seed)]
            t0 = time.time()
            try:
                result = subprocess.run(cmd, cwd=cwd)
                elapsed = time.time() - t0
                if result.returncode == 0:
                    log("  [OK]   ({:.1f} min)".format(elapsed / 60))
                    ok_count += 1
                else:
                    log("  [FAIL] exit={} ({:.1f} min)".format(result.returncode, elapsed / 60))
                    fail_count += 1
            except Exception as e:
                elapsed = time.time() - t0
                log("  [ERR]  {} ({:.1f} min)".format(str(e), elapsed / 60))
                fail_count += 1

    log("\n" + "=" * 60)
    log("  ALL DONE: OK={}  FAIL={}  TOTAL={}".format(ok_count, fail_count, total))
    log("=" * 60)

if __name__ == "__main__":
    main()
