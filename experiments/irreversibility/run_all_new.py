# -*- coding: utf-8 -*-
"""
DAPT2020 Irreversibility - One-click Runner
Usage: python run_all_bukeni_new.py
"""
import subprocess, sys, os, time
from datetime import datetime

PYTHON = sys.executable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_SCRIPT = os.path.join(SCRIPT_DIR, "eval_irreversibility_new.py")

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def main():
    log("=" * 60)
    log("  DAPT2020 Irreversibility Evaluation")
    log("  3 windows (60/300/900s) x 5 seeds")
    log("=" * 60)

    t0 = time.time()
    cmd = [PYTHON, EVAL_SCRIPT]
    try:
        result = subprocess.run(cmd, cwd=SCRIPT_DIR)
        elapsed = time.time() - t0
        if result.returncode == 0:
            log(f"[OK] DAPT2020 ({elapsed/60:.1f} min)")
        else:
            log(f"[FAIL] exit={result.returncode} ({elapsed/60:.1f} min)")
    except Exception as e:
        log(f"[ERR] {e}")

if __name__ == "__main__":
    main()
