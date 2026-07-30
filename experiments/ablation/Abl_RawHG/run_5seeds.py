# -*- coding: utf-8 -*-
from pathlib import Path
"""
Abl_RawHG / DAPT  5-seed runner
IDE 中点击 Run 即可依次运行 seed 2021~2025
"""
import subprocess, sys, os, time
from datetime import datetime
ROOT = str(Path(__file__).resolve().parent.parent.parent)  # project root

PYTHON = sys.executable
SEEDS = [2021, 2022, 2023, 2024, 2025]
CWD = os.path.dirname(os.path.abspath(__file__))
PROGRESS_DIR = os.path.join(ROOT, "progress_DAPT")
LOG_FILE = os.path.join(CWD, "DAPT_5seeds_log.txt")

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = "[{}] {}".format(ts, msg)
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def main():
    log("=" * 60)
    log("Abl_RawHG / DAPT  5-seed  PID={}".format(os.getpid()))
    log("Seeds: {}".format(SEEDS))
    log("=" * 60)
    ok, fail = 0, 0
    for i, seed in enumerate(SEEDS, 1):
        cmd = [PYTHON, os.path.join(CWD, "run_DAPT.py"), "--seed", str(seed)]
        log("[{}/5] seed={} STARTING".format(i, seed))
        t0 = time.time()
        try:
            result = subprocess.run(cmd, cwd=PROGRESS_DIR)
            elapsed = time.time() - t0
            if result.returncode == 0:
                log("[{}/5] seed={} OK ({:.1f} min)".format(i, seed, elapsed/60))
                ok += 1
            else:
                log("[{}/5] seed={} FAIL exit={} ({:.1f} min)".format(i, seed, result.returncode, elapsed/60))
                fail += 1
        except Exception as e:
            elapsed = time.time() - t0
            log("[{}/5] seed={} EXCEPTION: {} ({:.1f} min)".format(i, seed, str(e), elapsed/60))
            fail += 1
    log("=" * 60)
    log("DONE: OK={}  FAIL={}".format(ok, fail))
    log("=" * 60)

if __name__ == "__main__":
    main()
