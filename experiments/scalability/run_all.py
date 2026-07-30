# -*- coding: utf-8 -*-
"""
复杂度、运行时间、可扩展性实验 - 一键运行
=============================================

run from command line.

执行顺序:
  Step 1: 8.1 运行时开销测量 (OPTC + TCE5 + DAPT)
  Step 2: 8.2 可扩展性实验 (TCE5, 25%/50%/75%/100%)

预计时间:
  - Step 1: ~15-30 分钟 (取决于 GPU)
  - Step 2: ~15-25 分钟

结果输出:
  - runtime_cost.csv        — 8.1 运行时开销
  - scalability_tce5.csv    — 8.2 可扩展性数据
"""
import sys
import os
import time
import functools

print = functools.partial(print, flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def run_step(step_num, name, func):
    print()
    print("#" * 60)
    print(f"  Step {step_num}: {name}")
    print("#" * 60)
    print()

    t0 = time.time()
    try:
        func()
    except Exception as e:
        print(f"  ERROR in Step {step_num}: {e}")
        import traceback
        traceback.print_exc()
        return False

    elapsed = time.time() - t0
    print(f"\n  Step {step_num} completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("  SEDetector: Complexity, Runtime & Scalability")
    print("  (R1/R4: Computational Complexity & Scalability)")
    print("=" * 60)
    print()
    print("  Step 1: 8.1 Runtime cost measurement (3 datasets)")
    print("  Step 2: 8.2 Scalability experiment (TCE5)")
    print()
    print("  预计总时间: ~30-60 分钟")
    print("=" * 60)

    t_total = time.time()

    # Step 1: Runtime measurement
    def step1():
        from runtime_measurement import run
        run(out_dir=HERE)

    run_step(1, "8.1 Runtime Cost Measurement", step1)

    # Step 2: Scalability
    def step2():
        from scalability_experiment import run
        run(out_dir=HERE)

    run_step(2, "8.2 Scalability Experiment (TCE5)", step2)

    elapsed_total = time.time() - t_total
    print()
    print("=" * 60)
    print(f"  All done! Total time: {elapsed_total/60:.1f} minutes")
    print()
    print("  Output files:")
    print(f"    runtime_cost.csv:     {os.path.join(HERE, 'runtime_cost.csv')}")
    print(f"    scalability_tce5.csv: {os.path.join(HERE, 'scalability_tce5.csv')}")
    print("=" * 60)
