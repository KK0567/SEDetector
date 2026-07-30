# -*- coding: utf-8 -*-
"""
DAPT2020 稳健性实验 + 模态差异分析 - 一键运行
================================================

run from command line.

执行顺序:
  Step 1: DAPT2020 chronological split (按天划分: Mon+Tue+Wed / Thu / Fri)
  Step 2: 构建 chrono split 超图
  Step 3: 训练 SEDetector + SEU+MLP on chrono split
  Step 4: 模态差异统计 (OPTC vs TCE5 vs DAPT)

预计时间:
  - Step 1-2: ~2 分钟
  - Step 3: ~3-4 小时 (SEDetector ~2-3h, SEU+MLP ~30-60min)
  - Step 4: ~1 分钟

结果输出:
  - results/SEDetector_chrono/ — SEDetector 训练输出
  - results/SEU_MLP_chrono/    — SEU+MLP 训练输出
  - modality_stats.csv          — 模态差异统计
"""
import sys
import os
import time
import functools

print = functools.partial(print, flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def run_step(step_num, name, func):
    """运行一个步骤，带计时和错误处理"""
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
    print("  SEDetector: DAPT2020 Robustness + Modality Analysis")
    print("  (reviewer-comment: Chronological Split / Modality Differences)")
    print("=" * 60)
    print()
    print("  Step 1: DAPT2020 chronological split")
    print("  Step 2: Build hypergraphs for chrono split")
    print("  Step 3: Train SEDetector + SEU+MLP on chrono split")
    print("  Step 4: Modality-dependent statistics")
    print()
    print("  预计总时间: ~3-4 小时 (Step 3 占主要时间)")
    print("  可以离开电脑，全部自动完成")
    print("=" * 60)

    t_total = time.time()

    # Step 1: Chronological split
    def step1():
        from chrono_split import run
        run()

    ok = run_step(1, "DAPT2020 Chronological Split", step1)
    if not ok:
        sys.exit(1)

    # Step 2: Build hypergraphs
    def step2():
        from build_hyper_chrono import run
        run()

    ok = run_step(2, "Build Hypergraphs", step2)
    if not ok:
        sys.exit(1)

    # Step 3: Train models
    def step3():
        from train_chrono import run
        run()

    ok = run_step(3, "Train SEDetector + SEU+MLP", step3)
    # 训练失败不中断，继续跑 Step 4

    # Step 4: Modality analysis
    def step4():
        from modality_analysis import run
        run(out_dir=HERE)

    ok = run_step(4, "Modality-Dependent Statistics", step4)

    # 汇总
    elapsed_total = time.time() - t_total
    print()
    print("=" * 60)
    print(f"  All done! Total time: {elapsed_total/60:.1f} minutes")
    print()
    print("  Output files:")
    print(f"    Chrono data:   {os.path.join(HERE, '..', '..', 'data_DAPT_chrono')}")
    print(f"    Train results:  {os.path.join(HERE, 'results')}")
    print(f"    Modality CSV:   {os.path.join(HERE, 'modality_stats.csv')}")
    print("=" * 60)
