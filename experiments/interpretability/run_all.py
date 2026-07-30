# -*- coding: utf-8 -*-
"""
Top-k Perturbation Faithfulness - 一键运行
=============================================

run from command line.

对三个数据集 (OPTC, TCE5, DAPT) 运行 top-k perturbation 实验:
  - 加载已训练模型 (BEST/best.pt)
  - 对测试样本做 leave-one-out 重要性排名
  - Top-1/3/5 vs Random-1/3/5 扰动对比
  - 输出 faithful gap 汇总表

预计时间:
  - OPTC: ~2-3 分钟 (1,519 测试样本, 取 500)
  - TCE5: ~5-8 分钟 (大量测试样本, 取 500)
  - DAPT: ~5-8 分钟 (22,960 测试样本, 取 500)
  - 总计: ~15-20 分钟

结果输出:
  - perturbation_faithfulness.csv (详细数据)
  - 终端打印汇总表
"""
import sys
import os
import time
import functools

print = functools.partial(print, flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


if __name__ == "__main__":
    print("=" * 60)
    print("  SEDetector: Top-k Perturbation Faithfulness")
    print("  (R2/R4: Explainability Ranking Validation)")
    print("=" * 60)
    print()
    print("  数据集:   OPTC / TCE5 / DAPT")
    print("  Top-k:    1, 3, 5")
    print("  Random:   20 次重复取平均")
    print("  样本上限: 500 / 数据集")
    print()
    print("  预计时间: ~15-20 分钟")
    print("=" * 60)

    t0 = time.time()

    from perturbation_analysis import run
    results = run(out_dir=HERE)

    elapsed = time.time() - t0
    print(f"\n全部实验完成! 总耗时: {elapsed/60:.1f} 分钟")
    print(f"结果文件: {os.path.join(HERE, 'perturbation_faithfulness.csv')}")
