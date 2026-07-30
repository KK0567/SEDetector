# -*- coding: utf-8 -*-
"""
k-Anonymity 分析 - 一键运行
run from command line.

默认运行全部 3 个数据集, 预计 1-2 分钟完成.

结果输出:
  - 终端打印汇总表
  - k_anonymity_results.csv (详细数据)
"""
import sys
import os
import time

# 确保本目录在 path 中
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from k_anonymity_analysis import run_all

if __name__ == "__main__":
    print("=" * 60)
    print("  SEDetector k-Anonymity-Inspired Analysis")
    print("  (R3: Standard Privacy Framework Comparison)")
    print("=" * 60)
    print()
    print("  QID 设置:")
    print("    1. Token-only          - frozenset(event_types)")
    print("    2. Token+Role+Context  - (events, entity_cats, roles, context)")
    print()
    print("  数据集:   OPTC / TCE5 / DAPT")
    print("  预计时间: 1-2 分钟")
    print("=" * 60)

    t0 = time.time()

    results = run_all(
        datasets=["OPTC", "TCE5", "DAPT"],
        out_dir=HERE,
    )

    elapsed = time.time() - t0
    print(f"\n全部分析完成! 总耗时: {elapsed:.1f} 秒")
    print(f"结果文件: {os.path.join(HERE, 'k_anonymity_results.csv')}")
