# -*- coding: utf-8 -*-
# 推理——迭代数据划分数据集
"""
Merge three JSONL files into one JSONL file.
- Keep original lines unchanged
- Do NOT add or modify any fields
- Preserve JSONL format (one JSON object per line)
"""

import argparse
from pathlib import Path

def merge_three_jsonl(
    file1: Path,
    file2: Path,
    file3: Path,
    out_file: Path,
):
    out_file.parent.mkdir(parents=True, exist_ok=True)

    counts = []

    with out_file.open("w", encoding="utf-8") as fout:
        for fp in [file1, file2, file3]:
            cnt = 0
            with fp.open("r", encoding="utf-8", errors="ignore") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line.rstrip("\n") + "\n")
                        cnt += 1
            counts.append(cnt)

    print("[OK] Merge finished.")
    print(f"  file1 ({file1.name}): {counts[0]}")
    print(f"  file2 ({file2.name}): {counts[1]}")
    print(f"  file3 ({file3.name}): {counts[2]}")
    print(f"  total lines          : {sum(counts)}")
    print(f"  output               : {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Merge three JSONL files")
    parser.add_argument("--f1", default="../Result/5.split/0160/val.jsonl", help="first jsonl file")
    parser.add_argument("--f2", default="../Result/5.split/0165/val.jsonl", help="second jsonl file")
    parser.add_argument("--f3", default="../Result/5.split/0170/val.jsonl", help="third jsonl file")
    parser.add_argument("--out", default="../Result/6.Merge_all/val_all.jsonl", help="output jsonl file")
    args = parser.parse_args()

    merge_three_jsonl(
        file1=Path(args.f1),
        file2=Path(args.f2),
        file3=Path(args.f3),
        out_file=Path(args.out),
    )

if __name__ == "__main__":
    main()
