# -*- coding: utf-8 -*-
"""
Dummy Baseline 实验
===================
对每个数据集的 test set 计算三种 dummy baseline 的指标:
  - Majority (总是预测多数类)
  - Stratified (按训练集分布随机预测)
  - Uniform (均匀随机预测)

数据来源: 从每个变体的 preds_test.csv 中读取 y_true (test labels)。
指标: Accuracy, Balanced Accuracy, Macro-F1, Macro-Precision, Macro-Recall

运行: python run_dummy_baselines.py
输出: 同目录下 dummy_baseline_results.csv + dummy_baseline_summary.txt
"""
import os, sys, json, csv, re
import numpy as np
from collections import Counter
from pathlib import Path
ROOT = str(Path(__file__).resolve().parent.parent)  # project root

PROJ = Path(ROOT)
SEEDS = [2021, 2022, 2023, 2024, 2025]

DATASETS = {
    "OPTC": {"progress": PROJ / "progress_OPTC", "out_prefix": "outputs_OPTC"},
    "TCE5": {"progress": PROJ / "progress_TCE5", "out_prefix": "outputs_TCE5"},
    "DAPT": {"progress": PROJ / "progress_DAPT", "out_prefix": "outputs_DAPT"},
}

VARIANTS = {
    "Main":       {"suffix": "",             "mode": "head_kd"},
    "NoKD":       {"suffix": "_abl_NoKD",    "mode": "head_only"},
    "NoSEU":      {"suffix": "_abl_NoSEU",   "mode": "head_only"},
    "SEU_MLP":    {"suffix": "_abl_SEU_MLP", "mode": "head_only"},
    "SEU_GCN":    {"suffix": "_abl_SEU_GCN", "mode": "head_only"},
    "RawHG":      {"suffix": "_abl_RawHG",   "mode": "head_kd"},
    "NoOpCat":    {"suffix": "_abl_NoOpCat", "mode": "head_kd"},
    "NoTemplAbs": {"suffix": "_abl_NoTemplAbs", "mode": "head_kd"},
    "NoRole":     {"suffix": "_abl_NoRole",  "mode": "head_kd"},
}


def find_seed_run(base_dir, seed):
    if not base_dir.exists():
        return None
    matches = []
    for d in base_dir.iterdir():
        if d.is_dir() and f"seed{seed}_" in d.name:
            matches.append(d)
    if not matches:
        return None
    matches.sort(key=lambda x: x.name)
    return matches[-1]


def load_preds_test(run_dir):
    csv_path = run_dir / "preds_test.csv"
    if not csv_path.exists():
        return None, None, None, None
    y_true, y_pred, probs = [], [], []
    header = None
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            y_true.append(int(row[0]))
            y_pred.append(int(row[1]))
            prob_cols = header[4:]
            probs.append([float(row[4 + i]) for i in range(len(prob_cols))])
    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)
    probs = np.array(probs, dtype=np.float32) if probs else None
    class_names = [h.replace("p_", "") for h in header[4:]] if header else None
    return y_true, y_pred, probs, class_names


def compute_metrics(y_true, y_pred, n_classes):
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        precision_score, recall_score
    )
    return {
        "Acc": accuracy_score(y_true, y_pred),
        "Balanced_Acc": balanced_accuracy_score(y_true, y_pred),
        "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro_Prec": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro_Rec": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def dummy_majority(y_true, n_classes):
    c = Counter(y_true)
    return np.full_like(y_true, c.most_common(1)[0][0])


def dummy_stratified(y_true, n_classes, seed=42):
    rng = np.random.RandomState(seed)
    counts = np.zeros(n_classes, dtype=np.float64)
    for y in y_true:
        if 0 <= y < n_classes:
            counts[y] += 1
    p = counts / counts.sum()
    return rng.choice(n_classes, size=len(y_true), p=p)


def dummy_uniform(y_true, n_classes, seed=42):
    rng = np.random.RandomState(seed)
    return rng.choice(n_classes, size=len(y_true))


def main():
    all_rows = []
    summary_lines = []

    for ds_name, ds_cfg in DATASETS.items():
        print(f"\n{'='*70}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*70}")

        for var_name, var_cfg in VARIANTS.items():
            out_base = ds_cfg["progress"] / (ds_cfg["out_prefix"] + var_cfg["suffix"])
            if not out_base.exists():
                continue

            seed_metrics = {"Majority": [], "Stratified": [], "Uniform": []}
            n_classes = None
            class_names = None
            n_found = 0
            last_y_true = None

            for seed in SEEDS:
                run_dir = find_seed_run(out_base, seed)
                if run_dir is None:
                    continue
                y_true, y_pred, probs, cnames = load_preds_test(run_dir)
                if y_true is None:
                    continue
                if class_names is None:
                    class_names = cnames
                n_classes = len(class_names) if class_names else int(y_true.max()) + 1
                n_found += 1
                last_y_true = y_true

                seed_metrics["Majority"].append(
                    compute_metrics(y_true, dummy_majority(y_true, n_classes), n_classes))
                seed_metrics["Stratified"].append(
                    compute_metrics(y_true, dummy_stratified(y_true, n_classes, seed=seed), n_classes))
                seed_metrics["Uniform"].append(
                    compute_metrics(y_true, dummy_uniform(y_true, n_classes, seed=seed), n_classes))

            if n_found == 0:
                continue

            print(f"\n  Variant: {var_name} ({n_found} seeds, {n_classes} classes, "
                  f"n_test={len(last_y_true)})")

            for strat in ["Majority", "Stratified", "Uniform"]:
                mets_list = seed_metrics[strat]
                if not mets_list:
                    continue
                avg, std = {}, {}
                for k in mets_list[0]:
                    vals = [m[k] for m in mets_list]
                    avg[k] = float(np.mean(vals))
                    std[k] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

                row = {
                    "dataset": ds_name, "variant": var_name, "baseline": strat,
                    "n_seeds": n_found, "n_classes": n_classes, "n_test": len(last_y_true),
                }
                for k in avg:
                    row[k] = f"{avg[k]:.6f}"
                    row[f"{k}_std"] = f"{std[k]:.6f}"
                all_rows.append(row)

                print(f"    {strat:<14s} Acc={avg['Acc']:.4f}+/-{std['Acc']:.4f}  "
                      f"BAcc={avg['Balanced_Acc']:.4f}+/-{std['Balanced_Acc']:.4f}  "
                      f"F1={avg['Macro_F1']:.4f}+/-{std['Macro_F1']:.4f}")

                summary_lines.append(
                    f"{ds_name}\t{var_name}\t{strat}\t"
                    f"Acc={avg['Acc']:.4f}+/-{std['Acc']:.4f}\t"
                    f"BAcc={avg['Balanced_Acc']:.4f}+/-{std['Balanced_Acc']:.4f}\t"
                    f"F1={avg['Macro_F1']:.4f}+/-{std['Macro_F1']:.4f}")

    out_dir = Path(__file__).parent
    csv_path = out_dir / "dummy_baseline_results.csv"
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nCSV: {csv_path}")

    txt_path = out_dir / "dummy_baseline_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for line in summary_lines:
            f.write(line + "\n")
    print(f"TXT: {txt_path}")

    print_latex(all_rows)


def print_latex(all_rows):
    for ds_name in DATASETS:
        ds_rows = [r for r in all_rows if r["dataset"] == ds_name]
        if not ds_rows:
            continue
        print(f"\n% === Dummy Baselines for {ds_name} ===")
        print(r"\begin{tabular}{lccc}")
        print(r"\toprule")
        print(r"Baseline & Accuracy & Balanced Acc & Macro-F1 \\")
        print(r"\midrule")
        seen = set()
        for r in ds_rows:
            key = r["baseline"]
            if key in seen:
                continue
            seen.add(key)
            def fmt(v, s):
                v4 = ".".join(v.split(".")[:2])[:6]
                s4 = s.split(".")[1][:4] if "." in s else s
                return f"${v4}{{\\pm}}{s4}$"
            print(f"{key:<14s} & {fmt(r['Acc'], r['Acc_std'])} & "
                  f"{fmt(r['Balanced_Acc'], r['Balanced_Acc_std'])} & "
                  f"{fmt(r['Macro_F1'], r['Macro_F1_std'])} \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")


if __name__ == "__main__":
    main()
