# -*- coding: utf-8 -*-
"""
Balanced Accuracy + Confusion Matrix 计算
==========================================
从已有的 preds_test.csv 直接计算:
  - Balanced Accuracy
  - Normalized confusion matrix (row-normalized)
  - Per-class recall
  - 与现有 test_detail.json 的 Acc/F1 对比验证

不需要重新跑模型推理，直接读现有结果文件。

运行: python compute_balanced_acc_and_confusion.py
输出: 同目录下 balanced_acc_results.csv + confusion_matrices.json
"""
import os, sys, json, csv
import numpy as np
from collections import Counter, OrderedDict
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
    "Main":       {"suffix": ""},
    "NoKD":       {"suffix": "_abl_NoKD"},
    "NoSEU":      {"suffix": "_abl_NoSEU"},
    "SEU_MLP":    {"suffix": "_abl_SEU_MLP"},
    "SEU_GCN":    {"suffix": "_abl_SEU_GCN"},
    "RawHG":      {"suffix": "_abl_RawHG"},
    "NoOpCat":    {"suffix": "_abl_NoOpCat"},
    "NoTemplAbs": {"suffix": "_abl_NoTemplAbs"},
    "NoRole":     {"suffix": "_abl_NoRole"},
}


def find_seed_run(base_dir, seed):
    if not base_dir.exists():
        return None
    matches = [d for d in base_dir.iterdir()
               if d.is_dir() and f"seed{seed}_" in d.name]
    if not matches:
        return None
    matches.sort(key=lambda x: x.name)
    return matches[-1]


def load_preds_test(run_dir):
    csv_path = run_dir / "preds_test.csv"
    if not csv_path.exists():
        return None, None, None, None
    y_true, y_pred, probs = [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            y_true.append(int(row[0]))
            y_pred.append(int(row[1]))
            probs.append([float(row[4 + i]) for i in range(len(header) - 4)])
    class_names = [h.replace("p_", "") for h in header[4:]]
    return (np.array(y_true), np.array(y_pred),
            np.array(probs, dtype=np.float32), class_names)


def load_test_detail(run_dir):
    p = run_dir / "test_detail.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def confusion_matrix_np(y_true, y_pred, n_classes):
    mat = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            mat[t, p] += 1
    return mat


def main():
    all_rows = []
    confusion_data = OrderedDict()

    for ds_name, ds_cfg in DATASETS.items():
        print(f"\n{'='*70}")
        print(f"  Dataset: {ds_name}")
        print(f"{'='*70}")

        for var_name, var_cfg in VARIANTS.items():
            out_base = ds_cfg["progress"] / (ds_cfg["out_prefix"] + var_cfg["suffix"])
            if not out_base.exists():
                continue

            var_results = []
            class_names = None

            for seed in SEEDS:
                run_dir = find_seed_run(out_base, seed)
                if run_dir is None:
                    continue

                y_true, y_pred, probs, cnames = load_preds_test(run_dir)
                if y_true is None:
                    continue
                if class_names is None:
                    class_names = cnames
                n_classes = len(class_names)

                detail = load_test_detail(run_dir)

                from sklearn.metrics import (
                    accuracy_score, balanced_accuracy_score, f1_score,
                    precision_score, recall_score
                )

                acc = accuracy_score(y_true, y_pred)
                bacc = balanced_accuracy_score(y_true, y_pred)
                mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
                mprec = precision_score(y_true, y_pred, average="macro", zero_division=0)
                mrec = recall_score(y_true, y_pred, average="macro", zero_division=0)

                cm = confusion_matrix_np(y_true, y_pred, n_classes)
                cm_norm = cm.astype(np.float32)
                row_sums = cm_norm.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                cm_norm = cm_norm / row_sums

                per_class_rec = {}
                for i, cn in enumerate(class_names):
                    per_class_rec[cn] = float(cm_norm[i, i]) if i < n_classes else 0.0

                ref_acc = detail["test_metrics"]["Acc"] if detail else None
                ref_f1 = detail["test_metrics"]["Macro-F1"] if detail else None

                row = {
                    "dataset": ds_name, "variant": var_name, "seed": seed,
                    "n_test": len(y_true), "n_classes": n_classes,
                    "Acc": acc, "Balanced_Acc": bacc,
                    "Macro_F1": mf1, "Macro_Prec": mprec, "Macro_Rec": mrec,
                }
                if ref_acc is not None:
                    row["ref_Acc"] = ref_acc
                    row["ref_F1"] = ref_f1
                    row["Acc_match"] = abs(acc - ref_acc) < 1e-4
                all_rows.append(row)

                cm_key = f"{ds_name}/{var_name}/seed{seed}"
                confusion_data[cm_key] = {
                    "class_names": class_names,
                    "confusion_matrix": cm.tolist(),
                    "confusion_matrix_row_norm": [[round(float(x), 4) for x in row] for row in cm_norm],
                    "per_class_recall": {k: round(v, 4) for k, v in per_class_rec.items()},
                    "balanced_acc": round(bacc, 6),
                    "class_distribution": dict(Counter(y_true.tolist())),
                }

                var_results.append({
                    "seed": seed, "Acc": acc, "BAcc": bacc, "F1": mf1,
                })

                status = "OK" if (ref_acc is None or abs(acc - ref_acc) < 1e-4) else "MISMATCH"
                print(f"    seed{seed}: Acc={acc:.4f} BAcc={bacc:.4f} F1={mf1:.4f} [{status}]")

            if var_results:
                accs = [r["Acc"] for r in var_results]
                baccs = [r["BAcc"] for r in var_results]
                f1s = [r["F1"] for r in var_results]
                print(f"  => {var_name}: Acc={np.mean(accs):.4f}+/-{np.std(accs, ddof=1):.4f}  "
                      f"BAcc={np.mean(baccs):.4f}+/-{np.std(baccs, ddof=1):.4f}  "
                      f"F1={np.mean(f1s):.4f}+/-{np.std(f1s, ddof=1):.4f}")

    # --- CSV ---
    out_dir = Path(__file__).parent
    csv_path = out_dir / "balanced_acc_results.csv"
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in all_rows:
                w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
        print(f"\nCSV: {csv_path}")

    # --- Confusion matrices JSON ---
    json_path = out_dir / "confusion_matrices.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(confusion_data, f, indent=2, ensure_ascii=False)
    print(f"JSON: {json_path}")

    # --- Summary table ---
    print_summary_table(all_rows)
    print_latex_table(all_rows)


def print_summary_table(all_rows):
    print(f"\n{'='*90}")
    print(f"  SUMMARY: Balanced Accuracy by Variant (mean +/- std, 5 seeds)")
    print(f"{'='*90}")
    header = f"{'Dataset':<8} {'Variant':<14} {'Acc':>12} {'Balanced Acc':>14} {'Macro-F1':>12} {'n_cls':>5}"
    print(header)
    print("-" * len(header))
    for ds_name in DATASETS:
        for var_name in VARIANTS:
            rows = [r for r in all_rows if r["dataset"] == ds_name and r["variant"] == var_name]
            if not rows:
                continue
            acc = np.mean([r["Acc"] for r in rows])
            bacc = np.mean([r["Balanced_Acc"] for r in rows])
            f1 = np.mean([r["Macro_F1"] for r in rows])
            acc_s = np.std([r["Acc"] for r in rows], ddof=1) if len(rows) > 1 else 0
            bacc_s = np.std([r["Balanced_Acc"] for r in rows], ddof=1) if len(rows) > 1 else 0
            f1_s = np.std([r["Macro_F1"] for r in rows], ddof=1) if len(rows) > 1 else 0
            n_cls = rows[0]["n_classes"]
            print(f"{ds_name:<8} {var_name:<14} {acc:.4f}+/-{acc_s:.4f} "
                  f"{bacc:.4f}+/-{bacc_s:.4f} {f1:.4f}+/-{f1_s:.4f} {n_cls:>5}")


def print_latex_table(all_rows):
    for ds_name in DATASETS:
        ds_rows = [r for r in all_rows if r["dataset"] == ds_name]
        if not ds_rows:
            continue
        variants_seen = []
        for r in ds_rows:
            if r["variant"] not in variants_seen:
                variants_seen.append(r["variant"])

        print(f"\n% === {ds_name}: Main metrics + Balanced Accuracy ===")
        print(r"\begin{tabular}{l" + "cc" + "}")
        print(r"\toprule")
        print(r"Variant & Macro-F1 & Balanced Acc \\")
        print(r"\midrule")
        for vn in variants_seen:
            rows = [r for r in ds_rows if r["variant"] == vn]
            f1 = np.mean([r["Macro_F1"] for r in rows])
            f1_s = np.std([r["Macro_F1"] for r in rows], ddof=1) if len(rows) > 1 else 0
            bacc = np.mean([r["Balanced_Acc"] for r in rows])
            bacc_s = np.std([r["Balanced_Acc"] for r in rows], ddof=1) if len(rows) > 1 else 0
            print(f"{vn:<14s} & ${f1:.4f}{{\\pm}}{f1_s:.4f}$ & ${bacc:.4f}{{\\pm}}{bacc_s:.4f}$ \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")


if __name__ == "__main__":
    main()
