# -*- coding: utf-8 -*-
from pathlib import Path
"""
汇总 OPTC / TCE5 / DAPT 三个数据集 5-seed 实验结果
==================================================
读取 test_detail.json, 计算 mean +/- std, 生成 LaTeX 表格。

用法: python aggregate_results.py
"""

import json
import os
import math
ROOT = str(Path(__file__).resolve().parent.parent)  # project root

BASES = {
    "OPTC": os.path.join(ROOT, "progress_OPTC", "outputs_OPTC"),
    "TCE5": os.path.join(ROOT, "progress_TCE5", "outputs_TCE5"),
    "DAPT": os.path.join(ROOT, "progress_DAPT", "outputs_DAPT"),
}
SEEDS = [2021, 2022, 2023, 2024, 2025]

METRIC_MAP = [
    ("Macro-F1",   "Macro-F$_1$"),
    ("Acc",        "Accuracy"),
    ("Macro-Prec", "Macro-Prec"),
    ("Macro-Rec",  "Macro-Rec"),
    ("ROC-AUC",    "ROC-AUC"),
    ("PR-AUC",     "PR-AUC"),
]


def mean_std(vals):
    if not vals:
        return 0.0, 0.0
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in vals) / (n - 1)
    return m, math.sqrt(var)


def collect_dataset(base_dir):
    results = {mk: [] for mk, _ in METRIC_MAP}
    seeds_done, seeds_pending = [], []
    if not os.path.isdir(base_dir):
        return results, [], list(SEEDS)
    dirs = sorted(os.listdir(base_dir))
    for seed in SEEDS:
        tag = "seed{}".format(seed)
        matching = [d for d in dirs if tag in d and os.path.isdir(os.path.join(base_dir, d))]
        found = False
        for d in reversed(matching):
            td = os.path.join(base_dir, d, "test_detail.json")
            if os.path.isfile(td):
                try:
                    data = json.loads(open(td, "r", encoding="utf-8").read().strip())
                    if not data:
                        continue
                    tm = data.get("test_metrics", {})
                    for json_key, _ in METRIC_MAP:
                        v = tm.get(json_key)
                        if v is not None:
                            results[json_key].append(v)
                    seeds_done.append(seed)
                    found = True
                    break
                except:
                    continue
        if not found:
            seeds_pending.append(seed)
    return results, seeds_done, seeds_pending


def collect_per_seed(base_dir):
    per_seed = {}
    if not os.path.isdir(base_dir):
        return per_seed
    dirs = sorted(os.listdir(base_dir))
    for seed in SEEDS:
        tag = "seed{}".format(seed)
        matching = [d for d in dirs if tag in d and os.path.isdir(os.path.join(base_dir, d))]
        for d in reversed(matching):
            td = os.path.join(base_dir, d, "test_detail.json")
            if os.path.isfile(td):
                try:
                    data = json.loads(open(td, "r", encoding="utf-8").read().strip())
                    tm = data.get("test_metrics", {})
                    per_seed[seed] = {mk: tm.get(mk, None) for mk, _ in METRIC_MAP}
                    per_seed[seed]["best_epoch"] = data.get("best_epoch", None)
                    break
                except:
                    continue
    return per_seed


def main():
    all_results = {}
    all_per_seed = {}
    lines = []
    def P(s=""):
        print(s)
        lines.append(s)

    P("=" * 100)
    P("SEDetector  5-Seed Results  (3 datasets)")
    P("=" * 100)

    for ds, base in BASES.items():
        res, done, pending = collect_dataset(base)
        ps = collect_per_seed(base)
        all_results[ds] = (res, done, pending)
        all_per_seed[ds] = ps

        P("\n[{}]  {}/5 seeds complete   done={}  pending={}".format(
            ds, len(done), done, pending))

        P("  {:<6} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
            "Seed", "BestEp", "Macro-F1", "Acc", "Macro-Prec", "Macro-Rec", "ROC-AUC", "PR-AUC"))
        P("  " + "-" * 84)
        for seed in SEEDS:
            if seed in ps:
                v = ps[seed]
                P("  {:<6} {:>8} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
                    seed, v.get("best_epoch", "?"),
                    v.get("Macro-F1", 0) or 0, v.get("Acc", 0) or 0,
                    v.get("Macro-Prec", 0) or 0, v.get("Macro-Rec", 0) or 0,
                    v.get("ROC-AUC", 0) or 0, v.get("PR-AUC", 0) or 0))
            else:
                P("  {:<6} (pending)".format(seed))

        P("  " + "-" * 84)
        vals_list = []
        for json_key, _ in METRIC_MAP:
            m, s = mean_std(res[json_key])
            vals_list.append((json_key, m, s, len(res[json_key])))
        row_mean = "  {:<6} {:>8} ".format("Mean", "")
        row_std = "  {:<6} {:>8} ".format("Std", "")
        for _, m, s, n in vals_list:
            row_mean += "{:>10.4f} ".format(m)
            row_std += "{:>10.4f} ".format(s)
        P(row_mean)
        P(row_std)

    # LaTeX table
    P("\n" + "=" * 100)
    P("LaTeX TABLE")
    P("=" * 100)
    P("")

    tex = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{SEDetector performance on three APT datasets "
               r"(mean $\pm$ std over 5 random seeds).}")
    tex.append(r"\label{tab:main_results}")
    tex.append(r"\begin{tabular}{lccc}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Metric} & \textbf{OPTC} & \textbf{TCE5} & \textbf{DAPT2020} \\")
    tex.append(r"\midrule")

    for json_key, latex_name in METRIC_MAP:
        parts = [latex_name]
        for ds in ["OPTC", "TCE5", "DAPT"]:
            res, done, pending = all_results[ds]
            vals = res[json_key]
            if len(vals) >= 1:
                m, s = mean_std(vals)
                parts.append("${:.2f} \\pm {:.2f}$".format(m * 100, s * 100))
            else:
                parts.append("---")
        tex.append(" & ".join(parts) + r" \\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")

    latex_text = "\n".join(tex)
    P(latex_text)

    # Save
    out_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(out_dir, "results_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    P("\nSaved: {}".format(txt_path))

    tex_path = os.path.join(out_dir, "latex_table.txt")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_text)
    P("Saved: {}".format(tex_path))


if __name__ == "__main__":
    main()
