# -*- coding: utf-8 -*-
from pathlib import Path
"""
SEDetector 消融实验结果汇总
===========================
收集所有变体 x 数据集 x 种子的 test_detail.json，计算 mean +/- std，
生成论文用 LaTeX 表格。

用法: python aggregate_results.py
"""

import json
import os
import math
import glob
ROOT = str(Path(__file__).resolve().parent.parent)  # project root

SEED_DIR = os.path.dirname(os.path.abspath(__file__))
SEEDS = [2021, 2022, 2023, 2024, 2025]

VARIANTS = {
    "SEDetector":   None,       # full model (special handling)
    "SEU+MLP":      "Abl_SEU_MLP",
    "SEU+GCN":      "Abl_SEU_GCN",
    "Raw+HG":       "Abl_RawHG",
    "w/o KD":       "Abl_NoKD",
    "w/o SEU":      "Abl_NoSEU",
}

DATASETS = {
    "OPTC": {
        "progress_dir": os.path.join(ROOT, "progress_OPTC"),
        "full_out": "outputs_OPTC",
        "abl_prefix": "outputs_OPTC_abl_",
    },
    "TCE5": {
        "progress_dir": os.path.join(ROOT, "progress_TCE5"),
        "full_out": "outputs_TCE5",
        "abl_prefix": "outputs_TCE5_abl_",
    },
    "DAPT": {
        "progress_dir": os.path.join(ROOT, "progress_DAPT"),
        "full_out": "outputs_DAPT",
        "abl_prefix": "outputs_DAPT_abl_",
    },
}

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


def collect_from_dir(base_dir, seed_tag):
    """从 base_dir 中找到含 seed_tag 的子目录，读取 test_detail.json"""
    if not os.path.isdir(base_dir):
        return None
    dirs = sorted(os.listdir(base_dir))
    matching = [d for d in dirs if seed_tag in d and os.path.isdir(os.path.join(base_dir, d))]
    for d in reversed(matching):
        td = os.path.join(base_dir, d, "test_detail.json")
        if os.path.isfile(td):
            try:
                data = json.loads(open(td, "r", encoding="utf-8").read().strip())
                if data:
                    return data.get("test_metrics", {})
            except:
                continue
    return None


def collect_variant(ds_name, ds_cfg, variant_label, variant_dir):
    """收集某个变体在某数据集上所有种子的指标"""
    results = {mk: [] for mk, _ in METRIC_MAP}
    seeds_done = []

    if variant_dir is None:
        # Full SEDetector
        base = os.path.join(ds_cfg["progress_dir"], ds_cfg["full_out"])
    else:
        short = variant_dir.replace("Abl_", "")
        base = os.path.join(ds_cfg["progress_dir"],
                            ds_cfg["abl_prefix"] + short)

    for seed in SEEDS:
        tag = "seed{}".format(seed)
        metrics = collect_from_dir(base, tag)
        if metrics:
            for mk, _ in METRIC_MAP:
                v = metrics.get(mk)
                if v is not None:
                    results[mk].append(v)
            seeds_done.append(seed)

    return results, seeds_done


def main():
    lines = []
    def P(s=""):
        print(s)
        lines.append(s)

    P("=" * 100)
    P("  SEDetector Ablation Study Results  (mean +/- std over {} seeds)".format(len(SEEDS)))
    P("=" * 100)

    # 汇总表
    all_data = {}  # (variant_label, ds_name) -> (results, seeds_done)

    for ds_name, ds_cfg in DATASETS.items():
        P("\n[{}]".format(ds_name))
        P("  {:<14} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
            "Variant", "Seeds", "Macro-F1", "Acc", "Macro-Prec", "Macro-Rec", "ROC-AUC", "PR-AUC"))
        P("  " + "-" * 94)

        for vlabel, vdir in VARIANTS.items():
            res, done = collect_variant(ds_name, ds_cfg, vlabel, vdir)
            all_data[(vlabel, ds_name)] = (res, done)

            if done:
                vals = []
                for mk, _ in METRIC_MAP:
                    m, s = mean_std(res[mk])
                    vals.append(m)
                P("  {:<14} {:>6} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}".format(
                    vlabel, "{}/{}".format(len(done), len(SEEDS)),
                    vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]))
            else:
                P("  {:<14} {:>6}  (no results)".format(vlabel, "0/{}".format(len(SEEDS))))

    # LaTeX 表格
    P("\n" + "=" * 100)
    P("LaTeX TABLE")
    P("=" * 100)
    P("")

    tex = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\caption{Ablation study results (mean $\pm$ std over 5 random seeds).}")
    tex.append(r"\label{tab:ablation}")
    tex.append(r"\begin{tabular}{lccc}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Configuration} & \textbf{OPTC} & \textbf{TCE5} & \textbf{DAPT2020} \\")
    tex.append(r"\midrule")

    for vlabel, vdir in VARIANTS.items():
        parts = []
        if vlabel == "SEDetector":
            parts.append(r"\textbf{" + vlabel + "}")
        else:
            parts.append(vlabel)

        for ds_name in ["OPTC", "TCE5", "DAPT"]:
            res, done = all_data.get((vlabel, ds_name), ({mk: [] for mk, _ in METRIC_MAP}, []))
            f1_vals = res.get("Macro-F1", [])
            if len(f1_vals) >= 1:
                m, s = mean_std(f1_vals)
                if vlabel == "SEDetector":
                    parts.append(r"\textbf{${:.2f} \pm {:.2f}$}".format(m, s))
                else:
                    parts.append("${:.2f} \pm {:.2f}$".format(m, s))
            else:
                parts.append("---")

        tex.append(" & ".join(parts) + r" \\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")

    latex_text = "\n".join(tex)
    P(latex_text)

    # 保存
    out_path = os.path.join(SEED_DIR, "ablation_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    P("\nSaved: {}".format(out_path))

    tex_path = os.path.join(SEED_DIR, "ablation_latex.txt")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_text)
    P("Saved: {}".format(tex_path))


if __name__ == "__main__":
    main()
