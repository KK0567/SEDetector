# -*- coding: utf-8 -*-
"""
SEDetector 显著性检验 (Statistical Significance Tests)
======================================================
对三个数据集依次运行:
  1. 读取 SEDetector 5-seed 结果
  2. 读取各消融变体的单次结果 (作为 baseline)
  3. One-sample t-test: SEDetector mean 是否显著高于 baseline
  4. Cohen's d 效应量
  5. 输出格式化表格 + LaTeX 代码

用法: IDE 打开此文件，点 Run 即可。
三个数据集依次处理，无需手动切换。

输出: stat_test/significance_results.txt
      stat_test/significance_latex.txt
"""
import json, os, sys, glob, math, csv
from pathlib import Path
from collections import OrderedDict
ROOT = str(Path(__file__).resolve().parent.parent)  # project root

# ============================================================
# 配置
# ============================================================
BASE_DIR = Path(ROOT)
OUT_DIR  = Path(__file__).parent  # stat_test/

SEEDS = [2021, 2022, 2023, 2024, 2025]
METRICS = ["Macro-F1", "Accuracy", "ROC-AUC"]

# 各数据集 5-seed 结果所在的主目录
SEED_DIRS = {
    "DAPT":   BASE_DIR / "progress_DAPT" / "outputs_DAPT",
    "OPTC":   BASE_DIR / "progress_OPTC" / "outputs_OPTC",
    "TCE5":   BASE_DIR / "progress_TCE5" / "outputs_TCE5",
}

# 消融变体目录 (作为 baseline 对比)
ABL_NAMES = ["SEU_MLP", "SEU_GCN", "RawHG", "NoKD", "NoSEU"]
ABL_LABELS = {
    "SEU_MLP": "SEU+MLP (no HG MP)",
    "SEU_GCN": "SEU+GCN",
    "RawHG":   "Raw+HG",
    "NoKD":    "No KD",
    "NoSEU":   "No SEU",
}

DATASETS = ["DAPT", "OPTC", "TCE5"]


# ============================================================
# 工具函数
# ============================================================

def find_seed_dirs(parent: Path):
    """找到 parent 下所有包含 seed2021~2025 且含 test_detail.json 的目录"""
    results = {}
    for d in sorted(parent.iterdir()):
        if not d.is_dir():
            continue
        name = d.name.lower()
        for seed in SEEDS:
            if f"seed{seed}" in name and (d / "test_detail.json").exists():
                results[seed] = d
                break
    return results


def find_best_abl(parent: Path, abl_name: str):
    """找到 outputs_{DATASET}_abl_{VARIANT} 下最新的 test_detail.json"""
    pattern = f"outputs_*_abl_{abl_name}"
    dirs = sorted(parent.parent.glob(pattern))
    if not dirs:
        return None
    abl_dir = dirs[0]
    # 找最新的时间戳子目录中有 test_detail.json 的
    best = None
    for sub in sorted(abl_dir.iterdir(), reverse=True):
        if sub.is_dir() and (sub / "test_detail.json").exists():
            best = sub / "test_detail.json"
            break
    return best


def read_test_metrics(path: Path):
    """从 test_detail.json 读取指标"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for m in METRICS:
        # 兼容不同 key 格式: "Macro-F1" / "macro_f1" / "test_Macro-F1"
        for key in [m, m.lower().replace("-", "_"),
                     f"test_{m}", f"test_{m.lower().replace('-', '_')}"]:
            if key in data:
                result[m] = float(data[key])
                break
        if m not in result:
            result[m] = None
    return result


def one_sample_ttest(values, mu):
    """
    One-sample t-test: H0: mean(values) == mu
    Returns: t_stat, p_value (one-sided, testing mean > mu)
    """
    n = len(values)
    if n < 2:
        return None, None
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(var)
    if std == 0:
        return None, None
    t = (mean - mu) / (std / math.sqrt(n))
    # 用 t 分布近似 p-value (one-sided)
    # 近似: 使用 Student's t CDF (Abramowitz & Stegun 近似)
    p = _t_cdf_approx(-t, n - 1)  # P(T <= -t) for one-sided test of mean > mu
    return t, p


def _t_cdf_approx(t, df):
    """
    近似 Student's t CDF: P(T <= t) for given degrees of freedom.
    使用 regularized incomplete beta function 的近似。
    """
    if df <= 0:
        return 0.5
    x = df / (df + t * t)
    # I_x(a, b) where a = df/2, b = 0.5
    # 对于小 df (4), 用精确的 beta 函数
    a = df / 2.0
    b = 0.5
    ibeta = _reg_inc_beta(x, a, b)
    if t >= 0:
        return 1.0 - 0.5 * ibeta
    else:
        return 0.5 * ibeta


def _reg_inc_beta(x, a, b):
    """Regularized incomplete beta function I_x(a,b) via continued fraction (Lentz)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    # Use Lentz's continued fraction
    ln_beta = _ln_gamma(a) + _ln_gamma(b) - _ln_gamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - ln_beta) / a
    # Continued fraction
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = d
    for m in range(1, 200):
        # even step
        num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + num * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + num / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        f *= c * d
        # odd step
        num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + num / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        delta = c * d
        f *= delta
        if abs(delta - 1.0) < 1e-10:
            break
    return front * f


def _ln_gamma(z):
    """Lanczos approximation of ln(Gamma(z))."""
    g = 7
    c = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ]
    if z < 0.5:
        return math.log(math.pi / math.sin(math.pi * z)) - _ln_gamma(1 - z)
    z -= 1
    x = c[0]
    for i in range(1, g + 2):
        x += c[i] / (z + i)
    t = z + g + 0.5
    return 0.5 * math.log(2 * math.pi) + (z + 0.5) * math.log(t) - t + math.log(x)


def cohens_d(values, mu):
    """Cohen's d effect size."""
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(var)
    if std == 0:
        return None
    return (mean - mu) / std


def sig_stars(p):
    if p is None: return "n/a"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def fmt(v, width=8):
    if v is None:
        return " " * width + "n/a"
    return f"{v:>{width}.4f}"


# ============================================================
# 主逻辑
# ============================================================

def process_dataset(ds_name):
    """处理一个数据集，返回结果字典列表"""
    print(f"\n{'='*70}")
    print(f"  Dataset: {ds_name}")
    print(f"{'='*70}")

    seed_dir = SEED_DIRS[ds_name]

    # 1) 读取 5-seed SEDetector 结果
    seed_dirs = find_seed_dirs(seed_dir)
    if len(seed_dirs) < 5:
        print(f"  WARNING: 只找到 {len(seed_dirs)}/5 个 seed 结果!")
    
    seed_metrics = {m: [] for m in METRICS}
    for seed in SEEDS:
        if seed in seed_dirs:
            metrics = read_test_metrics(seed_dirs[seed] / "test_detail.json")
            for m in METRICS:
                if metrics[m] is not None:
                    seed_metrics[m].append(metrics[m])
            print(f"  Seed {seed}: Macro-F1={metrics.get('Macro-F1', 'N/A')}")

    # 2) 读取 BEST (full SEDetector single-run)
    best_dir = seed_dir / "BEST"
    best_metrics = {}
    if (best_dir / "test_detail.json").exists():
        best_metrics = read_test_metrics(best_dir / "test_detail.json")
        print(f"\n  BEST (single): Macro-F1={best_metrics.get('Macro-F1', 'N/A')}")

    # 3) 读取各消融变体
    abl_results = {}
    for abl in ABL_NAMES:
        abl_path = find_best_abl(seed_dir.parent, abl)
        if abl_path:
            abl_results[abl] = read_test_metrics(abl_path)
        else:
            abl_results[abl] = {m: None for m in METRICS}

    # 4) 统计检验
    results = []

    # -- SEDetector 自身统计 --
    se_mean = {}
    se_std = {}
    for m in METRICS:
        vals = seed_metrics[m]
        n = len(vals)
        if n > 0:
            mean = sum(vals) / n
            var = sum((x - mean) ** 2 for x in vals) / max(n - 1, 1) if n > 1 else 0
            se_mean[m] = mean
            se_std[m] = math.sqrt(var)
        else:
            se_mean[m] = None
            se_std[m] = None

    results.append({
        "variant": "SEDetector (5-seed)",
        "is_main": True,
        "mean": se_mean,
        "std": se_std,
        "n": len(seed_metrics["Macro-F1"]),
        "t_stats": {m: None for m in METRICS},
        "p_values": {m: None for m in METRICS},
        "cohens_d": {m: None for m in METRICS},
        "stars": {m: "" for m in METRICS},
    })

    # -- BEST single run --
    if best_metrics.get("Macro-F1") is not None:
        t_stats = {}
        p_values = {}
        d_vals = {}
        stars = {}
        for m in METRICS:
            if best_metrics[m] is not None and len(seed_metrics[m]) >= 2:
                t, p = one_sample_ttest(seed_metrics[m], best_metrics[m])
                t_stats[m] = t
                p_values[m] = p
                d_vals[m] = cohens_d(seed_metrics[m], best_metrics[m])
                stars[m] = sig_stars(p)
            else:
                t_stats[m] = p_values[m] = d_vals[m] = None
                stars[m] = "n/a"
        results.append({
            "variant": "SEDetector (BEST single)",
            "is_main": False,
            "mean": best_metrics,
            "std": {m: None for m in METRICS},
            "n": 1,
            "t_stats": t_stats,
            "p_values": p_values,
            "cohens_d": d_vals,
            "stars": stars,
        })

    # -- 各消融变体 --
    for abl in ABL_NAMES:
        abl_m = abl_results[abl]
        if abl_m.get("Macro-F1") is None:
            continue
        t_stats = {}
        p_values = {}
        d_vals = {}
        stars = {}
        for m in METRICS:
            if abl_m[m] is not None and len(seed_metrics[m]) >= 2:
                t, p = one_sample_ttest(seed_metrics[m], abl_m[m])
                t_stats[m] = t
                p_values[m] = p
                d_vals[m] = cohens_d(seed_metrics[m], abl_m[m])
                stars[m] = sig_stars(p)
            else:
                t_stats[m] = p_values[m] = d_vals[m] = None
                stars[m] = "n/a"
        results.append({
            "variant": ABL_LABELS[abl],
            "is_main": False,
            "mean": abl_m,
            "std": {m: None for m in METRICS},
            "n": 1,
            "t_stats": t_stats,
            "p_values": p_values,
            "cohens_d": d_vals,
            "stars": stars,
        })

    return results


def write_text_report(all_results, filepath):
    """写纯文本报告"""
    lines = []
    lines.append("=" * 90)
    lines.append("  SEDetector Statistical Significance Test Results")
    lines.append("  One-sample t-test: H0: mu(SEDetector 5-seed) <= mu(baseline)")
    lines.append("  Significance: *** p<0.001  ** p<0.01  * p<0.05  n.s. p>=0.05")
    lines.append("=" * 90)

    for ds_name, results in all_results.items():
        lines.append(f"\n{'─'*90}")
        lines.append(f"  Dataset: {ds_name}")
        lines.append(f"{'─'*90}")

        # Table header
        hdr = f"  {'Variant':<28s}"
        for m in METRICS:
            hdr += f"  {m:>10s}  {'t':>7s}  {'p':>7s}  {'d':>6s}  {'':3s}"
        lines.append(hdr)
        lines.append("  " + "-" * (len(hdr) - 2))

        for r in results:
            row = f"  {r['variant']:<28s}"
            for m in METRICS:
                mean_v = r["mean"].get(m)
                std_v = r["std"].get(m)
                if r["is_main"]:
                    row += f"  {fmt(mean_v)}±" if mean_v is not None else "       n/a "
                    row += f"{fmt(std_v, 6)}" if std_v is not None else "    n/a"
                    row += f"  {'---':>7s}  {'---':>7s}  {'---':>6s}  {'':3s}"
                else:
                    row += f"  {fmt(mean_v)}" if mean_v is not None else "       n/a"
                    row += f"  {'':>13s}"  # no std for single runs
                    t = r["t_stats"].get(m)
                    p = r["p_values"].get(m)
                    d = r["cohens_d"].get(m)
                    s = r["stars"].get(m, "")
                    row += f"  {fmt(t, 7) if t is not None else '    n/a'}"
                    row += f"  {fmt(p, 7) if p is not None else '    n/a'}"
                    row += f"  {fmt(d, 6) if d is not None else '   n/a'}"
                    row += f"  {s:3s}"
            lines.append(row)

    text = "\n".join(lines)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return text


def write_latex(all_results, filepath):
    """写 LaTeX 表格代码"""
    lines = []
    lines.append("% Statistical significance test results")
    lines.append("% One-sample t-test: SEDetector 5-seed mean vs each baseline")
    lines.append("% *** p<0.001  ** p<0.01  * p<0.05")
    lines.append("")

    for ds_name, results in all_results.items():
        lines.append(f"% === {ds_name} ===")
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(rf"\caption{{Statistical significance of SEDetector vs ablation variants on {ds_name}.}}")
        lines.append(r"\label{tab:sig-" + ds_name.lower() + "}")
        lines.append(r"\small")
        lines.append(r"\begin{tabular}{@{}l ccc ccc@{}}")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Variant} & \textbf{Macro-F1} & \textbf{$t$} & \textbf{$p$} & \textbf{Accuracy} & \textbf{$t$} & \textbf{$p$} \\")
        lines.append(r"\midrule")

        for r in results:
            name = r["variant"].replace("_", "\\_")
            if r["is_main"]:
                mf1 = r["mean"].get("Macro-F1")
                sf1 = r["std"].get("Macro-F1")
                acc = r["mean"].get("Accuracy")
                sacc = r["std"].get("Accuracy")
                mf1_str = f"${mf1:.4f} \\pm {sf1:.4f}$" if mf1 and sf1 else "n/a"
                acc_str = f"${acc:.4f} \\pm {sacc:.4f}$" if acc and sacc else "n/a"
                lines.append(rf"\textbf{{{name}}} & {mf1_str} & --- & --- & {acc_str} & --- & --- \\")
            else:
                mf1 = r["mean"].get("Macro-F1")
                t_f1 = r["t_stats"].get("Macro-F1")
                p_f1 = r["p_values"].get("Macro-F1")
                s_f1 = r["stars"].get("Macro-F1", "")
                acc = r["mean"].get("Accuracy")
                t_acc = r["t_stats"].get("Accuracy")
                p_acc = r["p_values"].get("Accuracy")
                s_acc = r["stars"].get("Accuracy", "")

                mf1_str = f"{mf1:.4f}" if mf1 else "n/a"
                t_f1_str = f"{t_f1:.2f}" if t_f1 is not None else "n/a"
                p_f1_str = f"{p_f1:.4f}{s_f1}" if p_f1 is not None else "n/a"
                acc_str = f"{acc:.4f}" if acc else "n/a"
                t_acc_str = f"{t_acc:.2f}" if t_acc is not None else "n/a"
                p_acc_str = f"{p_acc:.4f}{s_acc}" if p_acc is not None else "n/a"

                lines.append(rf"{name} & {mf1_str} & {t_f1_str} & {p_f1_str} & {acc_str} & {t_acc_str} & {p_acc_str} \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

    text = "\n".join(lines)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def write_csv(all_results, filepath):
    """写 CSV 方便后续处理"""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Variant", "N",
                          "Macro-F1_mean", "Macro-F1_std", "t_F1", "p_F1", "cohens_d_F1",
                          "Acc_mean", "Acc_std", "t_Acc", "p_Acc", "cohens_d_Acc",
                          "ROC-AUC_mean", "ROC-AUC_std", "t_AUC", "p_AUC", "cohens_d_AUC"])
        for ds_name, results in all_results.items():
            for r in results:
                row = [ds_name, r["variant"], r["n"]]
                for m in METRICS:
                    row.append(f"{r['mean'].get(m, ''):.4f}" if r['mean'].get(m) is not None else "")
                    row.append(f"{r['std'].get(m, ''):.4f}" if r['std'].get(m) is not None else "")
                    row.append(f"{r['t_stats'].get(m, ''):.4f}" if r['t_stats'].get(m) is not None else "")
                    row.append(f"{r['p_values'].get(m, ''):.6f}" if r['p_values'].get(m) is not None else "")
                    row.append(f"{r['cohens_d'].get(m, ''):.4f}" if r['cohens_d'].get(m) is not None else "")
                writer.writerow(row)


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    print("SEDetector Statistical Significance Test")
    print("=" * 50)
    
    # 检查 scipy 是否可用 (更精确的 p-value)
    try:
        from scipy import stats as sp_stats
        HAS_SCIPY = True
        # 覆盖 t-test 函数
        def one_sample_ttest(values, mu):
            t, p_two = sp_stats.ttest_1samp(values, mu)
            p = p_two / 2 if t > 0 else 1 - p_two / 2  # one-sided
            return t, p
        print("[INFO] scipy 可用，使用 scipy.stats.ttest_1samp")
    except ImportError:
        HAS_SCIPY = False
        print("[INFO] scipy 不可用，使用内置近似 (精度足够)")

    # 依次处理三个数据集
    all_results = OrderedDict()
    for ds in DATASETS:
        all_results[ds] = process_dataset(ds)

    # 输出
    txt_path = OUT_DIR / "significance_results.txt"
    tex_path = OUT_DIR / "significance_latex.txt"
    csv_path = OUT_DIR / "significance_results.csv"

    write_text_report(all_results, txt_path)
    write_latex(all_results, tex_path)
    write_csv(all_results, csv_path)

    print(f"\n{'='*70}")
    print(f"  Output files:")
    print(f"    {txt_path}")
    print(f"    {tex_path}")
    print(f"    {csv_path}")
    print(f"{'='*70}")
    print("\nDone!")
