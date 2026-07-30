# -*- coding: utf-8 -*-
# table10.py
#
# Table 10 -> compact square heatmaps
#
# Cell value = Metric value / Random baseline
#
# DAPT2020 row: SEU_strong representation (4-dim hash + Laplacian noise)
#
# Outputs:
#   table10.pdf
#   table10.png

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable


# =========================================================
# 全局样式
# =========================================================
try:
    font_manager.findfont(
        "Times New Roman",
        fallback_to_default=False
    )
    mpl.rcParams["font.family"] = "Times New Roman"
except ValueError:
    mpl.rcParams["font.family"] = "DejaVu Serif"

mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["axes.unicode_minus"] = False


# =========================================================
# 输出路径
# =========================================================
OUT_PDF = "./table10.pdf"
OUT_PNG = "./table10.png"


# =========================================================
# 固定顺序
# =========================================================
DATASETS = ["OpTC", "TCE5", "DAPT2020"]
WINDOWS = ["60", "300", "900"]


# =========================================================
# Table 10 数据
# =========================================================
LINKABILITY_METRIC = np.array([
    [1.25e-3, 2.83e-3, 1.41e-2],   # OpTC
    [5.59e-5, 2.72e-4, 6.62e-4],   # TCE5
    [2.7556e-3, 1.2757e-2, 3.2058e-2],  # DAPT2020 (SEU_strong)
])

LINKABILITY_BASELINE = np.array([
    [1.41e-3, 4.59e-3, 1.37e-2],
    [5.00e-4, 5.00e-4, 9.60e-4],
    [1.66e-3, 8.26e-3, 2.44e-2],
])

SINGLING_METRIC = np.array([
    [0.00,    3.33e-4, 8.28e-4],   # OpTC
    [0.00,    2.67e-5, 0.00],      # TCE5
    [0.00,    0.00,    0.00],      # DAPT2020 (SEU_strong)
])

SINGLING_BASELINE = np.full(
    shape=(3, 3),
    fill_value=3.68e-1
)


# =========================================================
# 计算相对随机基线的比例
# =========================================================
LINKABILITY_RATIO = (
    LINKABILITY_METRIC / LINKABILITY_BASELINE
)

SINGLING_RATIO = (
    SINGLING_METRIC / SINGLING_BASELINE
)


# =========================================================
# 转换为 log10 比例
#
# 0 映射到最低显示值 10^-4，
# 但单元格中仍显示为 0
# =========================================================
LOG_MIN = -4.0
LOG_MAX = np.log10(3.2)

link_safe = np.maximum(
    LINKABILITY_RATIO,
    10 ** LOG_MIN
)

sing_safe = np.maximum(
    SINGLING_RATIO,
    10 ** LOG_MIN
)

LINKABILITY_LOG = np.log10(link_safe)
SINGLING_LOG = np.log10(sing_safe)


# =========================================================
# 颜色设置
#
# log10(ratio) = 0 表示实测值等于随机基线
# =========================================================
NORM = TwoSlopeNorm(
    vmin=LOG_MIN,
    vcenter=0.0,
    vmax=LOG_MAX
)

CMAP = "RdBu_r"


# =========================================================
# 字号
# =========================================================
TITLE_FS = 16
AXIS_FS = 15
TICK_FS = 14
CELL_FS = 14
COLORBAR_FS = 14
COLORBAR_TICK_FS = 14


# =========================================================
# 标签格式
# =========================================================
def format_ratio(value):
    """
    显示相对于随机基线的倍数。
    """
    if value == 0:
        return "0"

    if value >= 0.1:
        return f"{value:.2f}" + r"$\times$"

    exponent = int(np.floor(np.log10(value)))
    coefficient = value / (10 ** exponent)

    return (
        rf"${coefficient:.2f}\times10^{{{exponent}}}$"
    )


def get_text_color(log_value):
    """
    深色背景使用白字，浅色背景使用黑字。
    """
    if log_value <= -1.7 or log_value >= 0.32:
        return "white"

    return "black"


# =========================================================
# 作图
# =========================================================
fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5.6)
)

PLOTS = [
    (
        axes[0],
        LINKABILITY_LOG,
        LINKABILITY_RATIO,
        "(a) Linkability"
    ),
    (
        axes[1],
        SINGLING_LOG,
        SINGLING_RATIO,
        "(b) Singling Out"
    ),
]


for ax, log_values, ratio_values, title in PLOTS:

    ax.imshow(
        log_values,
        cmap=CMAP,
        norm=NORM,
        interpolation="nearest",
        aspect="equal"
    )

    # 强制整个坐标区域为正方形
    ax.set_box_aspect(1)

    ax.set_title(
        title,
        fontsize=TITLE_FS,
        pad=12
    )

    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(
        WINDOWS,
        fontsize=TICK_FS
    )

    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(
        DATASETS,
        fontsize=TICK_FS
    )

    ax.set_xlabel(
        "Window (s)",
        fontsize=AXIS_FS,
        labelpad=7
    )

    # 单元格分割线
    ax.set_xticks(
        np.arange(-0.5, 3, 1),
        minor=True
    )
    ax.set_yticks(
        np.arange(-0.5, 3, 1),
        minor=True
    )

    ax.grid(
        which="minor",
        color="white",
        linewidth=2.0
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False
    )

    ax.tick_params(
        axis="both",
        which="major",
        width=1.0,
        length=4
    )

    # 每个单元格只保留一个比例值
    for row in range(3):
        for col in range(3):

            ratio = ratio_values[row, col]
            log_ratio = log_values[row, col]

            ax.text(
                col,
                row,
                format_ratio(ratio),
                ha="center",
                va="center",
                fontsize=CELL_FS,
                color=get_text_color(log_ratio)
            )

    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_linewidth(1.0)
        ax.spines[side].set_color("black")


# =========================================================
# 公共颜色条：改为右侧纵向放置
# =========================================================
scalar_map = ScalarMappable(
    norm=NORM,
    cmap=CMAP
)
scalar_map.set_array([])

# [left, bottom, width, height]
colorbar_ax = fig.add_axes([
    0.915,
    0.205,
    0.018,
    0.605
])

colorbar = fig.colorbar(
    scalar_map,
    cax=colorbar_ax,
    orientation="vertical"
)

colorbar.set_ticks([
    -4,
    -3,
    -2,
    -1,
    0,
    np.log10(3.2)
])

colorbar.set_ticklabels([
    r"$10^{-4}$",
    r"$10^{-3}$",
    r"$10^{-2}$",
    r"$10^{-1}$",
    r"$1$",
    r"$3.2$"
])

colorbar.ax.tick_params(
    labelsize=COLORBAR_TICK_FS,
    width=1.0,
    length=3
)

colorbar.set_label(
    "Observed / random baseline ($\\times$)",
    fontsize=COLORBAR_FS,
    rotation=90,
    labelpad=10
)


# =========================================================
# 布局与保存
# =========================================================
plt.subplots_adjust(
    left=0.075,
    right=0.885,
    top=0.88,
    bottom=0.15,
    wspace=0.28
)

plt.savefig(
    OUT_PDF,
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    OUT_PNG,
    dpi=600,
    bbox_inches="tight"
)

plt.show()

print("Saved:", OUT_PDF)
print("Saved:", OUT_PNG)
