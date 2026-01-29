import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 全局字体配置（也可以在这里设置标题默认大小）
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{times}",  # 使用 Times 字体
    # 以下是可选的全局字体大小设置：
    "axes.labelsize": 35,      # 坐标轴标签大小
    "axes.titlesize": 28,      # 标题默认大小
    "xtick.labelsize": 35,     # x 轴刻度字体大小
    "ytick.labelsize": 35,     # y 轴刻度字体大小
    "legend.fontsize": 30      # 图例字体大小
})

# 三个 CSV 文件的路径
file_paths = [
    "../output/1757642818.1116743.csv",
    "../output/1757642598.070661.csv"
]
labels = ["w/o ECA", "w/ ECA"]
color_map = {"w/o ECA": "tab:orange", "w/ ECA": "tab:green"}

# 读取三个 DataFrame
dfs = [pd.read_csv(path) for path in file_paths]

# 2. 指定整个 Figure 的大小：宽 12 英寸，高 5 英寸
fig, (ax_gamma, ax_dist) = plt.subplots(
    nrows=2, ncols=1,
    figsize=(12, 12),   # ← 修改这里来控制图像总体尺寸
    sharex=False
)

# ----- 第一张子图：time vs gamma -----
for df, label in zip(dfs, labels):
    ax_gamma.plot(
        df["time"], df["pred_distv"]-0.05,
        label=label,
        linewidth=4,
        color=color_map[label],
    )

# 在 gamma = 2.5 处画红虚线并标注
ax_gamma.axhline(y=0.05, color="red", linestyle="--", linewidth=4)
ax_gamma.text(
    x=4 - 0.3 * (4 - ax_gamma.get_xlim()[0]),
    y=0.05 + 0.0048,
    s=r"threshold",
    color="red",
    fontsize=35  # 这里单独指定“阈值”注释的字体大小
)
ax_gamma.set_ylim(0, 0.3)
ax_gamma.set_xlim(0, 4)
ax_gamma.set_xlabel(r"Time [s]")
ax_gamma.set_ylabel(r"$S_v(p, q,\dot q)$")

# 下面演示两种设置标题大小的方法，任选其一：
# 方法 A：直接在 set_title 里传 fontsize
# ax_gamma.set_title("Time vs. Gamma", fontsize=26)

# 方法 B（已通过 rcParams 设置全局 axes.titlesize=28，调用时不需再传 fontsize）
# ax_gamma.set_title("Time vs. Gamma")

ax_gamma.legend(loc="upper right")
ax_gamma.grid(linestyle=":", alpha=0.6)

# ----- 第二张子图：time vs dist -----
for df, label in zip(dfs, labels):
    ax_dist.plot(
        df["time"], df["real_dist"]-0.05,
        label=label,
        linewidth=4,
        color=color_map[label]
    )

ax_dist.axhline(y=0, color="red", linestyle="--", linewidth=4)
ax_dist.text(
    x=4 - 0.3 * (4 - ax_dist.get_xlim()[0]),
    y=0.0048,
    s=r"collision",
    color="red",
    fontsize=35  # 这里单独指定“阈值”注释的字体大小
)
ax_dist.set_ylim(-0.01, 0.3)
ax_dist.set_xlim(0, 4)
ax_dist.set_xlabel(r"Time [s]")
ax_dist.set_ylabel(r"Distance [m]")

# 同样可以在这里传 fontsize 或者依赖 rcParams 中的 axes.titlesize
# ax_dist.set_title("Time vs. Self-Collision Distance", fontsize=26)
# 或者： ax_dist.set_title("Time vs. Minimum Distance")

ax_dist.legend(loc="upper right")
ax_dist.grid(linestyle=":", alpha=0.6)

# 调整子图间距，避免重叠
plt.tight_layout()

# 保存图像（DPI 可根据需要调整）
plt.savefig("../output/plot_eca.png", dpi=1300, bbox_inches="tight")
plt.show()

