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
file_paths_sca = [
    "../output/1757557005.1038802.csv",
    "../output/1757562691.2759552.csv",
    "../output/1757555383.5267708.csv",
]
file_paths_eca = [
    "../output/1757557005.1038802.csv",
    "../output/1757562691.2759552.csv",
    "../output/1757555383.5267708.csv",
]
file_path = "../output/1757555383.5267708.csv"
labels = ["w/o SCA", "w/o ECA", "w/ ALL"]
color_map= {"w/o SCA": "tab:orange", "w/o ECA": "tab:purple", "w/ ALL": "tab:green"}


# 读取三个 DataFrame
dfs_sca = [pd.read_csv(path) for path in file_paths_sca]
dfs_eca = [pd.read_csv(path) for path in file_paths_eca]
df = pd.read_csv(file_path)

# 2. 指定整个 Figure 的大小：宽 12 英寸，高 5 英寸
fig, (ax_sca, ax_eca) = plt.subplots(
    nrows=2, ncols=1,
    figsize=(12, 12),   # ← 修改这里来控制图像总体尺寸
    sharex=False
)

# ----- 第一张子图：time vs gamma -----
for df, label in zip(dfs_sca, labels):
    ax_sca.plot(
        df["time"], df["dist"],
        label=label,
        linewidth=4,
        color=color_map[label],
    )

ax_sca.axhline(y=0.0, color="red", linestyle="--", linewidth=4)
ax_sca.text(
    x=ax_sca.get_xlim()[1] - 0.3 * (ax_sca.get_xlim()[1] - ax_sca.get_xlim()[0]),
    y=0.0015,
    s=r"collision",
    color="red",
    fontsize=35  # 这里单独指定“阈值”注释的字体大小
)
ax_sca.set_ylim(-0.01, 0.1)
ax_sca.set_xlim(0, 6)
ax_sca.set_xlabel(r"Time [s]")
ax_sca.set_ylabel(r"SCA Dist. [m]")

# 下面演示两种设置标题大小的方法，任选其一：
# 方法 A：直接在 set_title 里传 fontsize
# ax_gamma.set_title("Time vs. Gamma", fontsize=26)

# 方法 B（已通过 rcParams 设置全局 axes.titlesize=28，调用时不需再传 fontsize）
# ax_gamma.set_title("Time vs. Gamma")

ax_sca.legend(loc="upper right")
ax_sca.grid(linestyle=":", alpha=0.6)

# ----- 第一张子图：time vs gamma -----
for df, label in zip(dfs_eca, labels):
    ax_eca.plot(
        df["time"], df["real_dist"]-0.05,
        label=label,
        linewidth=4,
        color=color_map[label],
    )

# 在 gamma = 2.5 处画红虚线并标注
ax_eca.axhline(y=0.0, color="red", linestyle="--", linewidth=4)
ax_eca.text(
    x=ax_eca.get_xlim()[1] - 0.3 * (ax_eca.get_xlim()[1] - ax_eca.get_xlim()[0]),
    y=0.006,
    s=r"collision",
    color="red",
    fontsize=35  # 这里单独指定“阈值”注释的字体大小
)
ax_eca.set_ylim(-0.01, 0.4)
ax_eca.set_xlim(0, 6)
ax_eca.set_xlabel(r"Time [s]")
ax_eca.set_ylabel(r"ECA Dist. [m]")


ax_eca.legend(loc="upper right")
ax_eca.grid(linestyle=":", alpha=0.6)



# # ----- 第二张子图：time vs dist -----

# ax_dist.plot(
#     df["time"], df["dist"],
#     label=labels[0],
#     linewidth=4,
#     color=color_map[labels[0]]
# )
# ax_dist.plot(
#     df["time"], df["real_dist"],
#     label=labels[1],
#     linewidth=4,
#     color=color_map[labels[1]]
# )

# ax_dist.axhline(y=0, color="red", linestyle="--", linewidth=4)
# ax_dist.text(
#     x=ax_dist.get_xlim()[1] - 0.95 * (ax_dist.get_xlim()[1] - ax_dist.get_xlim()[0]),
#     y=0.0096,
#     s=r"collision",
#     color="red",
#     fontsize=35  # 这里单独指定“阈值”注释的字体大小
# )
# ax_dist.set_ylim(-0.01, 0.3)
# ax_dist.set_xlim(0, 6)
# ax_dist.set_xlabel(r"Time [s]")
# ax_dist.set_ylabel(r"Distance [m]")

# # 同样可以在这里传 fontsize 或者依赖 rcParams 中的 axes.titlesize
# # ax_dist.set_title("Time vs. Self-Collision Distance", fontsize=26)
# # 或者： ax_dist.set_title("Time vs. Minimum Distance")

# ax_dist.legend(loc="upper right")
# ax_dist.grid(linestyle=":", alpha=0.6)

# 调整子图间距，避免重叠
plt.tight_layout()

# 保存图像（DPI 可根据需要调整）
plt.savefig("../output/plot_all2.png", dpi=1300, bbox_inches="tight")
plt.show()

