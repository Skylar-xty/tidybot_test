import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 全局字体配置（也可以在这里设置标题默认大小）
plt.rcParams.update({
    "text.usetex": False,
    # 以下是可选的全局字体大小设置：
    "axes.labelsize": 24,      # 坐标轴标签大小
    "axes.titlesize": 28,      # 标题默认大小
    "xtick.labelsize": 20,     # x 轴刻度字体大小
    "ytick.labelsize": 20,     # y 轴刻度字体大小
    "legend.fontsize": 20      # 图例字体大小
})

# # 三个 CSV 文件的路径
# file_paths = [
#     "../output/both.csv",
#     "../output/both.csv",
#     "../output/both.csv"
# ]
# labels = ["none", "C1 only", "both"]

# # 读取三个 DataFrame
# dfs = [pd.read_csv(path) for path in file_paths]

# # 2. 指定整个 Figure 的大小：宽 12 英寸，高 5 英寸
# fig, (ax_gamma, ax_dist) = plt.subplots(
#     nrows=1, ncols=2,
#     figsize=(24, 5),   # ← 修改这里来控制图像总体尺寸
#     sharex=False
# )

# # ----- 第一张子图：time vs gamma -----
# for df, label in zip(dfs, labels):
#     ax_gamma.plot(
#         df["time"], df["gamma"],
#         label=label,
#         linewidth=2
#     )

# # 在 gamma = 2.5 处画红虚线并标注
# ax_gamma.axhline(y=2.5, color="red", linestyle="--", linewidth=2)
# ax_gamma.text(
#     x=ax_gamma.get_xlim()[1] - 0.3 * (ax_gamma.get_xlim()[1] - ax_gamma.get_xlim()[0]),
#     y=2.5 + 0.3,
#     s=r"threshold = $2.5$",
#     color="red",
#     fontsize=20  # 这里单独指定“阈值”注释的字体大小
# )

# ax_gamma.set_xlabel(r"time (s)")
# ax_gamma.set_ylabel(r"$\Gamma$")

# # 下面演示两种设置标题大小的方法，任选其一：
# # 方法 A：直接在 set_title 里传 fontsize
# ax_gamma.set_title("Time vs. Gamma", fontsize=26)

# # 方法 B（已通过 rcParams 设置全局 axes.titlesize=28，调用时不需再传 fontsize）
# # ax_gamma.set_title("Time vs. Gamma")

# ax_gamma.legend(loc="upper right")
# ax_gamma.grid(linestyle=":", alpha=0.6)

# # ----- 第二张子图：time vs dist -----
# for df, label in zip(dfs, labels):
#     ax_dist.plot(
#         df["time"], df["dist"],
#         label=label,
#         linewidth=2
#     )

# ax_dist.axhline(y=0, color="red", linestyle="--", linewidth=2)

# ax_dist.set_xlabel(r"time (s)")
# ax_dist.set_ylabel(r"$d_{\min}$")

# # 同样可以在这里传 fontsize 或者依赖 rcParams 中的 axes.titlesize
# ax_dist.set_title("Time vs. Minimum Distance", fontsize=26)
# # 或者： ax_dist.set_title("Time vs. Minimum Distance")

# ax_dist.legend(loc="upper right")
# ax_dist.grid(linestyle=":", alpha=0.6)

# # 调整子图间距，避免重叠
# plt.tight_layout()

# # 保存图像（DPI 可根据需要调整）
# plt.savefig("../output/comparison_plot.png", dpi=1300, bbox_inches="tight")
# # plt.show()

# 三个 CSV 文件的路径
file_path = "../output/1753337640.9479146.csv"
# labels = ["tar", "C1 only", "both"]

# 读取三个 DataFrame
df = pd.read_csv(file_path)

# 2. 指定整个 Figure 的大小：宽 12 英寸，高 5 英寸
fig, (ax_gamma, ax_dist) = plt.subplots(
    nrows=1, ncols=2,
    figsize=(24, 5),   # ← 修改这里来控制图像总体尺寸
    sharex=False
)

# ----- 第一张子图：time vs gamma -----
ax_gamma.plot(
        df["time"], df["tar_dist"],
        linewidth=2
    )

# 在 gamma = 2.5 处画红虚线并标注
# ax_gamma.axhline(y=2.5, color="red", linestyle="--", linewidth=2)
# ax_gamma.text(
#     x=ax_gamma.get_xlim()[1] - 0.3 * (ax_gamma.get_xlim()[1] - ax_gamma.get_xlim()[0]),
#     y=2.5 + 0.3,
#     s=r"threshold = $2.5$",
#     color="red",
#     fontsize=20  # 这里单独指定“阈值”注释的字体大小
# )

ax_gamma.set_xlabel(r"time (s)")
ax_gamma.set_ylabel(r"Dist to Target (m)")

# 下面演示两种设置标题大小的方法，任选其一：
# 方法 A：直接在 set_title 里传 fontsize
ax_gamma.set_title("Time vs. Distance to Target", fontsize=26)

# 方法 B（已通过 rcParams 设置全局 axes.titlesize=28，调用时不需再传 fontsize）
# ax_gamma.set_title("Time vs. Gamma")

# ax_gamma.legend(loc="upper right")
ax_gamma.grid(linestyle=":", alpha=0.6)

# ----- 第二张子图：time vs dist -----
ax_dist.plot(
        df["time"], df["real_dist"],
        label="Real Dist",
        linewidth=2
    )
ax_dist.plot(
        df["time"], df["pred_dist"],
        label="Pred Dist",
        linewidth=2
    )
ax_dist.plot(
        df["time"], df["pred_distv"],
        label="Pred Dist Viability",
        linewidth=2,
        linestyle="--"
    )
ax_dist.axhline(y=0.10, color="green", linestyle="--", linewidth=2)
ax_dist.axhline(y=0, color="red", linestyle="--", linewidth=2)

ax_dist.set_xlabel(r"time (s)")
ax_dist.set_ylabel(r"Dist to Obstacle (m)")

# 同样可以在这里传 fontsize 或者依赖 rcParams 中的 axes.titlesize
ax_dist.set_title("Time vs. Distance to Obstacle", fontsize=26)
# 或者： ax_dist.set_title("Time vs. Minimum Distance")

ax_dist.legend(loc="upper right")
ax_dist.grid(linestyle=":", alpha=0.6)

# 调整子图间距，避免重叠
plt.tight_layout()

# 保存图像（DPI 可根据需要调整）
plt.savefig("../output/comparison_plotntau2.png", dpi=1300, bbox_inches="tight")
plt.show()
