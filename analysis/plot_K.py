import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
import numpy as np

# ===== 1. Management Science 期刊风格参数 =====
# 参考: https://pubsonline.informs.org/page/mnsc/submission-guidelines
rcParams.update({
    # 字体设置（使用 serif 字体，符合学术期刊标准）
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    
    # 字号设置（标准期刊尺寸）
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    
    # 线条和轴设置
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    
    # 网格设置
    "grid.color": "#CCCCCC",
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.5,
    
    # 图例设置
    "legend.frameon": False,
    "legend.edgecolor": "none",
    
    # 刻度设置
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    
    # PDF输出设置（高质量）
    "pdf.fonttype": 42,  # TrueType fonts
    "ps.fonttype": 42,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ===== 2. 数据 =====
K = list(range(2, 11))
mst   = [27.21, 6.00, 5.89, 9.52, 7.83, 8.49, 3.90, 11.87, 11.66]
kmeans = [4.91, 2.01, 4.67, 8.16, 5.09, 1.27, -2.85, 0.52, -2.77]
gmm   = [5.23, 4.89, 5.80, 19.23, 10.58, 6.47, -0.51, 1.07, 8.94]
clr   = [7.39, 6.37, 7.06, 23.96, 8.20, 8.23, 5.02, 7.45, 8.50]

# ===== 3. 配色方案（与 plot_d 一致） =====
colors = {
    "KMeans": "#007ad1",  # 蓝
    "GMM": "#ff7f0e",     # 橙
    "CLR": "#1b9a1b",     # 绿
    "MST": "#da368d"      # 品红
}

# ===== 4. 绘制图表 =====
# Management Science 推荐：单栏 3.5", 双栏 7.5"
fig, ax = plt.subplots(figsize=(7, 4.5))

# 绘制数据线（统一使用圆形标记）
legend_labels = {
    "KMeans": "v.s. K-Means",
    "GMM": "v.s. GMM",
    "CLR": "v.s. CLR",
    "MST": "v.s. MST"
}

for name, data in [("KMeans", kmeans), ("GMM", gmm), ("CLR", clr), ("MST", mst)]:
    ax.plot(K, data, 
            color=colors[name],
            linewidth=1.5,
            alpha=0.9,
            label=legend_labels[name],
            linestyle="-",
            marker="o",
            markersize=5,
            markerfacecolor=colors[name],
            markeredgewidth=0.5,
            markeredgecolor='white')

# 零线参考（baseline）
ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.6, zorder=1)

# 轴标签（专业表述）
ax.set_xlabel("Number of True Clusters ($K$)", fontsize=11, fontweight='normal')
ax.set_ylabel("DAST Advantage Ratio (%)", fontsize=11, fontweight='normal')

# 标题
ax.set_title("DAST Performance Advantage", fontsize=12, pad=15, fontweight='normal')

# 设置 x 轴刻度
ax.set_xticks(K)
ax.set_xlim(1.5, 10.5)

# y 轴范围（根据数据自动调整，但确保包含0）
y_min = min([min(kmeans), min(gmm), min(clr), min(mst)])
y_max = max([max(kmeans), max(gmm), max(clr), max(mst)])
y_margin = (y_max - y_min) * 0.1
ax.set_ylim(y_min - y_margin, y_max + y_margin)

# 图例（期刊标准位置）
legend = ax.legend(
    loc="best",  # 自动选择最佳位置
    ncol=1,
    frameon=True,
    fancybox=False,
    shadow=False,
    framealpha=0.95,
    edgecolor='#CCCCCC',
    facecolor='white',
    handlelength=2.0,
    handleheight=1.0,
    handletextpad=0.5,
    columnspacing=1.0,
    borderpad=0.5,
    title="DAST Performance Advantage"
)
legend.get_frame().set_linewidth(0.8)
# 设置图例标题的字体
legend.get_title().set_fontsize(10)
legend.get_title().set_fontweight('normal')

# 网格（仅 y 轴，不干扰数据）
ax.grid(True, axis="y", alpha=0.3, zorder=0)

# 优化布局
plt.tight_layout()

# ===== 5. 保存高质量图片（期刊要求） =====
# 保存为多种格式
plt.savefig("figure_K_comparison.pdf", format='pdf', dpi=600, bbox_inches='tight')
plt.savefig("figure_K_comparison.png", format='png', dpi=600, bbox_inches='tight')

print("✓ Figures saved in publication quality:")
print("  - figure_K_comparison.pdf (recommended)")
print("  - figure_K_comparison.png (for review)")

# plt.show()
