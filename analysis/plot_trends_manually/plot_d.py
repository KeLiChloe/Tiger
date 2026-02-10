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
d = list(range(1, 21))

MST = [
    4.47, 6.42, 9.15, 7.07, 6.86, 9.74, 6.84, 10.87, 6.31, 7.54,
    3.29, 3.72, 5.89, 3.13, 2.96, 2.37, 2.02, 2.03, 1.10, 2.15
]

KMeans = [
    10.50, 5.43, 7.01, 6.41, 6.70, 5.08, 2.52, 3.49, 2.29, 1.29,
    2.92, 1.12, 2.03, -0.34, 1.74, 0.21, -0.47, -0.69, -0.17, -1.10
]

GMM = [
    7.20, 9.63, 4.87, 7.82, 10.67, 7.29, 8.51, 8.13, 7.19, 4.34,
    4.26, 4.30, 1.95, 1.78, 3.52, 1.34, -0.42, 0.36, 2.22, -0.73
]

CLR = [
    5.26, 4.86, 9.39, 15.98, 9.92, 8.60, 11.51, 19.02, 11.01, 7.17,
    8.87, 8.12, 7.51, 4.25, 2.57, 10.01, 3.57, 2.62, 2.85, 3.68
]

# ===== 3. 配色方案（原始配色） =====
colors = {
    "KMeans": "#007ad1",  # 蓝
    "GMM": "#ff7f0e",     # 橙
    "CLR": "#1b9a1b",     # 绿
    "MST": "#da368d"      # 品红
}

# ===== 4. 绘制图表（单栏宽度：3.5英寸） =====
# Management Science 推荐：单栏 3.5", 双栏 7.5"
fig, ax = plt.subplots(figsize=(7, 4.5))

# 绘制数据线（统一使用圆形标记）
legend_labels = {
    "KMeans": "v.s. K-Means",
    "GMM": "v.s. GMM",
    "CLR": "v.s. CLR",
    "MST": "v.s. MST"
}

for name, data in [("KMeans", KMeans), ("GMM", GMM), ("CLR", CLR), ("MST", MST)]:
    ax.plot(d, data, 
            color=colors[name],
            linewidth=1.5,
            alpha=0.9,
            label=legend_labels[name],
            linestyle="-",
            marker="o",
            markersize=4,
            markerfacecolor=colors[name],
            markeredgewidth=0.5,
            markeredgecolor='white')

# 零线参考（baseline）
ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.6, zorder=1)

# 轴标签（专业表述）
ax.set_xlabel("Covariate Dimension ($d$)", fontsize=11, fontweight='normal')
ax.set_ylabel("DAST Advantage Ratio (%)", fontsize=11, fontweight='normal')

# 标题
ax.set_title("DAST Performance Advantage", fontsize=12, pad=15, fontweight='normal')

# 设置 x 轴刻度
ax.set_xticks(range(2, 21, 2))
ax.set_xlim(0.5, 20.5)

# y 轴范围（根据数据自动调整，但确保包含0）
y_min = min([min(KMeans), min(GMM), min(CLR), min(MST)])
y_max = max([max(KMeans), max(GMM), max(CLR), max(MST)])
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
plt.savefig("figures/figure_dimension_comparison.pdf", format='pdf', dpi=600, bbox_inches='tight')
plt.savefig("figures/figure_dimension_comparison.png", format='png', dpi=600, bbox_inches='tight')

print("✓ Figures saved in publication quality:")
print("  - figure_dimension_comparison.pdf (recommended)")
print("  - figure_dimension_comparison.png (for review)")

# plt.show()
