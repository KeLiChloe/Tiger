import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

remove_3sigma = True 

# ===============================
# 1. 设置论文风格参数
# ===============================
rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "grid.color": "gray",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
})

# ===============================
# 2. 数据
# ===============================
data_remove_sigma = {
    "Percentage": [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    
    "Kmeans_mean_pct": [9.97, 22.96, 8.31, 2.60, 3.49, 1.67, 0.18, -2.18, -3.03],
    "Kmeans_std_unit": [0.36, 0.71, 0.18, 0.16, 0.18, 0.15, 0.11, 0.18, 0.10],
    
    "GMM_mean_pct": [8.71, 20.97, 9.41, 4.18, 2.79, 0.98, -0.91, -2.94, -3.26],
    "GMM_std_unit": [0.27, 0.56, 0.23, 0.15, 0.20, 0.18, 0.05, 0.17, 0.10],
    
    "CLR_mean_pct": [6.46, 13.38, 13.65, 18.39, 24.21, 22.34, 30.60, 13.05, 20.29],
    "CLR_std_unit": [0.46, 0.33, 0.20, 0.33, 0.48, 0.57, 1.39, 0.18, 0.44],
    
    "MST_mean_pct": [-6.19, 0.10, 2.48, 2.97, 4.69, 8.09, 4.15, 2.31, 5.04],
    "MST_std_unit": [0.25, 0.36, 0.13, 0.19, 0.13, 0.29, 0.11, 0.05, 0.15],
}



data_raw = {
    "Percentage": [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    
    "Kmeans_mean_pct": [16.66, 76.92, 13.88, 7.48, 5.22, 8.19, -2.04, -22.65, -73.56],
    "Kmeans_std_unit": [1.09, 6.97, 0.37, 0.39, 0.49, 0.89, 0.32, 2.7, 9.33],
    
    "GMM_mean_pct": [17.52, 36.47, 15.74, 8.42, 6.47, 5.37, -2.99, -11.73, -11.80],
    "GMM_std_unit": [0.65, 1.24, 0.58, 0.40, 0.85, 0.62, 0.23, 1.17, 1.13],
    
    "CLR_mean_pct": [-15.46, 16.23, 19.70, 22.17, 41.81, 47.02, 73.43, 14.77, 38.02],
    "CLR_std_unit": [2.78, 0.52, 0.42, 0.65, 1.71, 2.70, 5.91, 0.24, 1.72],
    
    "MST_mean_pct": [-2.64, 26.87, 4.07, 11.11, 6.15, 17.00, 4.71, 3.41, 7.61],
    "MST_std_unit": [0.42, 3.05, 0.30, 1.10, 0.24, 0.96, 0.23, 0.10, 0.37],
}

if remove_3sigma:
    data = data_remove_sigma
else:
    data = data_raw



df = pd.DataFrame(data)

colors = {
    "Kmeans": "#007ad1",  # 蓝
    "GMM": "#ff7f0e",     # 橙
    "CLR": "#1b9a1b",     # 绿
    "MST": "#da368d"      # 灰
}

# ===============================
# 3. 单独图（带 ±1 std）
# ===============================
def plot_single(df, name, save_path=None, color=None):
    x = np.array(df["Percentage"], dtype=float)
    mean_pct = np.array(df[f"{name}_mean_pct"], dtype=float)
    std_pct = np.array(df[f"{name}_std_unit"], dtype=float) * 100.0  # 转换成百分点

    plt.figure(figsize=(8, 5))
    plt.plot(x, mean_pct, color=color, marker="o", linewidth=2)
    plt.fill_between(x, mean_pct - std_pct, mean_pct + std_pct, color=color, alpha=0.15)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)

    plt.xlabel("Ratio of Signaling Features")
    plt.ylabel("DAST Profit Advantage (%)")
    plt.title(f"DAST Profit Advantage (%) over {name} ±1 Std")
    plt.grid(True, axis="y", alpha=0.4)
    # plt.legend([f"DAST Profit Advantage (%) over {name} ±1 Std"], loc="upper right", frameon=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # plt.show()



save_dir = "figures"
for comp in ["Kmeans", "GMM", "CLR", "MST"]:
    if remove_3sigma:
        plot_single(df, comp, save_path=f"{save_dir}/signal_to_ratio_{comp}_3sigma.png", color=colors[comp])
    else:
        plot_single(df, comp, save_path=f"{save_dir}/signal_to_ratio_{comp}_raw.png", color=colors[comp])
# ===============================
# 4. 汇总图（无 std）
# ===============================
plt.figure(figsize=(8.5, 5.5))
for comp in ["Kmeans", "GMM", "CLR", "MST"]:
    plt.plot(df["Percentage"], df[f"{comp}_mean_pct"], color=colors[comp],
             marker="o", linewidth=1.5, label=comp)

plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Ratio of Signaling Features")
plt.ylabel("DAST Profit Advantage (%)")
# Title (稍微上移，为 legend 留空间)
plt.title("DAST Profits Advantage across Comparators", pad=15)

# Legend 放在标题下方（中心对齐）
plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.05),  # 数值 <1 表示在标题下方
    ncol=4,
    frameon=False,
    handletextpad=0.5,
    columnspacing=1.2,
)

plt.grid(True, axis="y", alpha=0.4)
# plt.legend(title="Comparator", loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=4, frameon=False)
plt.tight_layout()
if remove_3sigma:
    plt.savefig(f"{save_dir}/signal_to_ratio_all_3sigma.png", dpi=300, bbox_inches="tight")
else:
    plt.savefig(f"{save_dir}/signal_to_ratio_all_raw.png", dpi=300, bbox_inches="tight")
# plt.show()
