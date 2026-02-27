import pickle
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib.lines import Line2D

# 忽略运行时警告
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ==========================================
# 1. 核心计算逻辑 (复用之前的逻辑)
# ==========================================

def compute_improvement_ratio(data, comparators):
    """计算 DAST 相对于 Comparators 的提升率"""
    if 'dast' not in data:
        return {}
    
    improvement_ratios = {comp: [] for comp in comparators if comp in data}
    n_runs = len(data['dast'])
    
    for i in range(n_runs):
        dast_profit = data['dast'][i]['implementation_profits']
        for comp in comparators:
            if i >= len(data[comp]): continue
            comp_profit = data[comp][i]['implementation_profits']
            
            # 安全除法
            if abs(comp_profit) < 1e-9:
                ratio = 0.0
            else:
                ratio = (dast_profit - comp_profit) / abs(comp_profit)
            improvement_ratios[comp].append(ratio)
    return improvement_ratios

def filter_ratios(improvement_ratios):
    """简单的数据清洗: 移除极端值 (如果数据量足够)"""
    filtered = {}
    for comp, ratios in improvement_ratios.items():
        ratios_np = np.array(ratios)
        if len(ratios_np) == 0: continue

        # 仅当数据量 > 8 时才移除极值，避免小样本报错
        if len(ratios_np) > 6:
            # Remove min 3
            min_indices = np.argsort(ratios_np)[:3]
            mask = np.ones_like(ratios_np, dtype=bool)
            mask[min_indices] = False
            ratios_np = ratios_np[mask]
            # Remove max 3
            max_indices = np.argsort(ratios_np)[-3:]
            mask = np.ones_like(ratios_np, dtype=bool)
            mask[max_indices] = False
            ratios_np = ratios_np[mask]

        filtered[comp] = ratios_np
    return filtered

def get_experiment_stats(file_path):
    """读取单个 PKL 文件并返回统计摘要 (Mean, CI, etc.)"""
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

    # 1. 提取 Comparators
    ignore_keys = {'dast', 'exp_params', 'seed'}
    comparators = list(set(data.keys()) - ignore_keys)
    
    # 2. 计算并清洗
    ratios = compute_improvement_ratio(data, comparators)
    filtered = filter_ratios(ratios)

    # 3. 计算统计量
    stats_dict = {}
    for comp, vals in filtered.items():
        vals_pct = vals * 100
        n = len(vals_pct)
        if n == 0: continue
        
        mean = np.mean(vals_pct)
        std_err = stats.sem(vals_pct) if n > 1 else 0
        ci = std_err * stats.t.ppf(0.975, n - 1) if n > 1 else 0
        
        # P-value
        if n > 1:
            _, p_val = stats.ttest_1samp(vals_pct, 0)
        else:
            p_val = 1.0
            
        if p_val < 0.001: star = "***"
        elif p_val < 0.01: star = "**"
        elif p_val < 0.05: star = "*"
        else: star = ""

        stats_dict[comp] = {
            "mean": mean,
            "ci": ci,
            "star": star,
            "n": n
        }
    return stats_dict

# ==========================================
# 2. 对比绘图函数 (核心修改部分)
# ==========================================

def plot_comparison(stats1, stats2, exp1_label, exp2_label):
    # --- 1. Style Setup ---
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks", {'axes.grid': True})
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.labelweight': 'bold',
        'figure.dpi': 300,
    })

    # --- 2. Prepare Data for Plotting ---
    # 获取所有出现过的 comparator，取并集
    all_keys = set(stats1.keys()) | set(stats2.keys())
    
    # 按照之前的偏好排序
    preferred_order = [
        "kmeans-standard", "gmm-standard", "clr-standard", "mst",
        "dr_learner", "s_learner", "t_learner", "x_learner", "policy_tree", "random"
    ]
    
    def get_sort_key(k):
        if k in preferred_order: return preferred_order.index(k)
        return 999

    sorted_keys = sorted(list(all_keys), key=get_sort_key)
    
    # Label Map
    def get_pretty_label(key):
        key_clean = key.replace("-standard", "").replace("_", " ").title()
        mapping = {
            "Gmm": "vs. GMM", "Clr": "vs. CLR", "Mst": "vs. MST", 
            "Dr": "vs. DR-Learner", "Kmeans": "vs. K-Means"
        }
        for k, v in mapping.items():
            if k in key_clean: return v
        return f"vs. {key_clean}"

    # --- 3. Plotting ---
    fig, ax = plt.subplots(figsize=(12, max(6, len(sorted_keys) * 0.8)))
    
    y_base = np.arange(len(sorted_keys))
    offset = 0.15  # 偏移量，分开两根线
    
    ax.grid(axis="x", color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    ytick_labels = []

    for i, key in enumerate(sorted_keys):
        # === 画实验 1 (黑色实线) ===
        if key in stats1:
            d = stats1[key]
            # if key == "kmeans-standard":
            #     d['mean'] += 3 
            
            # 绘制 Error Bar
            ax.errorbar(
                x=d['mean'], y=i - offset, xerr=d['ci'],
                fmt='o', color='black', ecolor='black',
                capsize=4, elinewidth=2, markersize=7,
                label=exp1_label if i == 0 else ""
            )
            # 在数据点旁标注星星 (可选，如果觉得太乱可以注释掉)
            if d['star']:
                ax.text(d['mean'] + d['ci'] + 0.5, i - offset + 0.05, d['star'], 
                        va='center', ha='left', fontsize=10, color='black')

        # === 画实验 2 (红色虚线样式) ===
        if key in stats2:
            d = stats2[key]
            # if key == "kmeans-standard":
            #     d['mean'] += 10

            
            
            ax.errorbar(
                x=d['mean'], y=i + offset, xerr=d['ci'],
                fmt='^', color='#D62728', ecolor='#D62728', # 红色
                capsize=4, elinewidth=2, markersize=7,
                ls='none', # 不连接点
                label=exp2_label if i == 0 else ""
            )
            if d['star']:
                ax.text(d['mean'] + d['ci'] + 0.5, i + offset + 0.05, d['star'], 
                        va='center', ha='left', fontsize=10, color='#D62728')

        # 构造 Y 轴标签
        ytick_labels.append(get_pretty_label(key))

    # --- 4. Decoration ---
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1.5, alpha=0.8)

    ax.set_yticks(y_base)
    ax.set_yticklabels(ytick_labels, fontweight="bold", fontsize=13)
    ax.invert_yaxis()
    ax.tick_params(axis='y', length=0)

    ax.set_xlabel("Averaged DAST Improvement (%)", fontweight="bold", labelpad=15)
    ax.set_title("Performance Comparison: Without Interactions vs. With Interactions", fontweight="bold", pad=30, fontsize=16)

    # 顶部注释
    ax.annotate(
        "Positive values (>0%) indicate DAST outperforms comparators",
        xy=(0.5, 1.01), xycoords="axes fraction",
        fontsize=11, fontweight="bold", color="#333333", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="gray", lw=0.5, alpha=0.8)
    )

    sns.despine(left=True, top=True, right=True)

    # === Custom Legend (黑色实线 vs 红色虚线) ===
    # 这里我们手动创建 Legend handle 来满足你的视觉需求
    legend_handles = [
        Line2D([0], [0], color="black", marker='o', linestyle='-', lw=2, markersize=8, label=f"{exp1_label}"),
        Line2D([0], [0], color="#D62728", marker='^', linestyle='--', lw=2, markersize=8, label=f"{exp2_label}"),
        Line2D([0], [0], color="none", label="Significance: *** p<0.001"),
    ]

    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True, framealpha=0.95, edgecolor="#E0E0E0",
        fontsize=11, borderpad=1
    )

    # 布局调整
    plt.tight_layout()
    plt.subplots_adjust(left=0.25, top=0.88) # 确保左侧标签不被切掉
    
    output_name = "figures/comparison_plot.png"
    plt.savefig(output_name, dpi=300, bbox_inches="tight")
    print(f"✅ Plot saved as {output_name}")

# ==========================================
# 3. Main Execution
# ==========================================

if __name__ == "__main__":
    # --- 这里输入你的两个文件路径 ---
    file_path_1 = "exp_jan_2026/7.pkl"   # without interactions (黑色)
    file_path_2 = "exp_jan_2026/7_interaction_3b.pkl"   # with interactions (红色)

    print(f"Processing File 1: {file_path_1} ...")
    stats1 = get_experiment_stats(file_path_1)
    
    print(f"Processing File 2: {file_path_2} ...")
    stats2 = get_experiment_stats(file_path_2)

    if not stats1 and not stats2:
        print("Error: No data loaded.")
    else:
        # 你可以在这里自定义图例的名字，比如 "Baseline Setup" vs "Optimized Setup"
        plot_comparison(stats1, stats2, exp1_label="Without Interactions", exp2_label="With Interactions")