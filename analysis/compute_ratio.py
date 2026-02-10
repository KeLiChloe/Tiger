import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats



def print_params(data):
        
    # === Extract Experiment Parameters for 'dast' ===

    dast_parameters = {
        'K': data.get('exp_params').get('K'),
        'd': data.get('exp_params').get('d'),
        'X_noise_std_scale': data.get('exp_params').get('X_noise_std_scale'),
        'Y_noise_std_scale': data.get('exp_params').get('Y_noise_std_scale'),
        'alpha_param_range': data.get('exp_params').get('param_range').get('alpha'),
        'beta_param_range': data.get('exp_params').get('param_range').get('beta'),
        'tau_param_range': data.get('exp_params').get('param_range').get('tau'),
        'delta_param_range': data.get('exp_params').get('param_range').get('delta'),
        'x_param_range': data.get('exp_params').get('param_range').get('x_mean'),
        'partial_x' : data.get('exp_params').get('partial_x'),
        'N_segment_size': data.get('exp_params').get('N_segment_size'),
        'implementation_scale': data.get('exp_params').get('implementation_scale', 1),
        'disallowed_ball_radius': data.get('exp_params').get('disallowed_ball_radius'),
        'sequence_seed': data.get('exp_params').get('sequence_seed'),
    }
        

    print("=== Experiment Parameters ===")
    for key, value in dast_parameters.items():
        print(f"{key:>20}: {value}")

def compute_improvement_ratio(data, comparators):
    improvement_ratios = {comp: [] for comp in comparators}

    for i in range(len(data['dast'])):
        
        dast_profit = data['dast'][i]['implementation_profits']
        
        for comp in comparators:
            comp_profit = data[comp][i]['implementation_profits']
            ratio = (dast_profit - comp_profit) / abs(comp_profit)
            improvement_ratios[comp].append(ratio)
    return improvement_ratios

def plot(filtered_ratios):
    import seaborn as sns
    from matplotlib.lines import Line2D
    import pandas as pd
    
    # ==========================================
    # 1. Style Configuration (Matched to Reference)
    # ==========================================
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks", {'axes.grid': True})

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['STIXGeneral', 'Times New Roman', 'Times', 'DejaVu Serif'],
        'axes.labelweight': 'bold',
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.dpi': 300,
    })

    # ==========================================
    # 2. Configuration Maps (Colors & Labels)
    # ==========================================
    # Map your raw keys to the reference colors
    palette = {
        "kmeans-standard": "#CCB974", # Gold
        "gmm-standard":    "#64B5CD", # Light Blue
        "clr-standard":    "#9467BD", # Purple
        "mst":             "#937860", # Brown
        "t_learner":       "#FF7F0E", # Orange
        "s_learner":       "#55A868", # Green
        "x_learner":       "#4EBEC4", # Teal
        "dr_learner":      "#D62728", # Red
        "policy_tree":     "#7F7F7F", # Gray
        "random":          "#8C613C",
    }
    
    # Map raw keys to pretty display names
    label_map = {
        "kmeans-standard": "vs. K-Means",
        "gmm-standard":    "vs. GMM",
        "clr-standard":    "vs. CLR",
        "mst":             "vs. MST",
        "t_learner":       "vs. T-Learner",
        "s_learner":       "vs. S-Learner",
        "x_learner":       "vs. X-Learner",
        "dr_learner":      "vs. DR-Learner",
        "policy_tree":     "vs. Policy Tree",
    }

    # Order of display (optional, can adjust)
    preferred_order = [
        "kmeans-standard", "gmm-standard", "clr-standard", "mst",
        "dr_learner", "s_learner", "t_learner", "x_learner", "policy_tree"
    ]

    # ==========================================
    # 3. Calculate Statistics
    # ==========================================
    summary_data = []

    print("\n📊 Summary Statistics:")
    
    # Iterate through available keys in your data
    for comp in filtered_ratios.keys():
        ratios = filtered_ratios[comp]
        if len(ratios) == 0:
            continue
            
        # 1. Convert to percentage
        ratios_pct = ratios * 100 
        
        # 2. Basic Stats
        n = len(ratios_pct)
        mean = np.mean(ratios_pct)
        std_err = stats.sem(ratios_pct)
        
        # 3. Confidence Interval (95%)
        # Using t-distribution for better accuracy on smaller N, converges to normal on large N
        ci_margin = std_err * stats.t.ppf(0.975, n - 1)
        
        # 4. Significance Test (One-sample t-test against 0 improvement)
        # Null Hypothesis: Mean improvement is 0.
        t_stat, p_val = stats.ttest_1samp(ratios_pct, 0)
        
        # Determine stars
        if p_val < 0.001: star = "***"
        elif p_val < 0.01: star = "**"
        elif p_val < 0.05: star = "*"
        else: star = None

        summary_data.append({
            "key": comp,
            "label": label_map.get(comp, f"vs. {comp}"),
            "mean": mean,
            "ci": ci_margin,
            "p_star": star,
            "n": n
        })
        
        # Print text summary (keeping your original logic)
        print(f"{comp:>20}: {n:3d} runs | Avg: {mean:7.2f}% | 95% CI: [{mean-ci_margin:7.2f}%, {mean+ci_margin:7.2f}%]")

    # Sort data based on preferred order or magnitude
    summary_data.sort(key=lambda x: preferred_order.index(x['key']) if x['key'] in preferred_order else 999)
    
    # Create DataFrame for easier plotting logic
    df = pd.DataFrame(summary_data)

    # ==========================================
    # 4. Plotting
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_positions = np.arange(len(df))
    
    # Grid configuration
    ax.grid(axis="x", color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    # Plot error bars
    for i, row in df.iterrows():
        color = palette.get(row['key'], "#333333") # Default to dark grey if key missing
        
        ax.errorbar(
            x=row['mean'], 
            y=i, 
            xerr=row['ci'], 
            fmt='o', 
            color=color, 
            ecolor=color, 
            capsize=4, 
            elinewidth=2, 
            markersize=8
        )

    # Vertical line at 0 (No improvement)
    ax.axvline(0, color="#E40606", linestyle="--", linewidth=1.6, alpha=0.8)

    # Y-Axis Labels (Name + Significance + N)
    ytick_labels = []
    for _, row in df.iterrows():
        lbl = row['label']
        if row['p_star']:
            lbl += f" ({row['p_star']})"
        if row['n'] < 30: # Optional: warn if N is low
            lbl += f" [n={row['n']}]"
        ytick_labels.append(lbl)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(ytick_labels, fontweight="bold", fontsize=13)
    ax.tick_params(axis="y", length=0) # Hide tick marks
    ax.invert_yaxis() # Top to bottom

    # Labels and Titles
    ax.set_xlabel("Averaged DAST Improvement (%) on Revenue Over Comparators", fontweight="bold", labelpad=20)
    ax.set_title(f"Averaged DAST Improvement (%) across {len(filtered_ratios)} runs", fontweight="bold", pad=30, fontsize=16)

    # Top Annotation Box
    ax.annotate(
        "Positive values (>0%) indicate DAST outperforms comparators",
        xy=(0.5, 1.02),
        xycoords="axes fraction",
        fontsize=12,
        fontweight="bold",
        color="#333333",
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="gray", lw=0.5, alpha=0.8),
    )

    # Clean borders (Seaborn despine)
    sns.despine(left=True, top=True, right=True)

    # Legend
    legend_handles = [
        Line2D([0], [0], color="black", marker="o", linestyle="-", linewidth=2, markersize=8, label="Mean ± 95% CI"),
        Line2D([0], [0], color="none", label="*** p < 0.001\n** p < 0.01\n* p < 0.05"),
        Line2D([0], [0], color="#E40606", linestyle="--", linewidth=1.6, label="No Improvement (0%)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#E0E0E0",
        fancybox=False,
        fontsize=11,
        borderpad=1,
    )
    
    plt.tight_layout()
    plt.savefig("figures/figure_improvement_ratio.png", dpi=300, bbox_inches="tight")

def filter_ratios(improvement_ratios, apply_remove_extreme, apply_sigma_clip):
    # === Simplified Outlier Removal ===
    filtered_ratios = {}

    for comp, ratios in improvement_ratios.items():
        ratios_np = np.array(ratios)

        # --- Step 1: Remove 3 extreme values,  both min and max (OPTIONAL) ---
        if apply_remove_extreme.get(comp, False):
            min_indices = np.argsort(ratios_np)[:3]
            mask = np.ones_like(ratios_np, dtype=bool)
            mask[min_indices] = False
            ratios_np = ratios_np[mask]  
            
            max_indices = np.argsort(ratios_np)[-3:]
            mask = np.ones_like(ratios_np, dtype=bool)
            mask[max_indices] = False
            ratios_np = ratios_np[mask]

        # --- Step 2: Remove values outside 3 sigma (OPTIONAL) ---

        if apply_sigma_clip:
            mean = np.mean(ratios_np)
            std = np.std(ratios_np)
            ratios_np = ratios_np[np.abs(ratios_np - mean) <= 3 * std]

        filtered_ratios[comp] = ratios_np
    return filtered_ratios

# === Load Data ===
file_path = "exp_feb_2026/varying_d/exp_d_03.pkl"  # Make sure this is the correct path on your machine7
with open(file_path, "rb") as f:
    data = pickle.load(f)


# comparators = [ "gmm", "mst", "kmeans", "policy_tree"]
# comparators = ["gmm-standard", "gmm-da"]
# read file and extract comparators
comparators = list(data.keys())
comparators.remove('dast')
comparators.remove('exp_params')
comparators.remove('X_overlap_score')
comparators.remove('y_overlap_score')
comparators.remove('X_y_overlap_score')
comparators.remove('ambiguity_score')
try:
    comparators.remove('seed')
except:
    pass
print(f"Comparators found in data: {comparators}")


apply_remove_extreme = {
                        "gmm-standard": True,  
                        "kmeans-standard": True,
                        "mst": True,
                        "clr-standard": True,
                        "t_learner": True,
                        "x_learner": True,
                        "dr_learner": True,
                        "s_learner": True,
                        "policy_tree": True,
                        }
apply_sigma_clip = False

print_params(data)


ratios = compute_improvement_ratio(data, comparators)
filtered_ratios = filter_ratios(ratios, apply_remove_extreme, apply_sigma_clip)

plot(filtered_ratios)

