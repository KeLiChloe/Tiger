import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats



def print_params(data):
        
    # === Extract Experiment Parameters for 'dast' ===
    if data.get('exp_params') is None:
        dast_parameters = {
                'K': data.get('K'),
                'd': data.get('d'),
                'X_noise_std_scale': data.get('X_noise_std_scale'),
                'Y_noise_std_scale': data.get('Y_noise_std_scale'),
                'tau_param_range': data.get('param_range').get('tau'),
                'x_param_range': data.get('param_range').get('x_mean'),
                'partial_x' : data.get('partial_x'),
                'N_segment_size': data.get('N_segment_size'),
                'implementation_scale': data.get('implementation_scale', 1),
                'disallowed_ball_radius': data.get('disallowed_ball_radius'),
                'seed': data.get('seed'),
                
        }
    else:
        dast_parameters = {
            'K': data.get('exp_params').get('K'),
            'd': data.get('exp_params').get('d'),
            'X_noise_std_scale': data.get('exp_params').get('X_noise_std_scale'),
            'Y_noise_std_scale': data.get('exp_params').get('Y_noise_std_scale'),
            'tau_param_range': data.get('exp_params').get('param_range').get('tau'),
            'x_param_range': data.get('exp_params').get('param_range').get('x_mean'),
            'partial_x' : data.get('exp_params').get('partial_x'),
            'N_segment_size': data.get('exp_params').get('N_segment_size'),
            'implementation_scale': data.get('exp_params').get('implementation_scale', 1),
            'disallowed_ball_radius': data.get('exp_params').get('disallowed_ball_radius'),
            'seed': data.get('exp_params').get('seed'),
    }
        
    # overlap_scores = {              # overlap
    #             'X_overlap': np.mean(data.get('X_overlap_score')),
    #             'Y_overlap': np.mean(data.get('y_overlap_score')),
    #             'X_y_overlap': np.mean(data.get('X_y_overlap_score')),
    #             'ambiguity_score': np.mean(data.get('ambiguity_score')),
    # }

    print("=== Experiment Parameters ===")
    for key, value in dast_parameters.items():
        print(f"{key:>20}: {value}")
    # print("\n=== Overlap Scores ===")
    # for key, value in overlap_scores.items():
    #     print(f"{key:>20}: {value:.4f}")

def compute_improvement_ratio(data, comparators):
    improvement_ratios = {comp: [] for comp in comparators}

    for i in range(len(data['dast'])):
        dast_profit = data['dast'][i]['implementation_profits']
        
        for comp in comparators:
            comp_profit = data[comp][i]['implementation_profits']
            ratio = (dast_profit - comp_profit) / abs(comp_profit)
            improvement_ratios[comp].append(ratio)
    return improvement_ratios

def filter_ratios(improvement_ratios, apply_remove_extreme, apply_sigma_clip):
    # === Simplified Outlier Removal ===
    filtered_ratios = {}

    for comp, ratios in improvement_ratios.items():
        ratios_np = np.array(ratios)

        # --- Step 1: Remove 3 extreme values,  both min and max(OPTIONAL) ---
        if apply_remove_extreme.get(comp, False):
            min_indices = np.argsort(ratios_np)[:1]
            mask = np.ones_like(ratios_np, dtype=bool)
            mask[min_indices] = False
            ratios_np = ratios_np[mask]  
            
            # max_indices = np.argsort(ratios_np)[-1:]
            # mask = np.ones_like(ratios_np, dtype=bool)
            # mask[max_indices] = False
            # ratios_np = ratios_np[mask]

        # --- Step 2: Remove values outside 3 sigma (OPTIONAL) ---

        if apply_sigma_clip:
            mean = np.mean(ratios_np)
            std = np.std(ratios_np)
            ratios_np = ratios_np[np.abs(ratios_np - mean) <= 3 * std]

        filtered_ratios[comp] = ratios_np
    return filtered_ratios

def plot(filtered_ratios):
    # === Print Summary After Outlier Removal ===
    print("\n📊 Summary After Outlier Removal:\n")
    for comp, ratios in filtered_ratios.items():
        n = len(ratios)
        mean = np.mean(ratios)
        std = np.std(ratios, ddof=1)  # Sample standard deviation (ddof=1)
        
        # Calculate 95% Confidence Interval
        if n >= 30:
            # Large sample: use normal distribution (z-score = 1.96 for 95% CI)
            z_critical = 1.96
            se = std / np.sqrt(n)  # Standard error
            margin_error = z_critical * se
            ci_lower = mean - margin_error
            ci_upper = mean + margin_error
        else:
            # Small sample: use t-distribution
            t_critical = stats.t.ppf(0.975, df=n-1)  # 95% CI, two-tailed
            se = std / np.sqrt(n)  # Standard error
            margin_error = t_critical * se
            ci_lower = mean - margin_error
            ci_upper = mean + margin_error
        
        print(f"{comp:>20}: {n:3d} runs | Avg Improvement: {mean*100:7.2f}% | "
              f"95% CI: [{ci_lower*100:7.2f}%, {ci_upper*100:7.2f}%]")

    # === Plot Boxplot with Mean Markers ===
    plt.figure(figsize=(10, 6))
    box_data = [filtered_ratios[comp] for comp in comparators]
    plt.boxplot(box_data, labels=comparators, patch_artist=False)

    # Plot means as red dots
    for i, comp in enumerate(comparators):
        mean = np.mean(filtered_ratios[comp])
        # compute standard deviation of the ratios
        plt.scatter(i + 1, mean, color='red', label=f'DAST avg improvement ratio over {comp}: {mean:.2%}')

    # Unique legend
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys(), loc="upper right", fontsize='x-large')

    plt.title("Relative Implementation Profit Improvement")
    plt.ylabel("Relative Improvement Ratio")
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

# === Load Data ===
file_path = "exp_11.08/main/varying_K/result_K3.pkl"  # Make sure this is the correct path on your machine7
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
print(f"Comparators found in data: {comparators}")


apply_remove_extreme = {
                        "gmm-standard": False,  
                        "kmeans-standard": False,
                        "mst": False,
                        "clr-standard": False
                        }
apply_sigma_clip = True

print_params(data)


ratios = compute_improvement_ratio(data, comparators)
filtered_ratios = filter_ratios(ratios, apply_remove_extreme, apply_sigma_clip)


plot(filtered_ratios)

