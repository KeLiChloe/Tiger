import pickle
import matplotlib.pyplot as plt
import numpy as np



def print_params(data):
        
    # === Extract Experiment Parameters for 'dast' ===
    if data.get('exp_params') is None:
        dast_parameters = {
                'K': data.get('K'),
                'd': data.get('d'),
                'covariate_noise': data.get('covariate_noise'),
                'noise_std': data.get('noise_std'),
                'tau_param_range': data.get('param_range').get('tau'),
                'x_param_range': data.get('param_range').get('x_mean'),
        }
    else:
        dast_parameters = {
            'K': data.get('exp_params').get('K'),
            'd': data.get('exp_params').get('d'),
            'covariate_noise': data.get('exp_params').get('covariate_noise'),
            'noise_std': data.get('exp_params').get('noise_std'),
            'tau_param_range': data.get('exp_params').get('param_range').get('tau'),
            'x_param_range': data.get('exp_params').get('param_range').get('x_mean'),
    }

    print("\n⚙️ Experiment Parameters for 'dast':\n")
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

def filter_ratios(improvement_ratios, apply_remove_extreme, apply_sigma_clip):
    # === Simplified Outlier Removal ===
    filtered_ratios = {}

    for comp, ratios in improvement_ratios.items():
        ratios_np = np.array(ratios)

        # --- Step 1: Remove 3 extreme values,  both min and max(OPTIONAL) ---
        if apply_remove_extreme[comp]:
            min_indices = np.argsort(ratios_np)[:1]
            mask = np.ones_like(ratios_np, dtype=bool)
            mask[min_indices] = False
            ratios_np = ratios_np[mask]  
            max_indices = np.argsort(ratios_np)[-1:]
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

def plot(filtered_ratios):
    # === Print Summary After Outlier Removal ===
    print("\n📊 Summary After Outlier Removal:\n")
    for comp, ratios in filtered_ratios.items():
        print(f"{comp:>12}: {len(ratios)} runs | Avg Improvement: {np.mean(ratios):.4%}")

    # === Plot Boxplot with Mean Markers ===
    plt.figure(figsize=(10, 6))
    box_data = [filtered_ratios[comp] for comp in comparators]
    plt.boxplot(box_data, labels=comparators, patch_artist=False)

    # Plot means as red dots
    for i, comp in enumerate(comparators):
        mean = np.mean(filtered_ratios[comp])
        plt.scatter(i + 1, mean, color='red', marker='o', label=f'Improvemnet over {comp}: {mean:.2%}')

    # Unique legend
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys(), loc="upper right", fontsize='x-large')

    plt.title("Relative Implementation Profit Improvement")
    plt.ylabel("Relative Improvement Ratio")
    plt.ylim(-0.5, 1)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

# === Load Data ===
file_path = "exp/ablation/correct_exp_ablation_gmm_4.pkl"  # Make sure this is the correct path on your machine

with open(file_path, "rb") as f:
    data = pickle.load(f)

save_ratio = False

# comparators = [ "gmm", "mst", "kmeans", "policy_tree"]
comparators = ["gmm-standard", "gmm-da"]

apply_remove_extreme = {"gmm-standard": False, 
                        "gmm-da": False,
                        # "mst": False,
                        # "policy_tree": False
                        }
apply_sigma_clip = False

print_params(data)

if 'filtered_ratios' in data:
    filtered_ratios = data['filtered_ratios']
else:
    ratios = compute_improvement_ratio(data, comparators)
    filtered_ratios = filter_ratios(ratios, apply_remove_extreme, apply_sigma_clip)
    
    if save_ratio:
        data['filtered_ratios'] = filtered_ratios
        save_file = f"{file_path.split('.')[0]}_ratios.pkl"
        with open(save_file, "wb") as f:
            print(f"Saving results to {save_file}")
            pickle.dump(data, f)


plot(filtered_ratios)

