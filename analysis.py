import pickle
import matplotlib.pyplot as plt
import numpy as np



def print_params(data):
        
    # === Extract Experiment Parameters for 'dast' ===
    if data.get('exp_params') is None:
        dast_parameters = {
                'K': data.get('K'),
                'd': data.get('d'),
                'signal_covariate_noise': data.get('signal_covariate_noise'),
                'disturb_covariate_noise': data.get('disturb_covariate_noise'),
                'noise_std': data.get('noise_std'),
                'tau_param_range': data.get('param_range').get('tau'),
                'x_param_range': data.get('param_range').get('x_mean'),

                
        }
    else:
        dast_parameters = {
            'K': data.get('exp_params').get('K'),
            'd': data.get('exp_params').get('d'),
            'signal_covariate_noise': data.get('exp_params').get('signal_covariate_noise'),
            'disturb_covariate_noise': data.get('exp_params').get('disturb_covariate_noise'),
            'noise_std': data.get('exp_params').get('noise_std'),
            'tau_param_range': data.get('exp_params').get('param_range').get('tau'),
            'x_param_range': data.get('exp_params').get('param_range').get('x_mean'),
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
            min_indices = np.argsort(ratios_np)[:0]
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
        print(f"{comp:>12}: {len(ratios)} runs | Avg Improvement: {np.mean(ratios):.4%}")

    # === Plot Boxplot with Mean Markers ===
    plt.figure(figsize=(10, 6))
    box_data = [filtered_ratios[comp] for comp in comparators]
    plt.boxplot(box_data, labels=comparators, patch_artist=False)

    # Plot means as red dots
    for i, comp in enumerate(comparators):
        mean = np.mean(filtered_ratios[comp])
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
file_path = "exp/main/4.pkl"  # Make sure this is the correct path on your machine

with open(file_path, "rb") as f:
    data = pickle.load(f)

save_ratio = False

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
                        "gmm-standard": True,  
                        "kmeans-standard": True,
                        "mst": True,
                        "clr-standard": False
                        }
apply_sigma_clip = False

print_params(data)

if 'filtered_ratios' in data:
    filtered_ratios = data['filtered_ratios']
    # print sorted filtered ratios
    for comp in comparators:
        print(f"{comp:>12}: ", end="")
        sorted_ratios = np.sort(filtered_ratios[comp])
        print(sorted_ratios)

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

