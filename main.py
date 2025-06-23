from ground_truth import PopulationSimulator
from plot import plot_ground_truth, plot_segmentation
from gmm import GMM_segment_and_estimate
from oracle import structure_oracle, estimation_oracle, policy_oracle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from policy_tree import policy_tree_segment_and_estimate

param_range = {
    "alpha": (-10, 10),
    "beta": (-10, 10),
    "tau": (-50, 50),
    "x_mean": (-10, 10)
}

N_segment = 100
d = 1
K = 3
covariate_noise = 2 # Controls how similar x_i are within a segment
noise_std = 1.0 # Standard deviation of noise in outcome generation
N = N_segment * K

results = []

seed = np.random.randint(0, 1000)
np.random.seed(381) # 381
print(f"Random seed: {seed}")

pop = PopulationSimulator(N, d, K, covariate_noise, param_range, noise_std)

algo = "policy_tree"  # ["gmm", "policy_tree"]

for M in [1,2,3,4,5,6,7]:
    if algo == "gmm":
        # Perform GMM segmentation and estimation
        gmm, bic = GMM_segment_and_estimate(pop, M, random_state=0)
    elif algo == "policy_tree":
        depth = 1 if M <= 2 else (2 if M <=4 else (3 if M <= 8 else 4)) 
        policy_tree_segment_and_estimate(pop, depth=depth, target_leaf_num=M)
    df = pop.to_dataframe()
    
    S_metrics = structure_oracle(df['true_segment_id'], df[f'{algo}_est_segment_id'], true_K=len(pop.true_segments))
    E_metrics = estimation_oracle(pop.customers, algo=algo)
    P_metrics = policy_oracle(pop.customers, algo=algo)
    
    # Record all
    results.append({
        "M": M,
        "BIC": bic if algo == "gmm" else None,
        "ARI": S_metrics["ARI"],
        "NMI": S_metrics["NMI"],
        "MSE_param": E_metrics["MSE_param"],
        "MSE_outcome": E_metrics["MSE_outcome"],
        "regret": P_metrics["regret"],
        "mistreatment_rate": P_metrics["mistreatment_rate"],
        "manager_profit": P_metrics["manager_profit"],
        "oracle_profit": P_metrics["oracle_profit"],
    })
    
    # print manager and oracle profits
    print(f"Results for M={M}, Manager profit: {results[-1]['manager_profit']:.2f}")
    
    # Plot segmentation
    if algo == "gmm":
        plot_segmentation(pop, algo=algo,gmm=gmm)
    elif algo == "policy_tree":
        plot_segmentation(pop, algo=algo)
    
    
df_results = pd.DataFrame(results)

# Highlight optimal M per criterion
highlight_criteria = {
    '(Information Cretiron) BIC': df_results['BIC'].idxmin(),
    '(Structure Oracle) ARI': df_results['ARI'].idxmax(),
    '(Structure Oracle) NMI': df_results['NMI'].idxmax(),
    '(Estima Oracle) MSE_param': df_results['MSE_param'].idxmin(),
    '(Estima Oracle) MSE_outcome': df_results['MSE_outcome'].idxmin(),
    '(Policy Oracle) Regret': df_results['regret'].idxmin(),
    '(Policy Oracle) Mistreat': df_results['mistreatment_rate'].idxmin()
}

print("Optimal M under each criterion:")
for metric, idx in highlight_criteria.items():
    # skip BIC for policy_tree
    if algo == "policy_tree" and metric == '(Information Cretiron) BIC':
        continue
    m_val = df_results.loc[idx, 'M']
    print(f"  {metric}: \tM = {m_val}, manager profit = {df_results.loc[idx, 'manager_profit']:.2f}, oracle profit = {df_results.loc[idx, 'oracle_profit']:.2f}")


plot_ground_truth(df)

    