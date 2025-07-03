from ground_truth import PopulationSimulator
from plot import plot_ground_truth, plot_segmentation
from gmm import GMM_segment_and_estimate
from oracle import structure_oracle, estimation_oracle, policy_oracle
import pandas as pd
import numpy as np
from policy_tree import policy_tree_segment_and_estimate
from oast import OAST_segment_and_estimate
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from tqdm import trange
import random

param_range = {
    "alpha": (-5, 5),
    "beta": (-5, 5),
    "tau": (-30, 30),
    "x_mean": (-10, 10)
}


N_segment = 100
d = 1
K = 3
covariate_noise = 3 # Controls how similar x_i are within a segment
noise_std = 1.0 # Standard deviation of noise in outcome generation
N = N_segment * K
DR_generation_method = "forest" # "forest" or "reg"
train_frac=0.8 # Fraction of customers to use for training (rest for validation)
N_sims = 1 # Number of simulations to run

exp_result_dict = {
    "K": K,
    "gmm": [],
    "policy_tree": [],
    "oast": []
}

for _ in trange(N_sims):
    
    seed = random.randint(0, 10000)
    np.random.seed(3733) # 470(d=1), 4300(d=2), 3733 (d=1, K=3, perfect case)
    # print(f"Random seed: {seed}")
    
    pop = PopulationSimulator(N, d, K, covariate_noise, param_range, noise_std, DR_generation_method)
    pop.train_customers, pop.val_customers, pop.train_indices, pop.val_indices = pop.split_customers_into_train_and_validate(train_frac=train_frac, seed=1024)

    algorithms = ["gmm", "policy_tree", "oast"] 
    tmp_algo_result_dict = {}
    
    try:
        for algo in algorithms:
            # print("=========================================================")
            # print(algo)
            # print("=========================================================")
            
            results_M = []
            
            
            for M in [2,3,4,5,6,7]:
                # Perform GMM segmentation and estimation
                if algo == "gmm":
                    bic = GMM_segment_and_estimate(pop, M, random_state=seed, pair_up=False)

                elif algo == "policy_tree":
                    depth_policy_tree = 1 if M <= 2 else (2 if M <=4 else (3 if M <= 8 else 4))
                    policy_tree_val_score = policy_tree_segment_and_estimate(pop, depth=depth_policy_tree, target_leaf_num=M)
            
                elif algo == "oast":
                    depth_oast = 1 if M <= 2 else (3 if M <=4 else (5 if M <= 6 else 6))
                    oast_val_score = OAST_segment_and_estimate(pop, M, depth_oast, min_leaf_size=1, epsilon=1e-6, threshold_grid=40)
                
                df = pop.to_dataframe()
                true_segment_ids_train = df['true_segment_id'].values[pop.train_indices]
                est_segment_ids_train = df[f'{algo}_est_segment_id'].values[pop.train_indices]
                
                S_metrics = structure_oracle(true_segment_ids_train, est_segment_ids_train, true_K=len(pop.true_segments))
                E_metrics = estimation_oracle(pop.train_customers, algo=algo)
                P_metrics = policy_oracle(pop.val_customers, algo=algo)
                
                # Record all
                results_M.append({
                    "M": M,
                    "oast_val": oast_val_score if algo == "oast" else None,
                    "policytree_val": policy_tree_val_score if algo == "policy_tree" else None,
                    "BIC": bic if algo == "gmm" else None,
                    "ARI": S_metrics["ARI"],
                    "NMI": S_metrics["NMI"],
                    "MSE_param": E_metrics["MSE_param"],
                    "MSE_outcome": E_metrics["MSE_outcome"],
                    "regret": P_metrics["regret"],
                    "mistreatment_rate": P_metrics["mistreatment_rate"],
                    "manager_profit": P_metrics["manager_profit"],
                })
                
                # # print manager and oracle profits
                # print(f"Results for M={M}, Manager profit: {results[-1]['manager_profit']:.2f}")
                
                # labels = df[f'{algo}_est_segment_id']
                # if algo == "gmm" and d == 1 and M == 2:
                #     train_labels = df[f'{algo}_est_segment_id'].values[pop.train_indices]
                #     df_train = df.iloc[pop.train_indices].reset_index(drop=True)
                #     plot_segmentation(train_labels, df_train, algo=algo, M=M)
                #     continue
                # elif algo == "policy_tree" and d == 1 and M == 2:
                #     plot_segmentation(labels, df, algo=algo, M=M)
                # elif algo == "oast" and d == 1 and M == 2:
                #     plot_segmentation(labels, df, algo=algo, M=M)
            
                pass_flag = True
            
            
            df_results_M = pd.DataFrame(results_M)

            # Highlight optimal M per criterion
            common_criteria = {
                'Oracle_ARI': df_results_M.at[df_results_M['ARI'].idxmax(), 'M'],
                'Oracle_NMI': df_results_M.at[df_results_M['NMI'].idxmax(), 'M'],
                'Oracle_MSE_param': df_results_M.at[df_results_M['MSE_param'].idxmin(), 'M'],
                'Oracle_MSE_outcome': df_results_M.at[df_results_M['MSE_outcome'].idxmin(), 'M'],
                'Oracle_Regret': df_results_M.at[df_results_M['regret'].idxmin(), 'M'],
                'Oracle_Mistreat': df_results_M.at[df_results_M['mistreatment_rate'].idxmin(), 'M'],
            }
            
            if algo == "gmm":
                algo_criteria = {
                    'gmm_picked_M': df_results_M.at[df_results_M['BIC'].idxmin(), 'M'],
                }
            elif algo == "policy_tree":
                algo_criteria = {
                    'policy_tree_picked_M': df_results_M.at[df_results_M['policytree_val'].idxmax(), 'M'],
                }
            elif algo == "oast":
                algo_criteria = {
                    'oast_picked_M': df_results_M.at[df_results_M['oast_val'].idxmax(), 'M'],
                }

            picked_M = {**common_criteria, **algo_criteria}
            
            # for metric, picked_m_val in highlight_criteria.items():
            #     idx = df_results_M[df_results_M['M'] == picked_m_val].index[0]
            #     print(f"  {metric}: \tM = {picked_m_val}, manager profit = {df_results_M.loc[idx, 'manager_profit']:.2f}, oracle profit = {df_results_M.loc[idx, 'oracle_profit']:.2f}")

            profit_results = df_results_M[['M', 'manager_profit']]

            tmp_algo_result_dict[algo] = {
                "picked_M": picked_M,
                "oracle_profit": P_metrics["oracle_profit"],
                "profit_results": profit_results,
                "profit_at_manager_picked_M": df_results_M.loc[df_results_M['M'] == picked_M[f'{algo}_picked_M'], 'manager_profit'].values[0],
            }
            
        
    except:
        # print the error
        import traceback
        print("An error occurred during the simulation:")
        traceback.print_exc()
        continue
    
    for algo in algorithms:
        exp_result_dict[algo].append(tmp_algo_result_dict[algo])
   

# save results
import pickle
with open("exp_result_dict.pkl", "wb") as f:
    print("Saving results to exp_result_dict.pkl")
    pickle.dump(exp_result_dict, f)