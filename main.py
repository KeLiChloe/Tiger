from ground_truth import PopulationSimulator
from plot import plot_ground_truth, plot_segmentation
from gmm import GMM_segment_and_estimate
from oracle import structure_oracle, estimation_oracle, policy_oracle
import pandas as pd
import numpy as np
from policy_tree import policy_tree_segment_and_estimate, assign_new_customers_to_pruned_tree
from dast import DAST_segment_and_estimate
from mst import MST_segment_and_estimate
from kmeans import KMeans_segment_and_estimate
from clr import CLR_segment_and_estimate
from utils import assign_new_customers_to_segments, pick_M_for_algo
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from tqdm import trange
import random
import multiprocessing
import pickle
multiprocessing.set_start_method('fork') 

param_range = {
    "alpha": (-10, 10),
    "beta": (-10, 10),
    "tau": (-50, 50),
    "x_mean": (-20, 20)
}


plot = False
compute_overlap = True

N_segment_size = 100 # number of customers per segment
d = 1 # dimensionality of covariates
K = 5 # number of segments
M_range = list(range(max(2, K-3), K+4))
covariate_noise = 4 # controls how similar x_i are within a segment
noise_std = 4 # std of noise in outcome generation

kmeans_coef = 0.3
clr_pair = False

N_total_pilot_customers = N_segment_size * K
N_total_implement_customers = N_total_pilot_customers

DR_generation_method = "mlp" # "forest", "reg", or "mlp". For DAST only. Policytree uses "forest" by default. I suggest use "MLP".

N_sims = 1 # Number of simulations to run


# algorithms =   ["gmm-standard", "gmm-da", "kmeans-standard", "kmeans-da", "clr-standard", "clr-da", "dast", "mst", "policy_tree"]
algorithms =   []

exp_result_dict = {
    "exp_params": {
        "K": K,
        "d": d,
        "covariate_noise": covariate_noise,
        "noise_std": noise_std,
        "param_range": param_range,
        "N_segment_size": N_segment_size,
        "DR_generation_method": DR_generation_method,
        "clr_pair": clr_pair,
        "kmeans_coef": kmeans_coef,
        "N_total_pilot_customers": N_total_pilot_customers,
        "N_total_implement_customers": N_total_implement_customers,
    },
    
    **{algo: [] for algo in algorithms},
    "X_overlap_score": [],
    "X_y_overlap_score": [],
    "ambiguity_score": [],
}


for _ in trange(N_sims):
    
    seed = random.randint(0, 100000)
    np.random.seed(seed)   # 5909, 67691
    print(f"Random seed: {seed}")
    
    pop = PopulationSimulator(N_total_pilot_customers, N_total_implement_customers, d, K, covariate_noise, param_range, noise_std, DR_generation_method)
    if compute_overlap:
        X_overlap_score = pop.compute_covariate_overlap()
        X_y_overlap_score = pop.compute_joint_overlap()
        ambiguity_score = pop.compute_assignment_ambiguity()
        print(f"X overlap score: {X_overlap_score:.4f}, (X, Y) overlap score: {X_y_overlap_score:.4f}, ambiguity score: {ambiguity_score:.4f}")
    
    # plot ground truth
    if plot:
        df = pop.to_dataframe()
        plot_ground_truth(df)
    
    algo_result_dict = {}
    
    try:
        for algo in algorithms:
            
            # split pilot customers into train and validation set
            if algo in ["clr-standard", "kmeans-standard", "gmm-standard"]:
                train_frac=1
            else:
                train_frac=0.75
            
            pop.split_pilot_customers_into_train_and_validate(train_frac=train_frac)
            x_mat = np.array([cust.x for cust in pop.train_customers])
            D_vec = np.array([cust.D_i for cust in pop.train_customers]).reshape(-1, 1)
            y_vec = np.array([cust.y for cust in pop.train_customers]).reshape(-1, 1)
            
            results_M = []
            
            for M in M_range:
                # print(f"Running {algo} with M={M}...")
                
                depth_policy_tree = 1 if M <= 2 else (2 if M <=4 else (3 if M <= 8 else 4))
                depth_dast = 1 if M <= 2 else (2 if M <=4 else (3 if M <= 6 else 4))
                depth_mst = 1 if M <= 2 else (2 if M <=4 else (3 if M <= 6 else 4))
                
                # Perform segmentation and estimation
                # What happened in each function
                # 1. Segment customers into M segments
                # 2. For each segment, estimate parameters
                # 3. Assign each train customer to estimated segment
                # 4. Return labels and validation score
                
                if algo == "gmm-standard":
                    bic_gmm, _ = GMM_segment_and_estimate(pop, M, x_mat, D_vec, y_vec, algo, random_state=seed)
                
                elif algo == "gmm-da":
                    DA_score_gmm, _ = GMM_segment_and_estimate(pop, M, x_mat, D_vec, y_vec, algo, random_state=seed)
                    
                elif algo == "kmeans-standard":
                    silhouette_score, _ = KMeans_segment_and_estimate(pop, M, x_mat, D_vec, y_vec, algo, random_state=seed)
                
                elif algo == "kmeans-da":
                    DA_score_kmeans, _ = KMeans_segment_and_estimate(pop, M, x_mat, D_vec, y_vec, algo, random_state=seed)
                
                elif algo == "clr-standard":
                    bic_clr, _ = CLR_segment_and_estimate(pop, M, x_mat, D_vec, y_vec, kmeans_coef=kmeans_coef, num_tries=8, algo=algo, random_state=seed)
                
                elif algo == "clr-da":
                    DA_score_clr, _ = CLR_segment_and_estimate(pop, M, x_mat, D_vec, y_vec, kmeans_coef=kmeans_coef, num_tries=8, algo=algo, random_state=seed)
                    
                elif algo == "dast":
                    dast_tree, dast_val_score, segment_dict = DAST_segment_and_estimate(pop, M, max_depth=depth_dast, min_leaf_size=2, epsilon=1e-2, threshold_grid=40)
                
                elif algo == "mst":
                    mst_tree, mst_val_score, segment_dict = MST_segment_and_estimate(pop, M, max_depth=depth_mst, min_leaf_size=2, epsilon=1e-2, threshold_grid=40)
                    
                elif algo == "policy_tree":
                    policy_tree_val_score, _, _ = policy_tree_segment_and_estimate(pop, depth_policy_tree, M, x_mat, D_vec, y_vec)
                
                else:
                    raise ValueError(f"Unknown algorithm: {algo}")
                
                # plot segmentation 
                df = pop.to_dataframe()
                labels = df[f'{algo}_est_segment_id'][:int(N_total_pilot_customers * train_frac)]
                if plot:
                    plot_segmentation(labels, x_mat, y_vec, D_vec, f'{algo}', M=M)
                
                # Oracle evaluation
                true_segment_ids_train = df['true_segment_id'].values[pop.train_indices]
                est_segment_ids_train = df[f'{algo}_est_segment_id'].values[pop.train_indices]
                S_metrics = structure_oracle(true_segment_ids_train, est_segment_ids_train)
                E_metrics = estimation_oracle(pop.pilot_customers, algo=algo)
                P_metrics = policy_oracle(pop.pilot_customers, algo=algo) #TODO: apply to implement customers
                
                # Record all
                results_M.append({
                    "M": M,
                    "dast_val": dast_val_score if algo == "dast" else None,
                    "policytree_val": policy_tree_val_score if algo == "policy_tree" else None,
                    "mst_val": mst_val_score if algo == "mst" else None,
                    "kmeans-standard_val": silhouette_score if algo == "kmeans-standard" else None,
                    "kmeans-da_val": DA_score_kmeans if algo == "kmeans-da" else None,
                    "gmm-standard_val": bic_gmm if algo == "gmm-standard" else None,
                    "gmm-da_val": DA_score_gmm if algo == "gmm-da" else None,
                    "clr-standard_val": bic_clr if algo == "clr-standard" else None,
                    "clr-da_val": DA_score_clr if algo == "clr-da" else None,
                    "ARI": S_metrics["ARI"],
                    "NMI": S_metrics["NMI"],
                    "MSE_param": E_metrics["MSE_param"],
                    "MSE_outcome": E_metrics["MSE_outcome"],
                    "regret": P_metrics["regret"],
                    "mistreatment_rate": P_metrics["mistreatment_rate"],
                    "manager_profit": P_metrics["manager_profit"],
                })
            
            
            df_results_M = pd.DataFrame(results_M)

            # Highlight optimal M per criterion
            oracle_picked_M = {
                'Oracle_ARI': df_results_M.at[df_results_M['ARI'].idxmax(), 'M'],
                'Oracle_NMI': df_results_M.at[df_results_M['NMI'].idxmax(), 'M'],
                'Oracle_MSE_param': df_results_M.at[df_results_M['MSE_param'].idxmin(), 'M'],
                'Oracle_MSE_outcome': df_results_M.at[df_results_M['MSE_outcome'].idxmin(), 'M'],
                'Oracle_Regret': df_results_M.at[df_results_M['regret'].idxmin(), 'M'],
                'Oracle_Mistreat': df_results_M.at[df_results_M['mistreatment_rate'].idxmin(), 'M'],
            }

            algo_picked_M = pick_M_for_algo(algo, df_results_M)

            picked_M = {**oracle_picked_M, **algo_picked_M}
            
            # for metric, picked_m_val in picked_M.items():
            #     idx = df_results_M[df_results_M['M'] == picked_m_val].index[0]
            #     print(f"  {metric}: \tM = {picked_m_val}, manager profit = {df_results_M.loc[idx, 'manager_profit']:.2f}")

            profit_results = df_results_M[['M', 'manager_profit']]

            algo_result_dict[algo] = {
                "picked_M": picked_M,
                "oracle_profit_validation": P_metrics["oracle_profit"],
                "profit_results": profit_results,
                "profit_at_manager_picked_M": df_results_M.loc[df_results_M['M'] == picked_M[f'{algo}_picked_M'], 'manager_profit'].values[0],
            }
            
            # Retrain and assign implementation customers to segments  
            
            pop.split_pilot_customers_into_train_and_validate(train_frac=1)
            x_mat = np.array([cust.x for cust in pop.train_customers])
            D_vec = np.array([cust.D_i for cust in pop.train_customers]).reshape(-1, 1)
            y_vec = np.array([cust.y for cust in pop.train_customers]).reshape(-1, 1)
            
            algo_picked_M = picked_M[f'{algo}_picked_M']
            if algo == "gmm-standard":
                bic_gmm, gmm_model = GMM_segment_and_estimate(pop, M, x_mat, D_vec, y_vec, algo, random_state=seed)  
                assign_new_customers_to_segments(pop, pop.implement_customers, gmm_model, algo)
            
            elif algo == "gmm-da":
                DA_score_gmm, gmm_model = GMM_segment_and_estimate(pop, M, x_mat, D_vec, y_vec, algo, random_state=seed)  
                assign_new_customers_to_segments(pop, pop.implement_customers, gmm_model, algo)
            
            elif algo == "policy_tree":
                depth_policy_tree = 1 if algo_picked_M <= 2 else (2 if algo_picked_M <=4 else (3 if algo_picked_M <= 8 else 4))
                _, optimal_policy_tree, leaf_to_pruned_segment = policy_tree_segment_and_estimate(pop, depth_policy_tree, algo_picked_M, x_mat, D_vec, y_vec)
                assign_new_customers_to_pruned_tree(optimal_policy_tree, pop, pop.implement_customers, leaf_to_pruned_segment)
                
            elif algo == "dast":
                depth_dast = 1 if algo_picked_M <= 2 else (2 if algo_picked_M <=4 else (3 if algo_picked_M <= 8 else 4))
                optimal_dast_tree, _, segment_dict = DAST_segment_and_estimate(pop, algo_picked_M, depth_dast, min_leaf_size=2, epsilon=1e-3, threshold_grid=50)
                optimal_dast_tree.predict_segment(pop.implement_customers, segment_dict) # Assign implementation customers to segments 
            
            elif algo == "mst":
                depth_mst = 1 if algo_picked_M <= 2 else (2 if algo_picked_M <=4 else (3 if algo_picked_M <= 8 else 4))
                optimal_mst_tree, mst_val_score, segment_dict = MST_segment_and_estimate(pop, algo_picked_M, max_depth=depth_mst, min_leaf_size=2, epsilon=1e-3, threshold_grid=30)
                optimal_mst_tree.predict_segment(pop.implement_customers, segment_dict)
            
            elif algo == "kmeans-standard":
                silhouette_score, kmeans_model = KMeans_segment_and_estimate(pop, algo_picked_M, x_mat, D_vec, y_vec, algo, random_state=seed)
                assign_new_customers_to_segments(pop, pop.implement_customers, kmeans_model, algo)
            
            elif algo == "kmeans-da":
                DA_score_kmeans, kmeans_model = KMeans_segment_and_estimate(pop, algo_picked_M, x_mat, D_vec, y_vec, algo, random_state=seed)
                assign_new_customers_to_segments(pop, pop.implement_customers, kmeans_model, algo)
            
            elif algo == "clr-standard":
                bic_clr, CLR = CLR_segment_and_estimate(pop, algo_picked_M, x_mat, D_vec, y_vec, kmeans_coef, num_tries=8, algo=algo, random_state=seed)
                assign_new_customers_to_segments(pop, pop.implement_customers, CLR, algo)
            elif algo == "clr-da":
                DA_score_clr, CLR = CLR_segment_and_estimate(pop, algo_picked_M, x_mat, D_vec, y_vec, kmeans_coef, num_tries=8, algo=algo, random_state=seed)
                assign_new_customers_to_segments(pop, pop.implement_customers, CLR, algo)   
            
            # Evaluate implementation outcome
            implementation_outcome = 0
            for cust in pop.implement_customers:
                implementation_outcome += cust.evaluate_profits(algo)
            print(f"Implementation outcome for {algo}: {implementation_outcome:.2f} with chosen M = {algo_picked_M}")
            algo_result_dict[algo]['implementation_profits'] = implementation_outcome
            
            oracle_profits_implementation = policy_oracle(pop.implement_customers, algo)
            
            
    except:
        import traceback
        traceback.print_exc()
        continue
    
    for algo in algorithms:
        exp_result_dict[algo].append(algo_result_dict[algo])
    
    if compute_overlap:
        exp_result_dict['X_overlap_score'].append(X_overlap_score)
        exp_result_dict['X_y_overlap_score'].append(X_y_overlap_score)
        exp_result_dict['ambiguity_score'].append(ambiguity_score)
    
  
    # print(f"Oracle profits: {oracle_profits_implementation['oracle_profit']:.2f}")

    # save the result after each simulation and print simulation number
    # save_file = "exp/ablation/correct_exp_ablation_gmm_4.pkl"
    # print(f"Completed {len(exp_result_dict['dast'])} / {N_sims} simulations.")
    # with open(save_file, "wb") as f:
    #     print(f"Saving results to {save_file}")
    #     pickle.dump(exp_result_dict, f)


# print mean value of overlap scores
if compute_overlap:
    print(f"Average X overlap score: {np.mean(exp_result_dict['X_overlap_score']):.4f} ± {np.std(exp_result_dict['X_overlap_score']):.4f}")
    print(f"Average (X, Y) overlap score: {np.mean(exp_result_dict['X_y_overlap_score']):.4f} ± {np.std(exp_result_dict['X_y_overlap_score']):.4f}")
    print(f"Average ambiguity score: {np.mean(exp_result_dict['ambiguity_score']):.4f} ± {np.std(exp_result_dict['ambiguity_score']):.4f}")