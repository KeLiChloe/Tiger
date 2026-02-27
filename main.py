from ground_truth import PopulationSimulator
from plot import plot_ground_truth, plot_segmentation
from gmm import GMM_segment_and_estimate
from oracle import structure_oracle, policy_oracle
import pandas as pd
import numpy as np
from policy_tree import policy_tree_segment_and_estimate, assign_new_customers_to_pruned_tree
from dast import DAST_segment_and_estimate
from mst import MST_segment_and_estimate
from kmeans import KMeans_segment_and_estimate
from clr import CLR_segment_and_estimate
from meta_learners import T_learner, S_learner, X_learner, DR_learner
from causal_forest import causal_forest_predict
from utils import assign_new_customers_to_segments, pick_M_for_algo, parse_args
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from tqdm import trange
import random
import multiprocessing
import pickle
multiprocessing.set_start_method('fork') 
import time
import json

def main(args, param_range):
    '''
    In this main function, we fix the experiment parameters, such as K, d, N, signal strength, noise level, etc.
    for each simulation, we generate a random new population,
    run all algorithms to segment and estimate,
    pick M for each algorithm based on validation set,
    assign implementation customers to segments,
    and evaluate implementation profits.
    '''
    
    outcome_type = args.outcome_type
    # interaction terms only make sense for continuous outcome
    include_interactions = (outcome_type == 'continuous') and hasattr(args, 'delta_range') and args.delta_range is not None
    print(f"Outcome type: {outcome_type}, Include interactions: {include_interactions}")
    
    N_total_pilot_customers = args.N_segment_size * args.K
    N_total_implement_customers = int(N_total_pilot_customers * args.implementation_scale)
    M_range = list(range(max(2, args.K-3), args.K+4))


    exp_result_dict = {
    "exp_params": {
        "sequence_seed": args.sequence_seed,
        "action_num": getattr(args, 'action_num', 2),
        "K": args.K,
        "d": args.d,
        "partial_x": args.partial_x,
        "X_noise_std_scale": args.X_noise_std_scale,
        "disturb_covariate_noise": args.disturb_covariate_noise,
        "Y_noise_std_scale": args.Y_noise_std_scale,
        "disallowed_ball_radius": getattr(args, 'disallowed_ball_radius', None),
        "param_range": param_range,
        "N_segment_size": args.N_segment_size,
        "DR_generation_method": args.DR_generation_method,
        "kmeans_coef": args.kmeans_coef,
        "N_total_pilot_customers": N_total_pilot_customers,
        "implementation_scale": args.implementation_scale,
    },
    
    **{algo: [] for algo in args.algorithms},
        "seed": [],
    }


    start_time = time.time()
    
    # Fix the random seed generator to get reproducible seed sequences
    if args.sequence_seed is not None:
        random.seed(args.sequence_seed)
        print(f"Using fixed sequence seed: {args.sequence_seed}")

    for _ in trange(args.N_sims):
        
        seed = random.randint(0, 100000)
        np.random.seed(seed)   # 5909, 67691
        print(f"Random seed: {seed}")
        
        
        pop = PopulationSimulator(N_total_pilot_customers, 
                                  N_total_implement_customers, 
                                  args.d, 
                                  args.K, 
                                  args.disturb_covariate_noise, 
                                  param_range, 
                                  args.DR_generation_method, 
                                  args.partial_x,
                                  action_num=getattr(args, 'action_num', 2),
                                  X_noise_std_scale=args.X_noise_std_scale,
                                  Y_noise_std_scale=getattr(args, 'Y_noise_std_scale', None),
                                  disallowed_ball_radius=getattr(args, 'disallowed_ball_radius', None),
                                  outcome_type=outcome_type)
        # plot ground truth
        if args.plot:
            df = pop.to_dataframe()
            plot_ground_truth(df)
        
        algo_result_dict = {}
        
        try:
            for algo in args.algorithms:
                
                # split pilot customers into train and validation set
                if algo in ["clr-standard", "kmeans-standard", "gmm-standard"]:
                    train_frac=1
                else:
                    train_frac=0.75
                
                pop.split_pilot_customers_into_train_and_validate(train_frac=train_frac)
                x_mat_tr = np.array([cust.x for cust in pop.train_customers])
                D_vec_tr = np.array([cust.D_i for cust in pop.train_customers]).reshape(-1, 1)
                y_vec_tr = np.array([cust.y for cust in pop.train_customers]).reshape(-1, 1)
                
                x_mat_val = np.array([cust.x for cust in pop.val_customers])
                D_vec_val = np.array([cust.D_i for cust in pop.val_customers]).reshape(-1, 1)
                y_vec_val = np.array([cust.y for cust in pop.val_customers]).reshape(-1, 1)
                
                results_M = []
                
                
                for M in M_range:
                    
                    # print(f"Running {algo} with M={M}...")
                    
                    depth_policy_tree = 1 if M <= 2 else (2 if M <=4 else (3 if M <= 8 else 4))
                    depth_dast = 1 if M <= 2 else (2 if M <=4 else (3 if M <= 6 else 4))
                    depth_mst = 1 if M <= 2 else (2 if M <=4 else (3 if M <= 6 else 4))
                    
                    # Initialize algorithm-specific variables
                    dast_val_score = None
                    mst_val_score = None
                    policy_tree_val_score = None
                    silhouette_score, bic_gmm, bic_clr = None, None, None
                    DA_score_kmeans, DA_score_gmm, DA_score_clr = None, None, None
                    
                    # Perform segmentation and estimation
                    # What happened in each function
                    # 1. Segment customers into M segments
                    # 2. For each segment, estimate parameters
                    # 3. Assign each train customer to estimated segment
                    # 4. Return labels and validation score
                    
                    if algo == "gmm-standard":
                        bic_gmm, _ = GMM_segment_and_estimate(pop, M, x_mat_tr, D_vec_tr, y_vec_tr, algo, include_interactions, random_state=seed)
                    
                    elif algo == "gmm-da":
                        DA_score_gmm, _ = GMM_segment_and_estimate(pop, M, x_mat_tr, D_vec_tr, y_vec_tr, algo, include_interactions, random_state=seed)
                        
                    elif algo == "kmeans-standard":
                        silhouette_score, _ = KMeans_segment_and_estimate(pop, M, x_mat_tr, D_vec_tr, y_vec_tr, algo, include_interactions, random_state=seed)
                    
                    elif algo == "kmeans-da":
                        DA_score_kmeans, _ = KMeans_segment_and_estimate(pop, M, x_mat_tr, D_vec_tr, y_vec_tr, algo, include_interactions, random_state=seed)
                    
                    elif algo == "clr-standard":
                        bic_clr, _ = CLR_segment_and_estimate(pop, M, x_mat_tr, D_vec_tr, y_vec_tr, kmeans_coef=args.kmeans_coef, num_tries=8, algo=algo, include_interactions=include_interactions, random_state=seed)
                    
                    elif algo == "clr-da":
                        DA_score_clr, _ = CLR_segment_and_estimate(pop, M, x_mat_tr, D_vec_tr, y_vec_tr, kmeans_coef=args.kmeans_coef, num_tries=8, algo=algo, include_interactions=include_interactions, random_state=seed)
                        
                    elif algo == "dast":
                        dast_tree, dast_val_score, segment_dict = DAST_segment_and_estimate(pop, M, max_depth=depth_dast, min_leaf_size=2,  algo=algo, include_interactions=include_interactions, use_hybrid_method=args.use_hybrid_method)
                    
                    elif algo == "mst":
                        mst_tree, mst_val_score, segment_dict = MST_segment_and_estimate(pop, M, max_depth=depth_mst, min_leaf_size=2, epsilon=1e-2, threshold_grid=30, algo=algo, include_interactions=include_interactions)
                        
                    elif algo == "policy_tree":
                        policy_tree_val_score, _, _ = policy_tree_segment_and_estimate(pop, depth_policy_tree, M, x_mat_tr, D_vec_tr, y_vec_tr, x_mat_val, D_vec_val, y_vec_val,include_interactions=include_interactions, use_hybrid_method=False, )
                    
                    elif algo in ["t_learner", "x_learner", "dr_learner", "s_learner", "causal_forest"]:
                        continue
                    else:
                        raise ValueError(f"Unknown algorithm: {algo}")
                    
                    # plot segmentation 
                    df = pop.to_dataframe()
                    labels = df[f'{algo}_est_segment_id'][:int(N_total_pilot_customers * train_frac)]
                    if args.plot:
                        # Pass tree object for DAST/MST to plot decision boundaries
                        tree_obj = None
                        if algo == "dast":
                            tree_obj = dast_tree
                        elif algo == "mst":
                            tree_obj = mst_tree
                        plot_segmentation(labels, x_mat_tr, y_vec_tr, D_vec_tr, f'{algo}', M=M, tree=tree_obj)
                    
                    # Oracle evaluation
                    true_segment_ids_train = df['true_segment_id'].values[pop.train_indices]
                    est_segment_ids_train = df[f'{algo}_est_segment_id'].values[pop.train_indices]
                    S_metrics = structure_oracle(true_segment_ids_train, est_segment_ids_train)
                    P_metrics = policy_oracle(pop.pilot_customers, algo=algo, signal_d=pop.signal_d)
                    
                    # Record all
                    results_M.append({
                        "M": M if algo not in ["t_learner", "x_learner", "dr_learner", "s_learner", "causal_forest"] else 0,
                        "dast_val": dast_val_score if algo == "dast" else None,
                        "policy_tree_val": policy_tree_val_score if algo == "policy_tree" else None,
                        "mst_val": mst_val_score if algo == "mst" else None,
                        "kmeans-standard_val": silhouette_score if algo == "kmeans-standard" else None,
                        "kmeans-da_val": DA_score_kmeans if algo == "kmeans-da" else None,
                        "gmm-standard_val": bic_gmm if algo == "gmm-standard" else None,
                        "gmm-da_val": DA_score_gmm if algo == "gmm-da" else None,
                        "clr-standard_val": bic_clr if algo == "clr-standard" else None,
                        "clr-da_val": DA_score_clr if algo == "clr-da" else None,
                        
                        "ARI": S_metrics["ARI"] if algo not in ["t_learner", "x_learner", "dr_learner", "s_learner", "causal_forest"] else None,
                        "NMI": S_metrics["NMI"] if algo not in ["t_learner", "x_learner", "dr_learner", "s_learner", "causal_forest"] else None,
                        "MSE_tau": E_metrics["MSE_tau"] if algo not in ["t_learner", "x_learner", "dr_learner", "s_learner", "causal_forest"] else None,
                        "regret": P_metrics["regret"] if algo not in ["t_learner", "x_learner", "dr_learner", "s_learner", "causal_forest"] else None,
                        "mistreatment_rate": P_metrics["mistreatment_rate"] if algo not in ["t_learner", "x_learner", "dr_learner", "s_learner", "causal_forest"] else None,
                        "manager_profit": P_metrics["manager_profit"] if algo not in ["t_learner", "x_learner", "dr_learner", "s_learner", "causal_forest"] else None,
                    })
                
                
                df_results_M = pd.DataFrame(results_M)

                # Highlight optimal M per criterion
                # the following metrics are problematic for meta-learners because none values. Should fix later. 
                
                oracle_picked_M = {
                    'Oracle_ARI': df_results_M.at[df_results_M['ARI'].idxmax(), 'M'] if algo not in ["t_learner", "x_learner", "dr_learner", "s_learner", "causal_forest"] else 0,
                    'Oracle_NMI': df_results_M.at[df_results_M['NMI'].idxmax(), 'M'] if algo not in ["t_learner", "x_learner", "dr_learner", "s_learner", "causal_forest"] else 0,
                    'Oracle_MSE_tau': df_results_M.at[df_results_M['MSE_tau'].idxmin(), 'M'] if algo not in ["t_learner", "x_learner", "dr_learner", "s_learner", "causal_forest"] else 0,
                    'Oracle_Regret': df_results_M.at[df_results_M['regret'].idxmin(), 'M'] if algo not in ["t_learner", "x_learner", "dr_learner", "s_learner", "causal_forest"] else 0,
                    'Oracle_Mistreat': df_results_M.at[df_results_M['mistreatment_rate'].idxmin(), 'M'] if algo not in ["t_learner", "x_learner", "dr_learner", "s_learner", "causal_forest"] else 0,
                }

                algo_picked_M = pick_M_for_algo(algo, df_results_M)

                picked_M = {**oracle_picked_M, **algo_picked_M}
                
                # for metric, picked_m_val in picked_M.items():
                #     idx = df_results_M[df_results_M['M'] == picked_m_val].index[0]
                #     print(f"  {metric}: \tM = {picked_m_val}, manager profit = {df_results_M.loc[idx, 'manager_profit']:.2f}")


                is_meta_learner = algo in ["t_learner", "x_learner", "dr_learner", "s_learner", "causal_forest"]
                if is_meta_learner:
                    row_at_picked_M = None
                else:
                    picked_m_val = picked_M[f'{algo}_picked_M']
                    row_at_picked_M = df_results_M.loc[df_results_M['M'] == picked_m_val].iloc[0]

                algo_result_dict[algo] = {
                    "picked_M": picked_M if not is_meta_learner else "Not applicable",
                    "profit_at_manager_picked_M": row_at_picked_M['manager_profit'] if not is_meta_learner else None,
                    "ARI": row_at_picked_M['ARI'] if not is_meta_learner else None,
                    "NMI": row_at_picked_M['NMI'] if not is_meta_learner else None,
                    "MSE_tau": row_at_picked_M['MSE_tau'] if not is_meta_learner else None,
                    "regret": row_at_picked_M['regret'] if not is_meta_learner else None,
                    "mistreatment_rate": row_at_picked_M['mistreatment_rate'] if not is_meta_learner else None,
                }
                
                # Retrain and assign implementation customers to segments  
            
                pop.split_pilot_customers_into_train_and_validate(train_frac=1)
                x_mat_tr = np.array([cust.x for cust in pop.train_customers])
                D_vec_tr = np.array([cust.D_i for cust in pop.train_customers])
                y_vec_tr = np.array([cust.y for cust in pop.train_customers])
                
                algo_picked_M = picked_M[f'{algo}_picked_M']
                if algo == "gmm-standard":
                    bic_gmm, gmm_model = GMM_segment_and_estimate(pop, algo_picked_M, x_mat_tr, D_vec_tr, y_vec_tr, algo, include_interactions, random_state=seed)  
                    assign_new_customers_to_segments(pop, pop.implement_customers, gmm_model, algo)
                
                elif algo == "gmm-da":
                    DA_score_gmm, gmm_model = GMM_segment_and_estimate(pop, algo_picked_M, x_mat_tr, D_vec_tr, y_vec_tr, algo, include_interactions, random_state=seed)  
                    assign_new_customers_to_segments(pop, pop.implement_customers, gmm_model, algo)
                
                elif algo == "policy_tree":
                    depth_policy_tree = 1 if algo_picked_M <= 2 else (2 if algo_picked_M <=4 else (3 if algo_picked_M <= 8 else 4))
                    _, optimal_policy_tree, leaf_to_pruned_segment = policy_tree_segment_and_estimate(pop, depth_policy_tree, algo_picked_M, x_mat_tr, D_vec_tr, y_vec_tr, use_hybrid_method=False, include_interactions=include_interactions)
                    assign_new_customers_to_pruned_tree(optimal_policy_tree, pop, pop.implement_customers, leaf_to_pruned_segment, algo)

                elif algo == "dast":
                    depth_dast = 1 if algo_picked_M <= 2 else (2 if algo_picked_M <=4 else (3 if algo_picked_M <= 6 else 4))
                    optimal_dast_tree, _, segment_dict = DAST_segment_and_estimate(pop, algo_picked_M, max_depth=depth_dast, min_leaf_size=2, algo=algo, include_interactions=include_interactions, use_hybrid_method=args.use_hybrid_method)
                    optimal_dast_tree.predict_segment(pop.implement_customers, segment_dict) # Assign implementation customers to segments 
                
                elif algo == "mst":
                    depth_mst = 1 if algo_picked_M <= 2 else (2 if algo_picked_M <=4 else (3 if algo_picked_M <= 6 else 4))
                    optimal_mst_tree, mst_val_score, segment_dict = MST_segment_and_estimate(pop, algo_picked_M, max_depth=depth_mst, min_leaf_size=2, epsilon=1e-2, threshold_grid=30, algo=algo, include_interactions=include_interactions)
                    optimal_mst_tree.predict_segment(pop.implement_customers, segment_dict)
                
                elif algo == "kmeans-standard":
                    silhouette_score, kmeans_model = KMeans_segment_and_estimate(pop, algo_picked_M, x_mat_tr, D_vec_tr, y_vec_tr, algo, include_interactions, random_state=seed)
                    assign_new_customers_to_segments(pop, pop.implement_customers, kmeans_model, algo)
                
                elif algo == "kmeans-da":
                    DA_score_kmeans, kmeans_model = KMeans_segment_and_estimate(pop, algo_picked_M, x_mat_tr, D_vec_tr, y_vec_tr, algo, include_interactions, random_state=seed)
                    assign_new_customers_to_segments(pop, pop.implement_customers, kmeans_model, algo)
                
                elif algo == "clr-standard":
                    bic_clr, CLR = CLR_segment_and_estimate(pop, algo_picked_M, x_mat_tr, D_vec_tr, y_vec_tr, args.kmeans_coef, num_tries=8, algo=algo, include_interactions=include_interactions, random_state=seed)
                    assign_new_customers_to_segments(pop, pop.implement_customers, CLR, algo)
                elif algo == "clr-da":
                    DA_score_clr, CLR = CLR_segment_and_estimate(pop, algo_picked_M, x_mat_tr, D_vec_tr, y_vec_tr, args.kmeans_coef, num_tries=8, algo=algo, include_interactions=include_interactions, random_state=seed)
                    assign_new_customers_to_segments(pop, pop.implement_customers, CLR, algo)   
                
                elif algo == "t_learner":
                    meta_learner_seg_labels_impl, action_identity = T_learner(pop.implement_customers, x_mat_tr, D_vec_tr, y_vec_tr)
                
                elif algo == "s_learner":
                    meta_learner_seg_labels_impl, action_identity = S_learner(pop.implement_customers, x_mat_tr, D_vec_tr, y_vec_tr)
                
                elif algo == "x_learner":
                    meta_learner_seg_labels_impl, action_identity = X_learner(pop.implement_customers, x_mat_tr, D_vec_tr, y_vec_tr)
                
                elif algo == "causal_forest":
                    meta_learner_seg_labels_impl, action_identity = causal_forest_predict(pop.implement_customers, x_mat_tr, D_vec_tr, y_vec_tr)
                
                elif algo == "dr_learner":
                    meta_learner_seg_labels_impl, action_identity = DR_learner(pop.implement_customers, x_mat_tr, D_vec_tr, y_vec_tr)
                    
                # Evaluate implementation outcome
                implementation_outcome = 0
                for i in range(len(pop.implement_customers)): 
                    cust = pop.implement_customers[i]
                    if algo in ["t_learner", "x_learner", "s_learner", "dr_learner", "causal_forest"]:
                        implement_action = action_identity[meta_learner_seg_labels_impl[i]]
                        implementation_outcome += cust.evaluate_profits(algo, implement_action)
                    else:
                        implementation_outcome += cust.evaluate_profits(algo)
                print(f"Implementation outcome for {algo}: {implementation_outcome:.2f} with chosen M = {algo_picked_M}")
                algo_result_dict[algo]['implementation_profits'] = implementation_outcome
                
                
                # oracle_profits_implementation = policy_oracle(pop.implement_customers, algo)
                
                
        except:
            import traceback
            traceback.print_exc()
            continue
        
        for algo in args.algorithms:
            exp_result_dict[algo].append(algo_result_dict[algo])
        
        exp_result_dict['seed'].append(seed)
        
        # print(f"Oracle profits: {oracle_profits_implementation['oracle_profit']:.2f}")

        # save the result after each simulation and print simulation number
        # Use first algorithm in the list to track progress
        first_algo = args.algorithms[0]
        print(f"Completed {len(exp_result_dict[first_algo])} / {args.N_sims} simulations.")
        
        if args.save_file is not None:
            with open(args.save_file, "wb") as f:
                print(f"Saving results to {args.save_file}")
                pickle.dump(exp_result_dict, f)

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")




if __name__ == "__main__":
    args = parse_args()

    print("==== Final Experiment Configuration ====")
    print(json.dumps(vars(args), indent=4))

    outcome_type = args.outcome_type

    def _is_set(name):
        return getattr(args, name, None) is not None

    # ── Parameters that belong exclusively to each outcome type ──────────────
    CONTINUOUS_ONLY = ['alpha_range', 'beta_range', 'tau_range', 'Y_noise_std_scale']
    CONTINUOUS_OPTIONAL = ['delta_range']   # allowed but not required for continuous
    DISCRETE_ONLY = ['p_range']

    if outcome_type == 'continuous':
        # Required params
        for r in CONTINUOUS_ONLY + ['x_mean_range']:
            if not _is_set(r):
                raise ValueError(f"outcome_type='continuous' requires '--{r}' to be set.")
        # Forbidden params
        for f in DISCRETE_ONLY:
            if _is_set(f):
                raise ValueError(
                    f"'--{f}' is only valid for outcome_type='discrete'. "
                    f"Remove it when using outcome_type='continuous'."
                )
        param_range = {
            "alpha": tuple(args.alpha_range),
            "beta":  tuple(args.beta_range),
            "tau":   tuple(args.tau_range),
            "delta": tuple(args.delta_range) if _is_set('delta_range') else None,
            "x_mean": tuple(args.x_mean_range),
            "p": None,
        }

    elif outcome_type == 'discrete':
        # Required params
        for r in DISCRETE_ONLY + ['x_mean_range']:
            if not _is_set(r):
                raise ValueError(f"outcome_type='discrete' requires '--{r}' to be set.")
        # Forbidden params
        for f in CONTINUOUS_ONLY + CONTINUOUS_OPTIONAL:
            if _is_set(f):
                raise ValueError(
                    f"'--{f}' is only valid for outcome_type='continuous'. "
                    f"Remove it when using outcome_type='discrete'."
                )
        param_range = {
            "alpha": None,
            "beta":  None,
            "tau":   None,
            "delta": None,
            "x_mean": tuple(args.x_mean_range),
            "p": tuple(args.p_range),
        }

    else:
        raise ValueError(f"Unknown outcome_type: '{outcome_type}'. Must be 'continuous' or 'discrete'.")

    main(args, param_range)