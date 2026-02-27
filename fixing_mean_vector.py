from ground_truth import PopulationSimulator
from plot import plot_ground_truth, plot_segmentation, plot_implementation_clustering
from gmm import GMM_segment_and_estimate
from oracle import structure_oracle, policy_oracle, policy_oracle_implementation
import pandas as pd
import numpy as np
from policy_tree import policy_tree_segment_and_estimate, assign_new_customers_to_pruned_tree
from dast import DAST_segment_and_estimate
from mst import MST_segment_and_estimate
from kmeans import KMeans_segment_and_estimate
from clr import CLR_segment_and_estimate
from utils import assign_new_customers_to_segments, pick_M_for_algo, load_config, merge_config, parse_args
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from tqdm import trange
import random
import multiprocessing
import pickle
multiprocessing.set_start_method('fork') 
import time
import json
def debug_segment_comparison(pop, algo):
    """
    Debug function to compare segment assignments and estimated parameters.
    
    Compares:
    1. Segment assignments for implementation customers
    2. Estimated parameters (alpha, beta, tau, action) for each segment
    3. Individual customer profits
    """
    print(f"\n{'='*70}")
    print(f"DEBUG COMPARISON FOR: {algo}")
    print(f"{'='*70}")
    
    # Get estimated segments
    segments = pop.est_segments_list[algo]
    print(f"\nNumber of segments: {len(segments)}")
    
    # Print segment parameters
    print(f"\n{'Seg ID':<8} {'Alpha':<10} {'Beta':<15} {'Tau':<10} {'Action':<8}")
    print("-" * 70)
    for seg in segments:
        beta_str = f"[{seg.est_beta[0]:.3f}]" if len(seg.est_beta) == 1 else str(seg.est_beta)
        print(f"{seg.segment_id:<8} {seg.est_alpha:<10.4f} {beta_str:<15} {seg.est_tau:<10.4f} {seg.est_action:<8}")
    
    # Count customers per segment (implementation)
    seg_counts = {}
    for cust in pop.implement_customers:
        seg_id = cust.est_segment[algo].segment_id
        seg_counts[seg_id] = seg_counts.get(seg_id, 0) + 1
    
    print(f"\nImplementation customers per segment:")
    for seg_id, count in sorted(seg_counts.items()):
        print(f"  Segment {seg_id}: {count} customers")
    

def compare_two_algorithms(pop, algo1, algo2):
    """
    Deep comparison between two algorithms to find profit differences.
    """
    print(f"\n{'='*80}")
    print(f"DEEP COMPARISON: {algo1} vs {algo2}")
    print(f"{'='*80}")
    
    # Compare segment parameters
    segments1 = pop.est_segments_list[algo1]
    segments2 = pop.est_segments_list[algo2]
    true_segments = pop.true_segments
    
    
    # Check if any customers are assigned to different segments
    diff_assignments = []
    for idx, cust in enumerate(pop.implement_customers):
        if algo1 in cust.est_segment and algo2 in cust.est_segment:
            seg1_id = cust.est_segment[algo1].segment_id
            seg2_id = cust.est_segment[algo2].segment_id
            if seg1_id != seg2_id:
                diff_assignments.append((idx, seg1_id, seg2_id))
    
    # Check for action differences (even if assigned to same logical segment)
    action_diffs = []
    profit_diffs = []
    for idx, cust in enumerate(pop.implement_customers):
        if algo1 in cust.est_segment and algo2 in cust.est_segment:
            action1 = cust.est_segment[algo1].est_action
            action2 = cust.est_segment[algo2].est_action
            true_action = cust.true_segment.action
            
            # Get customer info
            x_cov = cust.x
            seg1_id = cust.est_segment[algo1].segment_id
            seg2_id = cust.est_segment[algo2].segment_id
            tau1 = cust.est_segment[algo1].est_tau
            tau2 = cust.est_segment[algo2].est_tau
            true_tau = cust.true_segment.tau
            
            # Calculate individual profits
            profit1 = cust.evaluate_profits(algo1)
            profit2 = cust.evaluate_profits(algo2)
            profit_diff = profit1 - profit2
            
            if action1 != action2:
                action_diffs.append((idx, action1, action2, true_action, x_cov, 
                                   seg1_id, seg2_id, tau1, tau2, true_tau, profit_diff))
            
            if abs(profit_diff) > 1e-6:
                profit_diffs.append((idx, profit1, profit2, profit_diff))
    
    print(f"\n⚠️  Action Differences: {len(action_diffs)} customers have different recommended actions")
    if len(action_diffs) > 0:
        # Count which algorithm is more often correct
        algo1_correct = sum([1 for item in action_diffs if item[1] == item[3]])
        algo2_correct = sum([1 for item in action_diffs if item[2] == item[3]])
        both_wrong = sum([1 for item in action_diffs if item[1] != item[3] and item[2] != item[3]])
        
        print(f"\n   📊 Accuracy on disagreements:")
        print(f"      {algo1} correct: {algo1_correct}/{len(action_diffs)} ({100*algo1_correct/len(action_diffs):.1f}%)")
        print(f"      {algo2} correct: {algo2_correct}/{len(action_diffs)} ({100*algo2_correct/len(action_diffs):.1f}%)")
        print(f"      Both wrong: {both_wrong}/{len(action_diffs)} ({100*both_wrong/len(action_diffs):.1f}%)")
        
        print(f"\n   Detailed disagreements (first 10):")
        print(f"   {'Idx':<6} {'X':<10} {algo1+' seg':<10} {algo2+' seg':<10} {'True τ':<10} {algo1+' act':<11} {algo2+' act':<11} {'Δ profit':<10}")
        print(f"   {'-'*130}")
        for idx, a1, a2, true_a, x_cov, seg1, seg2, tau1, tau2, true_tau, pdiff in action_diffs[:10]:
            match1 = "✓" if a1 == true_a else "✗"
            match2 = "✓" if a2 == true_a else "✗"
            # Format covariates
            if len(x_cov) <= 2:
                x_str = "[" + ", ".join([f"{v:.2f}" for v in x_cov]) + "]"
            else:
                x_str = "[" + ", ".join([f"{v:.2f}" for v in x_cov[:2]]) + "...]"
            print(f"   {idx:<6} {x_str:<20} {seg1:<10} {seg2:<10} {true_tau:+8.3f} {a1} {match1:<9} {a2} {match2:<9} {pdiff:+8.4f}")
    
    print(f"\n💰 Profit Differences: {len(profit_diffs)} customers have different profits")
    if len(profit_diffs) > 0:
        print(f"   First 10:")
        for idx, p1, p2, pdiff in profit_diffs[:10]:
            print(f"   Customer {idx}: {algo1} profit={p1:.4f}, {algo2} profit={p2:.4f}, diff={pdiff:.4f}")
        
        total_diff = sum([pd for _, _, _, pd in profit_diffs])
        avg_diff = sum([pd for _, _, _, pd in profit_diffs]) / len(profit_diffs) if len(profit_diffs) > 0 else 0

    # Calculate total profits
    total1 = sum([cust.evaluate_profits(algo1) for cust in pop.implement_customers])
    total2 = sum([cust.evaluate_profits(algo2) for cust in pop.implement_customers])
    
    print(f"\n💵 Total Implementation Profits:")
    print(f"   {algo1}: {total1:.4f}")
    print(f"   {algo2}: {total2:.4f}")
    print(f"   Difference: {total1 - total2:.4f} ({((total1-total2)/total2*100):.4f}%)")
    
    print(f"{'='*80}\n")

def main(args):
    '''
    In this main function, we fix the experiment parameters, such as K, d, N, signal strength, noise level, etc.
    for each simulation, we generate a random new population,
    run all algorithms to segment and estimate,
    pick M for each algorithm based on validation set,
    assign implementation customers to segments,
    and evaluate implementation profits.
    '''
    
    N_total_pilot_customers = args.N_segment_size * args.K
    N_total_implement_customers = int(N_total_pilot_customers * args.implementation_scale)
    M_range = list(range(max(2, args.K-3), args.K+4))

    exp_result_dict = {
    "exp_params": {
        "K": args.K,
        "d": args.d,
        "partial_x": args.partial_x,
        "X_noise_std_scale": args.X_noise_std_scale,
        "Y_noise_std_scale": args.Y_noise_std_scale,
        "param_range": param_range,
        "N_segment_size": args.N_segment_size,
        "DR_generation_method": args.DR_generation_method,
        "kmeans_coef": args.kmeans_coef,
        "N_total_pilot_customers": N_total_pilot_customers,
        "implementation_scale": args.implementation_scale,
    },
    
    **{algo: [] for algo in args.algorithms},
    }


    start_time = time.time()
    
    # Fixed mean vectors for reproducibility (3 clusters for d=1)
    # X_mean_vectors = np.random.uniform(-20, 20, size=(3, 1))
    X_mean_vectors = np.array([
        [-2.14],
        [8.75],
        [13.58],
    ])
    print(f"X mean vectors for segments (shape={X_mean_vectors.shape}): {[x[0] for x in X_mean_vectors]}")

    for _ in trange(args.N_sims):
        
        seed = random.randint(0, 100000)
        np.random.seed(65484)   # 80707, 5909, 67691
        random.seed(65484)
        print(f"Random seed: {seed}")
        
        pop = PopulationSimulator(N_total_pilot_customers, 
                                  N_total_implement_customers, 
                                  args.d, 
                                  args.K, 
                                  args.disturb_covariate_noise, 
                                  param_range, 
                                  args.DR_generation_method, 
                                  args.partial_x, 
                                  X_mean_vectors,
                                  X_noise_std_scale=args.X_noise_std_scale,
                                  Y_noise_std_scale=args.Y_noise_std_scale,
                                  disallowed_ball_radius=args.disallowed_ball_radius
                                  )
        
        # Print Ground Truth Information
        print(f"\n{'='*70}")
        print(f"Ground Truth Segments (K={len(pop.true_segments)}):")
        print(f"{'Seg':<5} {'Alpha':<10} {'Tau':<10} {'Action':<8} {'X_mean':<20}")
        print("-" * 70)
        for seg in pop.true_segments:
            x_mean_str = "[" + ", ".join([f"{v:.2f}" for v in seg.x_mean[:min(3, len(seg.x_mean))]]) + "]"
            if len(seg.x_mean) > 3:
                x_mean_str = x_mean_str[:-1] + "...]"
            print(f"{seg.segment_id:<5} {seg.alpha:<10.4f} {seg.tau:<10.4f} {seg.action:<8} {x_mean_str:<20}")
        print(f"{'='*70}\n")
        
        # BUG FIX: 重置随机种子，确保后续算法使用相同的随机数序列
        # 无论 implementation_scale 如何，pilot customers 和算法训练都保持一致
        np.random.seed(92)
        random.seed(92)

        

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
                    
                    # Perform segmentation and estimation
                    # What happened in each function
                    # 1. Segment customers into M segments
                    # 2. For each segment, estimate parameters
                    # 3. Assign each train customer to estimated segment
                    # 4. Return labels and validation score
                    
                    if algo == "gmm-standard":
                        bic_gmm, _ = GMM_segment_and_estimate(pop, M, x_mat_tr, D_vec_tr, y_vec_tr, algo, random_state=seed)
                    
                    elif algo == "gmm-da":
                        DA_score_gmm, _ = GMM_segment_and_estimate(pop, M, x_mat_tr, D_vec_tr, y_vec_tr, algo, random_state=seed)
                        
                    elif algo == "kmeans-standard":
                        silhouette_score, _ = KMeans_segment_and_estimate(pop, M, x_mat_tr, D_vec_tr, y_vec_tr, algo, random_state=seed)
                    
                    elif algo == "kmeans-da":
                        DA_score_kmeans, _ = KMeans_segment_and_estimate(pop, M, x_mat_tr, D_vec_tr, y_vec_tr, algo, random_state=seed)
                    
                    elif algo == "clr-standard":
                        bic_clr, _ = CLR_segment_and_estimate(pop, M, x_mat_tr, D_vec_tr, y_vec_tr, kmeans_coef=args.kmeans_coef, num_tries=8, algo=algo, random_state=seed)
                    
                    elif algo == "clr-da":
                        DA_score_clr, _ = CLR_segment_and_estimate(pop, M, x_mat_tr, D_vec_tr, y_vec_tr, kmeans_coef=args.kmeans_coef, num_tries=8, algo=algo, random_state=seed)
                        
                    elif algo == "dast":
                        dast_tree, dast_val_score, segment_dict = DAST_segment_and_estimate(pop, M, max_depth=depth_dast, min_leaf_size=2, epsilon=1e-2, algo=algo, debug=True)
                    
                    elif algo == "mst":
                        mst_tree, mst_val_score, segment_dict = MST_segment_and_estimate(pop, M, max_depth=depth_mst, min_leaf_size=2, epsilon=1e-2, threshold_grid=40, algo=algo)
                        
                    elif algo == "policy_tree":
                        policy_tree_val_score, _, _ = policy_tree_segment_and_estimate(pop, depth_policy_tree, M, x_mat_tr, D_vec_tr, y_vec_tr, x_mat_val, D_vec_val, y_vec_val, buff=False)
                    
                    elif algo == "policy_tree-buff":
                        policy_tree_buff_val_score, _, _ = policy_tree_segment_and_estimate(pop, depth_policy_tree, M, x_mat_tr, D_vec_tr, y_vec_tr, x_mat_val, D_vec_val, y_vec_val, buff=True)
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
                    P_metrics = policy_oracle(pop.pilot_customers, algo=algo) #TODO: apply to implement customers
                    
                    # Record all
                    results_M.append({
                        "M": M,
                        "dast_val": dast_val_score if algo == "dast" else None,
                        "policy_tree_val": policy_tree_val_score if algo == "policy_tree" else None,
                        "policy_tree-buff_val": policy_tree_buff_val_score if algo == "policy_tree-buff" else None,
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
                
                # DEBUG: Print validation scores for each M
                if algo == "dast":
                    print(f"\n{'='*60}")
                    print(f"Algorithm: {algo}")
                    print(f"{'='*60}")
                    if f'{algo}_val' in df_results_M.columns:
                        print(f"\nValidation Scores for each M:")
                        for idx, row in df_results_M.iterrows():
                            score = row[f'{algo}_val']
                            if score is not None:
                                print(f"  M = {row['M']:2d}: val_score = {score:8.6f}")
                            else:
                                print(f"  M = {row['M']:2d}: val_score = None")
                        
                        max_idx = df_results_M[f'{algo}_val'].idxmax()
                        print(f"\n  → Maximum validation score at M = {df_results_M.at[max_idx, 'M']} (score = {df_results_M.at[max_idx, f'{algo}_val']:.6f})")
                    print(f"{'='*60}\n")

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
                
                # DEBUG: Print picked M
                if algo == "dast":
                    print(f"📌 DAST picked M = {picked_M['dast_picked_M']}\n")
                
                # for metric, picked_m_val in picked_M.items():
                #     idx = df_results_M[df_results_M['M'] == picked_m_val].index[0]
                #     print(f"  {metric}: \tM = {picked_m_val}, manager profit = {df_results_M.loc[idx, 'manager_profit']:.2f}")


                algo_result_dict[algo] = {
                    "picked_M": picked_M,
                    "profit_at_manager_picked_M": df_results_M.loc[df_results_M['M'] == picked_M[f'{algo}_picked_M'], 'manager_profit'].values[0],
                    "ARI": S_metrics["ARI"],
                    "NMI": S_metrics["NMI"],
                    "MSE_param": E_metrics["MSE_param"],
                    "MSE_outcome": E_metrics["MSE_outcome"],
                    "regret": P_metrics["regret"],
                    "mistreatment_rate": P_metrics["mistreatment_rate"],
                }
                
                # Retrain and assign implementation customers to segments  
                
                pop.split_pilot_customers_into_train_and_validate(train_frac=1)
                x_mat_tr = np.array([cust.x for cust in pop.train_customers])
                D_vec_tr = np.array([cust.D_i for cust in pop.train_customers]).reshape(-1, 1)
                y_vec_tr = np.array([cust.y for cust in pop.train_customers]).reshape(-1, 1)
                
                algo_picked_M = picked_M[f'{algo}_picked_M']
                if algo == "gmm-standard":
                    bic_gmm, gmm_model = GMM_segment_and_estimate(pop, algo_picked_M, x_mat_tr, D_vec_tr, y_vec_tr, algo, random_state=seed)  
                    assign_new_customers_to_segments(pop, pop.implement_customers, gmm_model, algo)
                
                elif algo == "gmm-da":
                    DA_score_gmm, gmm_model = GMM_segment_and_estimate(pop, algo_picked_M, x_mat_tr, D_vec_tr, y_vec_tr, algo, random_state=seed)  
                    assign_new_customers_to_segments(pop, pop.implement_customers, gmm_model, algo)
                
                elif algo == "policy_tree":
                    depth_policy_tree = 1 if algo_picked_M <= 2 else (2 if algo_picked_M <=4 else (3 if algo_picked_M <= 8 else 4))
                    _, optimal_policy_tree, leaf_to_pruned_segment = policy_tree_segment_and_estimate(pop, depth_policy_tree, algo_picked_M, x_mat_tr, D_vec_tr, y_vec_tr, buff=False)
                    assign_new_customers_to_pruned_tree(optimal_policy_tree, pop, pop.implement_customers, leaf_to_pruned_segment, algo)
                elif algo == "policy_tree-buff":
                    depth_policy_tree = 1 if algo_picked_M <= 2 else (2 if algo_picked_M <=4 else (3 if algo_picked_M <= 8 else 4))
                    _, optimal_policy_tree_buff, leaf_to_pruned_segment = policy_tree_segment_and_estimate(pop, depth_policy_tree, algo_picked_M, x_mat_tr, D_vec_tr, y_vec_tr, buff=True)
                    assign_new_customers_to_pruned_tree(optimal_policy_tree_buff, pop, pop.implement_customers, leaf_to_pruned_segment, algo)
                    
                elif algo == "dast":
                    depth_dast = 1 if algo_picked_M <= 2 else (2 if algo_picked_M <=4 else (3 if algo_picked_M <= 8 else 4))
                    print(f"\n🎯 Final DAST training with M={algo_picked_M}, depth={depth_dast}")
                    optimal_dast_tree, _, segment_dict = DAST_segment_and_estimate(pop, algo_picked_M, depth_dast, min_leaf_size=2, epsilon=0.0, algo=algo, debug=False)
                    optimal_dast_tree.predict_segment(pop.implement_customers, segment_dict) # Assign implementation customers to segments 
                
                elif algo == "mst":
                    depth_mst = 1 if algo_picked_M <= 2 else (2 if algo_picked_M <=4 else (3 if algo_picked_M <= 8 else 4))
                    optimal_mst_tree, mst_val_score, segment_dict = MST_segment_and_estimate(pop, algo_picked_M, max_depth=depth_mst, min_leaf_size=2, epsilon=1e-3, threshold_grid=30, algo=algo)
                    optimal_mst_tree.predict_segment(pop.implement_customers, segment_dict)
                
                elif algo == "kmeans-standard":
                    silhouette_score, kmeans_model = KMeans_segment_and_estimate(pop, algo_picked_M, x_mat_tr, D_vec_tr, y_vec_tr, algo, random_state=seed)
                    assign_new_customers_to_segments(pop, pop.implement_customers, kmeans_model, algo)
                
                elif algo == "kmeans-da":
                    DA_score_kmeans, kmeans_model = KMeans_segment_and_estimate(pop, algo_picked_M, x_mat_tr, D_vec_tr, y_vec_tr, algo, random_state=seed)
                    assign_new_customers_to_segments(pop, pop.implement_customers, kmeans_model, algo)
                
                elif algo == "clr-standard":
                    bic_clr, CLR = CLR_segment_and_estimate(pop, algo_picked_M, x_mat_tr, D_vec_tr, y_vec_tr, args.kmeans_coef, num_tries=8, algo=algo, random_state=seed)
                    assign_new_customers_to_segments(pop, pop.implement_customers, CLR, algo)
                elif algo == "clr-da":
                    DA_score_clr, CLR = CLR_segment_and_estimate(pop, algo_picked_M, x_mat_tr, D_vec_tr, y_vec_tr, args.kmeans_coef, num_tries=8, algo=algo, random_state=seed)
                    assign_new_customers_to_segments(pop, pop.implement_customers, CLR, algo)   
                
                # Evaluate implementation outcome
                implementation_outcome = 0
                for cust in pop.implement_customers:
                    implementation_outcome += cust.evaluate_profits(algo)
                print(f"Implementation outcome for {algo}: {implementation_outcome:.2f} with chosen M = {algo_picked_M}")
                algo_result_dict[algo]['implementation_profits'] = implementation_outcome
                
                # DEBUG: Compare DAST and KMeans if both have perfect segmentation
                if algo in ["dast", "kmeans-da", "kmeans-standard"] and args.debug_comparison:
                    debug_segment_comparison(pop, algo)
                
                # Plot implementation customers clustering results
                if args.plot:
                    plot_implementation_clustering(pop.implement_customers, algo)
                
                oracle_profits_implementation = policy_oracle(pop.implement_customers, algo)
                
                
        except:
            import traceback
            traceback.print_exc()
            continue
        
        for algo in args.algorithms:
            exp_result_dict[algo].append(algo_result_dict[algo])
        
        # Deep comparison between DAST and KMeans if both are present
        if args.debug_comparison:
            compare_two_algorithms(pop, "dast", "mst")
    
        # print(f"Oracle profits: {oracle_profits_implementation['oracle_profit']:.2f}")

        # save the result after each simulation and print simulation number
        
        print(f"Completed {len(exp_result_dict['dast'])} / {args.N_sims} simulations.")
        
        if args.save_file is not None:
            with open(args.save_file, "wb") as f:
                print(f"Saving results to {args.save_file}")
                pickle.dump(exp_result_dict, f)

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")



if __name__ == "__main__":
    args_user = parse_args()
    config_yml = load_config(args_user.config)
    merged_config = merge_config(args_user, config_yml)
    
    param_range = {
        "alpha": tuple(merged_config.alpha_range),
        "beta": tuple(merged_config.beta_range),
        "tau": tuple(merged_config.tau_range),
        "x_mean": tuple(merged_config.x_mean_range),
    }

    print("==== Final Experiment Configuration ====")
    print(json.dumps(vars(merged_config), indent=4))


    main(merged_config)