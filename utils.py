
import numpy as np
from ground_truth import PopulationSimulator
from sklearn.linear_model import LinearRegression
import pandas as pd
import plotly.graph_objects as go
import os
import yaml
import json
import argparse

def build_design_matrix(x_array, D_array, include_interactions):
    """
    Add intercept and treatment column to covariates.
    
    Parameters:
        x_array: (N, d) array of covariates
        D_array: (N,) array of treatment indicators
        include_interactions: If True, includes treatment * covariate interactions
    
    Returns:
        Design matrix with columns: [intercept, x, D, x*D (if include_interactions)]
    """
    N, d = x_array.shape
    intercept = np.ones((N, 1))
    D_col = D_array.reshape(-1, 1)
    
    if include_interactions:
        # Add interaction terms: x_j * D for each covariate j
        interactions = x_array * D_col  # Element-wise multiplication
        return np.hstack([intercept, x_array, D_col, interactions])
    else:
        return np.hstack([intercept, x_array, D_col])





def assign_trained_customers_to_segments(pop: PopulationSimulator, segment_labels, algo):
    """
    Assign customers to estimated segments based on segment labels.
    
    Parameters:
        pop: PopulationSimulator object with all simulated data
        segment_labels: array of segment labels for each customer
        
    Returns:
        None, modifies pop.customers in-place
    """
    for i, cust in enumerate(pop.train_customers):
        m = segment_labels[i]
        cust.est_segment[algo] = pop.est_segments_list[algo][m]
        assert cust.est_segment[algo].segment_id == m, f"Segment ID mismatch for customer {cust.customer_id}: expected {m}, got {cust.est_segment[algo].segment_id}"
        

def estimate_segment_parameters(X, D, Y, include_interactions, action_num=None):
    """
    Estimate treatment effect and recommend action.
    
    STRICT REQUIREMENT: All actions in 0..action_num-1 must have samples in the data.
    
    For binary treatment (2 actions):
        - est_tau = mean(Y|D=1) - mean(Y|D=0)
        - est_action = 1 if est_tau >= 0 else 0
    
    For multi-arm treatment (>2 actions):
        - For each action a, compute mean_a = mean(Y|D=a)
        - est_action = argmax_a mean_a
        - est_tau = mean_{est_action} - mean_0 (effect relative to action 0)
    
    Parameters:
        X: (N, d) array of covariates (not used, kept for API compatibility)
        D: (N,) array of action indicators
        Y: (N,) array of outcomes
        include_interactions: not used, kept for API compatibility
        action_num: int, total number of possible actions (strict check: all must be present)
    
    Returns:
        est_tau: float, estimated treatment effect
        est_action: int, recommended action
    """
    Y = np.ravel(Y)
    D = np.ravel(D)
    
    unique_actions_in_data = np.unique(D)
    
    # Strict check: all actions 0..action_num-1 must be present
    if action_num is not None:
        for a in range(action_num):
            if a not in unique_actions_in_data:
                print(f"Warning: Action {a} missing in segment data.")
                return 404, 404
    
    
    # Compute mean outcome for each action
    action_means = {}
    for action in unique_actions_in_data:
        action_means[int(action)] = np.mean(Y[D == action])
    
    # Recommend action with highest mean outcome
    est_action = max(action_means, key=action_means.get)
    
    # Treatment effect relative to action 0
    baseline_action = 0 if 0 in action_means else min(action_means.keys())
    est_tau = action_means[est_action] - action_means[baseline_action]
    
    return est_tau, int(est_action)


def plot_segment_sankey(original, pruned):
    df = pd.DataFrame({'orig': original, 'pruned': pruned})
    flow = df.groupby(['orig', 'pruned']).size().reset_index(name='count')

    # Create node labels
    orig_labels = sorted(df['orig'].unique())
    pruned_labels = sorted(df['pruned'].unique())
    all_labels = [f'Orig {i}' for i in orig_labels] + [f'Pruned {i}' for i in pruned_labels]

    label_to_index = {label: idx for idx, label in enumerate(all_labels)}

    # Build Sankey diagram
    source = [label_to_index[f'Orig {o}'] for o in flow['orig']]
    target = [label_to_index[f'Pruned {p}'] for p in flow['pruned']]
    value = flow['count'].tolist()

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=all_labels),
        link=dict(source=source, target=target, value=value)
    )])
    fig.update_layout(title_text="Segment Merge Flow (Original → Pruned)", font_size=12)
    fig.show()
    


# estimated total profits of a segment after applying the learnt policy to the customers in that segment
def compute_node_DR_value(Y, D, gamma, indices, use_hybrid_method, action_num=None):
    D_m = D[indices]
    Y_m = Y[indices]

    unique_actions_in_data = np.unique(D_m)
    
    # Strict check: all actions 0..action_num-1 must be present
    if action_num is not None:
        for a in range(action_num):
            if a not in unique_actions_in_data:
                return 0
    elif len(unique_actions_in_data) < 2:
        return 0
    
    # Compute mean outcome for each action
    action_means = {}
    for action in unique_actions_in_data:
        Y_a = Y_m[D_m == action]
        action_means[int(action)] = np.mean(Y_a) if len(Y_a) > 0 else 0
    
    # Recommend action with highest mean outcome
    a_i = max(action_means, key=action_means.get)
    
    # Compute treatment effect relative to action 0 (or minimum action)
    baseline_action = 0 if 0 in action_means else min(action_means.keys())
    tau_hat = action_means[a_i] - action_means[baseline_action]
    
    
    if use_hybrid_method is True:
        # Method 1: Direct + Gamma
        # If D_m[i] = a_i, then we get profit Y_m[i]
        # If D_m[i] != a_i, then we get profit gamma_a_i_m[i]
        gamma_a_i_m = gamma[indices, a_i].reshape(-1, 1)
        value = np.sum(Y_m[D_m == a_i]) + np.sum(gamma_a_i_m[D_m != a_i])
    else:
        # Method 2: Gamma only
        # Sum of gamma values for the recommended action
        gamma_a_i_m = gamma[indices, a_i]
        value = np.sum(gamma_a_i_m)
    
    return value


def evaluate_on_validation(pop: PopulationSimulator, algo, Gamma_val, customers=None):
    """
    Evaluate estimated policy on a set of customers using doubly-robust value.
    
    Args:
        pop: PopulationSimulator object
        algo: Algorithm name
        Gamma_val: Gamma matrix for the customers
        customers: List of customers to evaluate (default: pop.val_customers)
    """
    # Use validation customers by default, or specified customers
    if customers is None:
        customers = pop.val_customers
    
    if len(customers) == 0:
        return None

    V = 0
    for i, cust in enumerate(customers):
        assigned_action = cust.est_segment[algo].est_action
        
        # BUG FIX: Only handle the case when action is undecided (404)
        if assigned_action == 404:
            # For algorithms that can't decide (action=404), need a fallback
            # Get action_num from true segment's action range (assuming actions are 0 to action_num-1)
            # For simplicity, use action 0 or 1 as fallback
            if algo in ["mst", "policy_tree", "gmm-standard", "gmm-da", "kmeans-standard", "kmeans-da"]:
                # These algorithms randomly assign when undecided
                # Use action 0 as safe fallback (or could randomize among available actions)
                assigned_action = 0
                cust.est_segment[algo].est_action = assigned_action
            else:
                # For other algorithms, use true action as fallback
                assigned_action = cust.true_segment.action
                cust.est_segment[algo].est_action = cust.true_segment.action
        # Otherwise, use the estimated action (0 or 1) directly - no modification needed!
        
        if int(assigned_action) == int(cust.D_i):
            V += cust.y
        else:
            V += Gamma_val[i, assigned_action]

    return V / len(customers)

def assign_new_customers_to_segments(pop: PopulationSimulator, customers, model, algo):
    """
    Assign new customers to segments based on the trained model.
    
    Parameters:
        pop: PopulationSimulator object with all simulated data
        algo: algorithm used for segmentation
    Returns:
        None, modifies pop.customers in-place
    """
    
    x_new = np.array([cust.x for cust in customers])
    if x_new.shape[0] > 0:
        labels = model.predict(x_new)
        for cust, seg_id in zip(customers, labels):
            segment = pop.est_segments_list[f"{algo}"][seg_id]
            cust.est_segment[f"{algo}"] = segment


def pick_M_for_algo(algo, df_results_M):

    val_col = f'{algo}_val'

    max_val_algos = ["gmm-da", "kmeans-da", "clr-da", "policy_tree", 
                     "dast", "mst", "kmeans-standard"]
    min_val_algos = ["gmm-standard", "clr-standard"]
    meta_learners = ["t_learner", "s_learner", "x_learner", 
                     "dr_learner", "causal_forest"]

    if algo not in max_val_algos and algo not in min_val_algos and algo not in meta_learners:
        raise ValueError(f"Unknown algorithm: {algo}")

    is_maximize = algo in max_val_algos
    is_minimize = algo in min_val_algos
    is_meta_learner = algo in meta_learners

    if is_maximize:
        best_score = df_results_M[val_col].max()
        candidates = df_results_M[df_results_M[val_col] == best_score].copy()

        if algo == "dast":
            picked_M = int(candidates["M"].max())  
        else:
            picked_M = int(candidates["M"].min())

    elif is_minimize:
        best_score = df_results_M[val_col].min()
        candidates = df_results_M[df_results_M[val_col] == best_score].copy()
        picked_M = int(candidates["M"].min())

    elif is_meta_learner:
        picked_M = "Not applicable"

    return {f'{algo}_picked_M': picked_M}




def load_config(config_path):
    """Load configuration from YAML or JSON file."""
    if not os.path.exists(config_path):
        print(f"⚠️ Config file not found: {config_path}. Using defaults.")
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            return yaml.safe_load(f)
        elif config_path.endswith(".json"):
            return json.load(f)
        else:
            raise ValueError("Unsupported config format. Use .yaml or .json")


def parse_args():
    parser = argparse.ArgumentParser(description="Simulation configuration for experiment")

    parser.add_argument("--config", type=str, default="config.yml", help="Path to configuration file")
    
    parser.add_argument("--plot", action="store_true", help="Enable plotting")
    parser.add_argument("--compute_overlap", action="store_true", help="Compute overlap between segments")
    parser.add_argument("--debug_comparison", action="store_true", help="Enable detailed debug comparison of segment estimates")


    parser.add_argument("--alpha_range", type=float, nargs=2, help="Range for alpha parameter")
    parser.add_argument("--beta_range", type=float, nargs=2, help="Range for beta parameter")
    parser.add_argument("--tau_range", type=float, nargs=2, help="Range for tau parameter")
    parser.add_argument("--delta_range", type=float, nargs=2, help="Range for delta (interaction) parameter")
    parser.add_argument("--x_mean_range", type=float, nargs=2, help="Range for x_mean parameter")

    
    parser.add_argument("--N_segment_size", type=int, help="Number of customers per segment")
    parser.add_argument("--d", type=int, help="Dimensionality of covariates")
    parser.add_argument("--partial_x", type=float, help="Fraction of x used in outcome generation")
    parser.add_argument("--K", type=int, help="Number of segments")
    parser.add_argument("--action_num", type=int, default=2, help="Number of actions (default: 2 for binary treatment)")
    parser.add_argument("--disallowed_ball_radius", type=float, help="Minimum distance between mean vectors as scale factor (e.g., 0.8 means min_dist = 0.8 * space_range/K^(1/d)). Default: 0.5")
    parser.add_argument("--X_noise_std_scale", type=float, required=True, help="Scale factor for within-cluster covariate noise as a multiple of average distance between mean vectors")
    parser.add_argument("--disturb_covariate_noise", type=float, help="Covariate noise across segments")
    parser.add_argument("--Y_noise_std_scale", type=float, required=True, help="Scale factor for outcome noise as a multiple of average |tau| (treatment effect magnitude)")
    
    parser.add_argument("--kmeans_coef", type=float, help="Coefficient for k-means weighting")
    

    parser.add_argument("--DR_generation_method", type=str, choices=["mlp", "forest", "reg"],
                        help="DR generation method (for DAST only)")
    
    # default is True
    parser.add_argument("--use_hybrid_method", type=lambda x: str(x).lower() == 'true', 
                        default=True, 
                        help="Use hybrid method for tree splitting and evaluation (default: True). Pass True or False")
    
    parser.add_argument("--implementation_scale", type=float, help="Scale of implementation population")

    parser.add_argument("--N_sims", type=int, help="Number of simulations to run")
    
    parser.add_argument("--save_file", type=str, help="Path to save experiment results")
    
    parser.add_argument("--sequence_seed", type=int, help="Random seed for simulation sequence")

    # 算法列表
    parser.add_argument("--algorithms", type=str, nargs="+",
                        help="List of algorithms to run")

    args = parser.parse_args()
    return args


from types import SimpleNamespace

def merge_config(args, config):
    merged = dict(config)
    for key, value in vars(args).items():
        if value is not None:
            merged[key] = value
    return SimpleNamespace(**merged)

