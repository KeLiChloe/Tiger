import numpy as np
from itertools import combinations
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, default_converter
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from sklearn.linear_model import LinearRegression
from ground_truth import PopulationSimulator, SegmentEstimate
from utils import build_design_matrix
from IPython.display import SVG, display
import pandas as pd
import plotly.graph_objects as go



# Import R packages
grf = importr('grf')
policytree = importr('policytree')
DiagrammeRsvg = importr('DiagrammeRsvg')
grdevices = importr('grDevices')

# Load R libraries
ro.r('library(policytree)')
ro.r('library(DiagrammeRsvg)')

def policy_tree_segment_and_estimate(pop: PopulationSimulator, depth: int, target_leaf_num: int):
    """
    Perform policy tree-based segmentation and OLS-based estimation per segment.

    Parameters:
        pop: PopulationSimulator object with simulated data
        depth: maximum depth of the policy tree
        random_state: optional random seed for reproducibility
    """
    df = pop.to_dataframe()
    x_cols = [f'x_{j}' for j in range(pop.d)]
    x_mat = df[x_cols].values
   
    y_vec = df['outcome'].values.reshape(-1, 1)
    D_vec = df['D_i'].values.reshape(-1, 1) 
    
    with localconverter(default_converter + numpy2ri.converter):
        X_r = ro.conversion.py2rpy(np.asarray(x_mat))
        Y_r = ro.conversion.py2rpy(np.asarray(y_vec))
        D_r = ro.conversion.py2rpy(np.asarray(D_vec))
    
    cforest = grf.causal_forest(X_r, Y_r, D_r)
    Gamma = policytree.double_robust_scores(cforest)
    tree = policytree.policy_tree(X_r, Gamma, depth=depth)
    
    # Plot the original policy tree 
    # plot_policy_tree(tree, depth)
    
    segment_r = policytree.predict_policy_tree(tree, X_r, type="node.id")
    action_r = policytree.predict_policy_tree(tree, X_r, type="action.id")

    # Convert back to Python lists
    segment_labels_raw = list(ro.conversion.rpy2py(segment_r))
    action_ids_raw = list(ro.conversion.rpy2py(action_r))  # (1 = action 0, 2 = action 1 by default)

    # Normalize segment and action labels to start from 0
    segment_labels = normalize_segment_labels(segment_labels_raw)
    action_ids = normalize_action_ids(action_ids_raw)
    
    segment_labels_pruned, action_ids_pruned, _ = post_prune_tree(
        segment_labels=segment_labels,
        action_ids=action_ids,
        Gamma=np.array(Gamma),
        target_leaf_num=target_leaf_num
    )
    
    # plot_segment_sankey(segment_labels, segment_labels_pruned)

    # Assign each customer to estimated segment
    estimate_segment_and_assign(pop, target_leaf_num, segment_labels_pruned, x_mat, df, action_ids_pruned)

def estimate_segment_and_assign(pop: PopulationSimulator, target_leaf_num, segment_labels,x_mat, df, action_ids):
    """
    Estimate parameters for each segment and assign customers to segments.
    Returns:
        None, modifies pop.customers in-place
    """
    pop.policy_tree_est_segments = []
    for seg_id in range(target_leaf_num):
        pop.policy_tree_est_segments.append(None)  # placeholder for now
    estimate_per_segment(pop, target_leaf_num, segment_labels, x_mat, df, action_ids)
    assign_customers_to_segments(pop, segment_labels)
 
def estimate_per_segment(pop: PopulationSimulator, n_segments:int, segment_labels, x_mat, df, action_ids):
    """
    Estimate parameters per segment using OLS regression.
    
    Parameters:
        pop: PopulationSimulator object with simulated data
        n_segments: number of segments to estimate
        
    Returns:
        None, modifies segment_estimates in-place
    """
    for m in range(n_segments):
        idx_m = np.where(segment_labels == m)[0]
        if len(idx_m) == 0:
            raise ValueError(f"No customers assigned to segment {m}. ")

        x_m = x_mat[idx_m]
        D_m = df.loc[idx_m, 'D_i'].values
        y_m = df.loc[idx_m, 'outcome'].values

        X_design = build_design_matrix(x_m, D_m)
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X_design, y_m)
        theta = reg.coef_.ravel()
        est_alpha = theta[0]
        est_beta = theta[1:-1]
        est_tau = theta[-1]
        
        # Create estimated segment object
        # The actions should be same within each segment (leafnode)
        idx_m_action = np.where(action_ids[idx_m] != action_ids[idx_m[0]])[0]
        if len(idx_m_action) > 0:
            raise ValueError(f"Multiple actions found in segment {m}. Expected single action per segment.")
        
        est_action = action_ids[idx_m[0]]
        est_seg = SegmentEstimate(est_alpha, est_beta, est_tau, est_action, segment_id=m)
        est_seg.count = len(idx_m)
        pop.policy_tree_est_segments[m] = est_seg
  
def assign_customers_to_segments(pop: PopulationSimulator, segment_labels):
    """
    Assign customers to estimated segments based on segment labels.
    
    Parameters:
        pop: PopulationSimulator object with simulated data
        segment_labels: array of segment labels for each customer
        
    Returns:
        None, modifies pop.customers in-place
    """
    for i, cust in enumerate(pop.customers):
        m = segment_labels[i]
        cust.policy_tree_est_segment = pop.policy_tree_est_segments[m]
   
def normalize_segment_labels(segment_labels_raw):
    """
    Normalize segment labels to start from 0.
    """
    unique_ids = sorted(set(segment_labels_raw))
    id_map = {orig_id: new_id for new_id, orig_id in enumerate(unique_ids)}
    normalized_labels = np.array([id_map[seg_id] for seg_id in segment_labels_raw])
    
    return normalized_labels

def normalize_action_ids(action_ids_raw):
    """
    Normalize action IDs to start from 0.
    """
    action_ids = np.array([a - 1 for a in action_ids_raw])
    return action_ids

def post_prune_tree(segment_labels, action_ids, Gamma, target_leaf_num):
    """
    Post-prune segments to reduce the number of leaves to `target_leaf_num`.

    Parameters:
        segment_labels (np.ndarray): Original segment labels (0-based)
        action_ids (np.ndarray): Action IDs (0 = control, 1 = treatment)
        Gamma (np.ndarray): N x 2 reward matrix
        target_leaf_num (int): Desired number of segments after pruning

    Returns:
        pruned_segment_labels (np.ndarray): New segment labels (0-based)
        pruned_action_ids (np.ndarray): New action for each sample
        segment_action_map (dict): Mapping from final segment to action
    """
    # Step 1: Build initial mappings from segments
    unique_segments = sorted(set(segment_labels))
    segment_map = {
        s: np.array(np.where(segment_labels == s)[0], dtype=int)
        for s in unique_segments
    }
    action_map = {
        s: action_ids[indices[0]]  # All actions in a segment are the same
        for s, indices in segment_map.items()
    }

    def seg_welfare(indices, action):
        return Gamma[indices, int(action)].mean()

    if len(segment_map) < target_leaf_num:
        raise ValueError(
            f"Number of segments ({len(segment_map)}) is less than or equal to target ({target_leaf_num})."
        )

    while len(segment_map) > target_leaf_num:
        best_pair = None
        best_action = None
        min_welfare_loss = float("inf")

        segments = list(segment_map.keys())
        for s1, s2 in combinations(segments, 2):
            idx1, idx2 = segment_map[s1], segment_map[s2]
            merged_idx = np.concatenate([idx1, idx2])

            # Evaluate welfare for each possible action
            mean_rewards = [Gamma[merged_idx, a].mean() for a in range(Gamma.shape[1])]
            best_a = int(np.argmax(mean_rewards))
            merged_welfare = mean_rewards[best_a]

            # Original weighted welfare
            w1 = seg_welfare(idx1, action_map[s1])
            w2 = seg_welfare(idx2, action_map[s2])
            n1, n2 = len(idx1), len(idx2)
            original_total = (n1 * w1 + n2 * w2) / (n1 + n2)

            loss = original_total - merged_welfare
            if loss < min_welfare_loss:
                best_pair = (s1, s2)
                best_action = best_a
                min_welfare_loss = loss

        # Merge best pair
        s1, s2 = best_pair
        new_seg_id = min(s1, s2)
        merged_indices = np.concatenate([segment_map[s1], segment_map[s2]])

        segment_map[new_seg_id] = merged_indices
        action_map[new_seg_id] = best_action
        del segment_map[s1 if s1 != new_seg_id else s2]
        del action_map[s1 if s1 != new_seg_id else s2]

    # Reindex segment labels to 0-based
    final_segments = sorted(segment_map.keys())
    seg_id_map = {old: new for new, old in enumerate(final_segments)}

    pruned_segment_labels = np.zeros(len(segment_labels), dtype=int)
    pruned_action_ids = np.zeros(len(segment_labels), dtype=int)

    for old_seg, indices in segment_map.items():
        new_seg = seg_id_map[old_seg]
        pruned_segment_labels[indices] = new_seg
        pruned_action_ids[indices] = action_map[old_seg]

    segment_action_map = {
        seg_id_map[old]: action_map[old] for old in final_segments
    }

    return pruned_segment_labels, pruned_action_ids, segment_action_map


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

def plot_policy_tree(tree, depth):
    svg_filename=f'policy_tree_depth_{depth}.svg'
    ro.globalenv['tree'] = tree
    svg_string = ro.r('DiagrammeRsvg::export_svg(plot(tree))')[0]
    
    with open(svg_filename, "w") as f:
        f.write(svg_string)
    display(SVG(filename=svg_filename))