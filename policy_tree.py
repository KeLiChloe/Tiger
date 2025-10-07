import numpy as np
from itertools import combinations
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, default_converter
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from sklearn.linear_model import LinearRegression
from ground_truth import PopulationSimulator, SegmentEstimate
from utils import assign_trained_customers_to_segments, estimate_segment_parameters, plot_segment_sankey, evaluate_on_validation, compute_node_DR_value
from IPython.display import SVG, display



# Import R packages
grf = importr('grf')
policytree = importr('policytree')
DiagrammeRsvg = importr('DiagrammeRsvg')
grdevices = importr('grDevices')

# Load R libraries
ro.r('library(policytree)')
ro.r('library(DiagrammeRsvg)')

# Define R function to extract leaf-parent relationships
ro.r('''
    extract_leaf_parent_map <- function(tree) {
    node_list <- as.list(tree)$nodes
    leaf_to_parent <- list()

    # Helper: walk by index (each node is stored by ID/index)
    walk_tree <- function(node_id, parent_id = NA) {
        node <- node_list[[node_id]]
        if (is.null(node)) return(NULL)

        if (!is.null(node$is_leaf) && node$is_leaf) {
        leaf_to_parent[[as.character(node_id)]] <<- parent_id
        } else {
        walk_tree(node$left_child, node_id)
        walk_tree(node$right_child, node_id)
        }
    }

    walk_tree(1)  # Start from root node ID = 1
    return(leaf_to_parent)
    }
    ''')


def compute_gamma_in_policy_tree_R(X_r, y_r, D_r, depth):
    ro.globalenv['X_r'] = X_r
    ro.globalenv['y_r'] = y_r
    ro.globalenv['D_r'] = D_r
    cforest = grf.causal_forest(X_r, y_r, D_r)
    Gamma_r = policytree.double_robust_scores(cforest)
    ro.globalenv['Gamma'] = Gamma_r
    ro.r(f'tree <- policy_tree(X_r, Gamma, depth={depth})') 
    with localconverter(default_converter + numpy2ri.converter):
        Gamma = ro.conversion.rpy2py(Gamma_r)
    return Gamma

def policy_tree_segment_and_estimate(pop: PopulationSimulator, depth: int, target_leaf_num: int, x_mat, D_vec, y_vec):
    """
    Perform policy tree-based segmentation and OLS-based estimation per segment.

    Parameters:
        pop: PopulationSimulator object with simulated data
        depth: maximum depth of the policy tree
        random_state: optional random seed for reproducibility
    """
    
    with localconverter(default_converter + numpy2ri.converter):
        X_r = ro.conversion.py2rpy(x_mat)
        y_r = ro.conversion.py2rpy(y_vec)
        D_r = ro.conversion.py2rpy(D_vec)
    
    # Gamma = np.array(pop.gamma[[cust.customer_id for cust in pop.train_customers]])  # shape (N, 2)
    Gamma = compute_gamma_in_policy_tree_R(X_r, y_r, D_r, depth)
    
    # Extract tree structure
    leaf_to_parent_r = ro.r('extract_leaf_parent_map(tree)')
    leaf_to_parent_map = {
        int(k): int(leaf_to_parent_r.rx2(k)[0])
        for k in leaf_to_parent_r.names
    }

    # Predict segments/actions for training set
    tree = ro.r('tree')
    segment_r = policytree.predict_policy_tree(tree, X_r, type="node.id")
    action_r = policytree.predict_policy_tree(tree, X_r, type="action.id")
    segment_labels_raw = list(ro.conversion.rpy2py(segment_r))

    action_ids_raw = list(ro.conversion.rpy2py(action_r))  # (by default value 1 = action 0 (1st col in Gamma), value 2 = action 1 (2nd col in Gamma) )
    action_ids = np.array([a - 1 for a in action_ids_raw]) # Normalize segment and action labels to start from 0
    
    segment_labels_pruned, action_ids_pruned, leaf_to_pruned_segment = post_prune_tree(
        Y = y_vec,
        D = D_vec,
        segment_labels=np.array(segment_labels_raw), # ✅ raw IDs
        action_ids=np.array(action_ids),
        Gamma=Gamma,
        target_leaf_num=target_leaf_num,
        leaf_to_parent_map=leaf_to_parent_map
    )

    # Assign each train customer to estimated segment
    estimate_segment_and_assign(pop, target_leaf_num, segment_labels_pruned, x_mat, D_vec, y_vec)

    # assign validation customers to segments
    val_score = None
    if len(pop.val_customers) > 0:
        assign_new_customers_to_pruned_tree(tree, pop, pop.val_customers, leaf_to_pruned_segment)
        val_score = evaluate_on_validation(pop, algo="policy_tree")
    
    # plot_segment_sankey(segment_labels, segment_labels_pruned)
    
    # Clean up R environment to avoid memory leaks
    ro.r('rm(tree, X_r, Gamma)')
    ro.r('gc()')  # Trigger R's garbage collector
    return val_score, tree, leaf_to_pruned_segment



def estimate_segment_and_assign(pop: PopulationSimulator, target_leaf_num, segment_labels, x_mat, D_vec, y_vec):
    """
    Estimate parameters for each segment and assign customers to segments.
    Returns:
        None, modifies pop.customers in-place
    """
    # important to reset!!!
    pop.est_segments_list["policy_tree"] = []  # Reset
    
    for m in range(target_leaf_num):
        idx_m = np.where(segment_labels == m)[0]
        if len(idx_m) == 0:
            raise ValueError(f"No customers assigned to segment {m}. ")

        x_m = x_mat[idx_m]
        D_m = D_vec[idx_m]
        y_m = y_vec[idx_m]

        est_alpha, est_beta, est_tau, est_action = estimate_segment_parameters(x_m, D_m, y_m)
        
        # est_action = action_ids[idx_m[0]]
        est_seg = SegmentEstimate(est_alpha, est_beta, est_tau, est_action, segment_id=m)
        pop.est_segments_list["policy_tree"].append(est_seg)
    
    assign_trained_customers_to_segments(pop, segment_labels, "policy_tree")

import numpy as np
from itertools import combinations

def post_prune_tree(Y, D, segment_labels, action_ids, Gamma, target_leaf_num, leaf_to_parent_map):
    """
    Prune only sibling leaf segments (same parent in tree structure).

    Parameters:
        segment_labels: np.ndarray of segment IDs (raw R leaf IDs)
        action_ids: np.ndarray of actions per sample
        Gamma: np.ndarray (n x A), reward matrix (A actions)
        target_leaf_num: desired number of leaf segments
        leaf_to_parent_map: dict {leaf_id → parent_id}, from R
        verbose: bool, print merge steps if True

    Returns:
        pruned_segment_labels, pruned_action_ids, leaf_to_pruned_segment
    """

    # Build initial segment-to-sample index map
    # Root node idx is 1. 
    # Here the segment IDs "s" start from leaf IDs. For example, when the depth = 2, the segment leaves IDs are 4, 5, 6, 7.
    segment_map = {
        s: np.where(segment_labels == s)[0].astype(int)
        for s in sorted(set(segment_labels))
    } 
    segment_to_leaves = {s: {s} for s in segment_map}
    
    # Segment action: start from the action of (any) sample in the segment
    # (If segments were pure leaves initially, this is fine.)
    action_map = {s: int(action_ids[idxs[0]]) for s, idxs in segment_map.items()}

    while len(segment_map) > target_leaf_num:
        best_pair = None
        best_action = None
        min_welfare_loss = float("inf")

        segments = list(segment_map.keys())

        for s1, s2 in combinations(segments, 2):
            # Only merge if ALL constituent leaves share the same parent
            combined_leaves = segment_to_leaves[s1] | segment_to_leaves[s2]
            combined_parents = {leaf_to_parent_map[leaf] for leaf in combined_leaves}
            if len(combined_parents) != 1:
                continue

            # Evaluate welfare of the merged segment (choose best action for merged)
            idx1, idx2 = segment_map[s1], segment_map[s2]
            merged_idx = np.concatenate([idx1, idx2])

            # Choose the action that maximizes merged welfare
            merged_node_value, merged_tau_hat = compute_node_DR_value(Y, D, Gamma, merged_idx)
                
            merged_node_action = int(merged_tau_hat >= 0) 
            

            # Original welfare = sum of each segment's welfare under its own action
            w1 = compute_node_DR_value(Y, D, Gamma, idx1)[0]
            w2 = compute_node_DR_value(Y, D, Gamma, idx2)[0]
            original_total = w1 + w2

            loss = original_total - merged_node_value
            if loss <= min_welfare_loss:
                best_pair = (s1, s2)
                best_action = merged_node_action
                min_welfare_loss = loss

        # If no legal sibling pair exists, stop (or raise if you require strict target)
        if best_pair is None:
            raise ValueError(
                f"No legal sibling pairs found to prune to {target_leaf_num} segments. "
                f"Current segments: {len(segment_map)}"
            )

        s1, s2 = best_pair
        new_seg_id = min(s1, s2)  # keep a stable id
        drop_seg_id = s2 if new_seg_id == s1 else s1

        # Merge data
        merged_indices = np.concatenate([segment_map[s1], segment_map[s2]])
        segment_map[new_seg_id] = merged_indices
        action_map[new_seg_id] = best_action
        segment_to_leaves[new_seg_id] = segment_to_leaves[s1] | segment_to_leaves[s2]
        
        # Remove dropped segment
        for d in (drop_seg_id,):
            del segment_map[d]
            del action_map[d]
            del segment_to_leaves[d]

    # Reindex to 0-based contiguous segments
    final_segments = sorted(segment_map.keys())
    seg_id_map = {old: new for new, old in enumerate(final_segments)}

    pruned_segment_labels = np.zeros(len(segment_labels), dtype=int)
    pruned_action_ids = np.zeros(len(segment_labels), dtype=int)

    for old_seg, indices in segment_map.items():
        new_seg = seg_id_map[old_seg]
        pruned_segment_labels[indices] = new_seg
        pruned_action_ids[indices] = int(action_map[old_seg])

    # Map original leaves → pruned segment ids
    leaf_to_pruned_segment = {}
    for seg_id, leaf_set in segment_to_leaves.items():
        for leaf in leaf_set:
            if seg_id in seg_id_map:  # seg_id should exist, but be defensive
                leaf_to_pruned_segment[leaf] = seg_id_map[seg_id]

    return pruned_segment_labels, pruned_action_ids, leaf_to_pruned_segment




def plot_policy_tree(depth):
    svg_filename=f'figures/policy_tree_depth_{depth}.svg'
    svg_string = ro.r('DiagrammeRsvg::export_svg(plot(tree))')[0]
    
    with open(svg_filename, "w") as f:
        f.write(svg_string)
    display(SVG(filename=svg_filename))
    
def assign_new_customers_to_pruned_tree(tree, pop, new_customers, leaf_to_pruned_segment):
    """
    Assign each validation customer to a policy tree segment (based on pruned structure).
    """
    x_mat_new = np.array([cust.x for cust in new_customers])

    with localconverter(default_converter + numpy2ri.converter):
        X_new_r = ro.conversion.py2rpy(x_mat_new)

    
    # Predict raw segment (leaf ID, starting not from 0, but from a positive number) for each val customer
    segment_r = policytree.predict_policy_tree(tree, X_new_r, type="node.id")
    segment_ids = list(ro.conversion.rpy2py(segment_r))

    for cust, raw_leaf in zip(new_customers, segment_ids):
        pruned_seg = leaf_to_pruned_segment[raw_leaf]  # maps raw leaf ID → pruned segment index
        segment_obj = pop.est_segments_list["policy_tree"][pruned_seg]

        # assign segment to val customer
        cust.est_segment["policy_tree"] = segment_obj


