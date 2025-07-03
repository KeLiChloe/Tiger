import numpy as np
from itertools import combinations
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, default_converter
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from sklearn.linear_model import LinearRegression
from ground_truth import PopulationSimulator, SegmentEstimate
from utils import assign_trained_customers_to_segments, estimate_segment_parameters, plot_segment_sankey
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


def policy_tree_segment_and_estimate(pop: PopulationSimulator, depth: int, target_leaf_num: int):
    """
    Perform policy tree-based segmentation and OLS-based estimation per segment.

    Parameters:
        pop: PopulationSimulator object with simulated data
        depth: maximum depth of the policy tree
        random_state: optional random seed for reproducibility
    """
    x_mat = np.array([cust.x for cust in pop.train_customers])
    D_vec = np.array([cust.D_i for cust in pop.train_customers]).reshape(-1,1)
    y_vec = np.array([cust.y for cust in pop.train_customers]).reshape(-1, 1)
    Gamma = np.array(pop.gamma[[cust.customer_id for cust in pop.train_customers]])  # shape (N, 2)
    
    with localconverter(default_converter + numpy2ri.converter):
        X_r = ro.conversion.py2rpy(np.asarray(x_mat))
        Gamma_r = ro.conversion.py2rpy(Gamma)
    
    ro.globalenv['X'] = X_r
    ro.globalenv['Gamma'] = Gamma_r
    ro.r(f'tree <- policy_tree(X, Gamma, depth={depth})') 
    
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

    action_ids_raw = list(ro.conversion.rpy2py(action_r))  # (1 = action 0, 2 = action 1 by default)
    action_ids = np.array([a - 1 for a in action_ids_raw]) # Normalize segment and action labels to start from 0
    
    segment_labels_pruned, action_ids_pruned, segment_action_map, leaf_to_pruned_segment = post_prune_tree(
        segment_labels=np.array(segment_labels_raw), # ✅ raw IDs
        action_ids=np.array(action_ids),
        Gamma=Gamma,
        target_leaf_num=target_leaf_num,
        leaf_to_parent_map=leaf_to_parent_map
    )

    # Assign each customer to estimated segment
    estimate_segment_and_assign(pop, target_leaf_num, segment_labels_pruned, x_mat, D_vec, y_vec, action_ids_pruned)
    
    assign_val_customers_to_pruned_tree(tree, pop, leaf_to_pruned_segment)
    val_score = evaluate_policy_tree_on_validation(pop)
    
    # plot_segment_sankey(segment_labels, segment_labels_pruned)

    return val_score


def evaluate_policy_tree_on_validation(pop: PopulationSimulator):
    """
    Assign validation customers to policy tree segments and compute DR-based policy value.
    """
    Gamma_val = pop.gamma[[cust.customer_id for cust in pop.val_customers]]

    actions = np.array([
        cust.est_segment["policy_tree"].est_action for cust in pop.val_customers
    ])

    V = (1 - actions) * Gamma_val[:, 0] + actions * Gamma_val[:, 1]
    return V.mean()


def estimate_segment_and_assign(pop: PopulationSimulator, target_leaf_num, segment_labels, x_mat, D_vec, y_vec, action_ids):
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

        est_alpha, est_beta, est_tau, est_action, _ = estimate_segment_parameters(x_m, D_m, y_m)
        
        est_action = action_ids[idx_m[0]]
        est_seg = SegmentEstimate(est_alpha, est_beta, est_tau, est_action, len(idx_m), segment_id=m)
        pop.est_segments_list["policy_tree"].append(est_seg)
    
    assign_trained_customers_to_segments(pop, segment_labels, "policy_tree")

def post_prune_tree(segment_labels, action_ids, Gamma, target_leaf_num, leaf_to_parent_map):
    """
    Prune only sibling leaf segments (same parent in tree structure).

    Parameters:
        segment_labels: np.ndarray of segment IDs (raw R leaf IDs)
        action_ids: np.ndarray of actions per sample
        Gamma: np.ndarray (n x 2), reward matrix
        target_leaf_num: desired number of leaf segments
        leaf_to_parent_map: dict {leaf_id → parent_id}, from R

    Returns:
        pruned_segment_labels, pruned_action_ids, segment_action_map
    """
    # Build initial segment-to-sample index map
    segment_map = {
        s: np.array(np.where(segment_labels == s)[0], dtype=int)
        for s in sorted(set(segment_labels))
    }

    # Track original leaf IDs in each segment
    segment_to_leaves = {s: {s} for s in segment_map}
    action_map = {s: action_ids[indices[0]] for s, indices in segment_map.items()}

    def seg_welfare(indices, action):
        return Gamma[indices, int(action)].mean()

    while len(segment_map) > target_leaf_num:
        best_pair = None
        best_action = None
        min_welfare_loss = float("inf")

        segments = list(segment_map.keys())
        for s1, s2 in combinations(segments, 2):
            # ✅ Only merge if all leaves have the same parent
            combined_leaves = segment_to_leaves[s1] | segment_to_leaves[s2]
            combined_parents = {leaf_to_parent_map[leaf] for leaf in combined_leaves}

            if len(combined_parents) > 1:
                continue  # Not siblings — illegal merge

            # Evaluate welfare of merged segment
            idx1, idx2 = segment_map[s1], segment_map[s2]
            merged_idx = np.concatenate([idx1, idx2])

            mean_rewards = [Gamma[merged_idx, a].mean() for a in range(Gamma.shape[1])]
            best_a = int(np.argmax(mean_rewards))
            merged_welfare = mean_rewards[best_a]

            # Original welfare (weighted avg)
            w1 = seg_welfare(idx1, action_map[s1])
            w2 = seg_welfare(idx2, action_map[s2])
            n1, n2 = len(idx1), len(idx2)
            original_total = (n1 * w1 + n2 * w2) / (n1 + n2)

            loss = original_total - merged_welfare
            if loss < min_welfare_loss:
                best_pair = (s1, s2)
                best_action = best_a
                min_welfare_loss = loss

        s1, s2 = best_pair
        new_seg_id = min(s1, s2)
        merged_indices = np.concatenate([segment_map[s1], segment_map[s2]])

        segment_map[new_seg_id] = merged_indices
        action_map[new_seg_id] = best_action
        segment_to_leaves[new_seg_id] = segment_to_leaves[s1] | segment_to_leaves[s2]

        # Remove the other (now merged) segment
        del segment_map[s1 if s1 != new_seg_id else s2]
        del action_map[s1 if s1 != new_seg_id else s2]
        del segment_to_leaves[s1 if s1 != new_seg_id else s2]

    # Reindex to 0-based contiguous segments
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

    # print seg_id_map
    leaf_to_pruned_segment = {}
    for seg_id, leaf_set in segment_to_leaves.items():  # segment_to_leaves: original leaves in each segment
        for leaf in leaf_set:
            leaf_to_pruned_segment[leaf] = seg_id_map[seg_id]
            
    return pruned_segment_labels, pruned_action_ids, segment_action_map, leaf_to_pruned_segment

def plot_policy_tree(depth):
    svg_filename=f'figures/policy_tree_depth_{depth}.svg'
    svg_string = ro.r('DiagrammeRsvg::export_svg(plot(tree))')[0]
    
    with open(svg_filename, "w") as f:
        f.write(svg_string)
    display(SVG(filename=svg_filename))
    
def assign_val_customers_to_pruned_tree(tree, pop, pruned_segment_map):
    """
    Assign each validation customer to a policy tree segment (based on pruned structure).
    """
    x_mat_val = np.array([cust.x for cust in pop.val_customers])

    with localconverter(default_converter + numpy2ri.converter):
        X_val_r = ro.conversion.py2rpy(x_mat_val)

    # Predict raw segment (leaf ID) for each val customer
    segment_r = policytree.predict_policy_tree(tree, X_val_r, type="node.id")
    segment_ids = list(ro.conversion.rpy2py(segment_r))

    for cust, raw_leaf in zip(pop.val_customers, segment_ids):
        pruned_seg = pruned_segment_map[raw_leaf]  # maps raw leaf ID → pruned segment index
        segment_obj = pop.est_segments_list["policy_tree"][pruned_seg]

        # Attach the full SegmentEstimate object
        cust.est_segment["policy_tree"] = segment_obj


