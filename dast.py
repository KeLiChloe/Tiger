"""
DAST (Doubly-Robust Algorithm for Segmentation Trees)

A decision tree algorithm for customer segmentation that maximizes implementation profit
using doubly-robust value estimation with a variance-based fallback criterion.
"""

import numpy as np
from ground_truth import SegmentEstimate, PopulationSimulator
from utils import estimate_segment_parameters, evaluate_on_validation, compute_node_DR_value


class DASTNode:
    """Node in a DAST tree."""
    
    def __init__(self, indices, depth=0):
        self.indices = indices
        self.depth = depth
        self.value = None
        
        # Split info
        self.split_feature = None
        self.split_threshold = None
        self.left = None
        self.right = None
        self.is_leaf = True
        
        # Segment info (set after pruning)
        self.segment_id = None
    
    def prune(self):
        """Convert internal node back to leaf."""
        self.left = None
        self.right = None
        self.is_leaf = True


class DASTree:
    """DAST decision tree for customer segmentation."""
    
    def __init__(self, x, y, D, gamma, candidate_thresholds, min_leaf_size, max_depth, algo, use_hybrid_method, action_num=None):
        self.x = x
        self.y = y
        self.D = D
        self.gamma = gamma
        self.H = candidate_thresholds
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.algo = algo
        self.action_num = action_num

        self.root = None
        self.leaf_nodes = []
        self.debug = False
        
        # Tolerance values for floating point comparisons
        self.tolerance_pruning = 1e-5  # For pruning gain comparisons
        self.use_hybrid_method = use_hybrid_method

    def build(self, debug=False):
        """Build the full tree by recursively growing nodes."""
        self.debug = debug
        all_indices = np.arange(self.x.shape[0])
        
        if debug:
            print(f"\n{'='*60}")
            print(f"🌱 Building DAST Tree (N={len(all_indices)}, max_depth={self.max_depth})")
            print(f"{'='*60}")
        
        self.root = self._grow_node(all_indices, depth=0)
        
        if debug:
            print(f"\n✅ Tree built! Total leaves: {len(self._get_leaf_nodes())}")
    
    def prune_to_segments_and_estimate(self, M, data, customers, include_interactions, debug=False):
        """
        Prune tree to M leaves, estimate segment parameters, and assign customers.
        Returns list of SegmentEstimate objects.
        """
        current_leaves = self._get_leaf_nodes()

        if debug:
            print(f"   Initial leaves: {len(current_leaves)}, Target M: {M}")

        # Iteratively prune least valuable splits
        # if len(current_leaves) < M:
        #     raise ValueError(f"Current leaves ({len(current_leaves)}) are already less than or equal to M ({M}). No pruning needed.")
        
        while len(current_leaves) > M:
            prunable_nodes = self._get_internal_nodes_with_leaf_children()
            
            # Compute pruning gains
            gains = [(node, self._compute_pruning_gain(node)) for node in prunable_nodes]
            
            # Initialize max_gain properly (like MST)
            if not gains:
                break  # No prunable nodes, stop
            max_gain = max(g for _, g in gains)
            candidates = [node for node, g in gains if abs(g - max_gain) < self.tolerance_pruning]
            
            # Tie-breaking: if multiple nodes have same gain, use X variance
            if len(candidates) > 1:
                # Select node that minimizes within-cluster X variance increase
                best_node = min(candidates, key=self._compute_variance_after_pruning)
                if debug:
                    print(f"   Pruning node (gain={max_gain:.4f}, tie-break by X variance), leaves: {len(current_leaves)} → {len(current_leaves)-1}")
            else:
                best_node = candidates[0]
                if debug:
                    print(f"   Pruning node (gain={max_gain:.4f}), leaves: {len(current_leaves)} → {len(current_leaves)-1}")
            
            best_node.prune()
            current_leaves = self._get_leaf_nodes()

        self.leaf_nodes = current_leaves
        
        # Assign segment IDs
        for i, node in enumerate(self.leaf_nodes):
            node.segment_id = i

        # Estimate parameters for each segment
        segments = [
            self._fit_segment_and_assign(customers, node.indices, data, node.segment_id, include_interactions)
            for node in self.leaf_nodes
        ]
        
        if debug:
            print(f"   Final leaves: {len(self.leaf_nodes)}")
            print(f"   Segment details:")
            for i, (seg, node) in enumerate(zip(segments, self.leaf_nodes)):
                print(f"     Seg{i}: N={len(node.indices)}, tau={seg.est_tau:+7.2f}, action={seg.est_action}")
        
        return segments

    def predict_segment(self, customers, segment_dict):
        """Assign segment to each customer by traversing the tree."""
        for cust in customers:
            node = self.root
            while not node.is_leaf:
                if cust.x[node.split_feature] <= node.split_threshold:
                    node = node.left
                else:
                    node = node.right
            cust.est_segment[self.algo] = segment_dict[node.segment_id]

    # ============================================================
    # Growing: Build tree recursively
    # ============================================================

    def _grow_node(self, indices, depth):
        """Recursively grow tree by finding best split at each node."""
        node = DASTNode(indices, depth)
        node.value = compute_node_DR_value(self.y, self.D, self.gamma, indices, use_hybrid_method=self.use_hybrid_method, action_num=self.action_num)

        if self.debug:
            self._debug_print_node_info(node, indices)
        
        # Stop if max depth reached
        if depth == self.max_depth:
            if self.debug:
                print(f"{'  '*depth}  ⛔ Max depth reached → Leaf")
            self.leaf_nodes.append(node)
            return node
        
        # Evaluate all candidate splits
        gain_splits, var_splits = self._evaluate_all_splits(node, indices)
        
        final_split, criterion_used = self._select_best_split(gain_splits, var_splits)
        
        if self.debug:
            self._debug_print_split_info(depth, gain_splits, var_splits, final_split, criterion_used, node.value)
        
        # No valid split found -> leaf
        if final_split is None:
            self.leaf_nodes.append(node)
            return node
        
        # Execute split
        node.is_leaf = False
        node.split_feature, node.split_threshold, left_idx, right_idx = final_split
        node.left = self._grow_node(left_idx, depth + 1)
        node.right = self._grow_node(right_idx, depth + 1)
        
        return node
    
    def _evaluate_all_splits(self, node, indices):
        """
        Evaluate all candidate splits and return two lists:
        - gain_splits: [(gain, feature, threshold, left_idx, right_idx), ...]
        - var_splits: [(var_reduction, feature, threshold, left_idx, right_idx), ...]
        """
        gain_splits = []
        var_splits = []

        for j in range(self.x.shape[1]):
            for t in self.H[j]:
                left_idx = indices[self.x[indices, j] <= t]
                right_idx = indices[self.x[indices, j] > t]

                # Check minimum leaf size constraints
                if not (self._check_leaf_constraints(left_idx) and self._check_leaf_constraints(right_idx)):
                    continue
                
                # Criterion 1: Gain (profit improvement)
                left_val = compute_node_DR_value(self.y, self.D, self.gamma, left_idx, use_hybrid_method=self.use_hybrid_method, action_num=self.action_num)
                right_val = compute_node_DR_value(self.y, self.D, self.gamma, right_idx, use_hybrid_method=self.use_hybrid_method, action_num=self.action_num)
                gain = left_val + right_val - node.value
                
                # DEBUG: Print splits around x=11

                gain_splits.append((gain, j, t, left_idx, right_idx))
                
                # Criterion 2: Variance reduction (covariate homogeneity)
                var_reduction = self._compute_variance_reduction(indices, left_idx, right_idx)
                
                
                var_splits.append((var_reduction, j, t, left_idx, right_idx))
        
        return gain_splits, var_splits
    
    def _select_best_split(self, gain_splits, var_splits):
        """
        Select best split 
        
        Tie-breaking: When multiple splits have the same maximum gain (tie),
        use variance reduction to break the tie.
        
        Returns: (split_tuple, criterion_name) or (None, None)
        """
        if not gain_splits:
            return None, None
        
        # Find maximum gain
        max_gain = max(gain_splits, key=lambda x: x[0])[0]
        
        # Find all splits with maximum gain (tie detection)
        tied_indices = [idx for idx, (g, _, _, _, _) in enumerate(gain_splits) 
                        if abs(g - max_gain) <= 1e-6]
        
        if len(tied_indices) == 1:
            # No tie: use the only candidate directly
            _, feature, threshold, left_idx, right_idx = gain_splits[tied_indices[0]]
            return (feature, threshold, left_idx, right_idx), "gain"
        else:
            # Tie: use variance reduction to break the tie
            # ONLY select from the tied candidates (those with same max gain)
            # Since gain_splits and var_splits are generated in the same order,
            # we can use indices to find corresponding variance reductions
            tied_with_var = [(var_splits[idx][0], gain_splits[idx][1], gain_splits[idx][2], 
                                gain_splits[idx][3], gain_splits[idx][4], idx)
                            for idx in tied_indices if idx < len(var_splits)]
            
            if tied_with_var:
                # Find maximum variance reduction WITHIN the tied candidates
                max_var = max(tied_with_var, key=lambda x: x[0])[0]
                
                # Get all tied candidates with maximum variance
                best_var_candidates = [item for item in tied_with_var 
                                        if abs(item[0] - max_var) <= 1e-6]
                
                # If still tied, use stable tie-breaking (threshold, feature)
                if len(best_var_candidates) > 1:
                    best_var_candidates_sorted = sorted(best_var_candidates, 
                                                        key=lambda x: (x[2], x[1]))  # threshold, then feature
                    best_split = best_var_candidates_sorted[0]
                else:
                    best_split = best_var_candidates[0]
                
                _, feature, threshold, left_idx, right_idx, _ = best_split
                return (feature, threshold, left_idx, right_idx), "gain-variance"
    
        
        return None, None
    
    def _select_with_tie_breaking(self, candidates, key_func, reverse=False, tolerance=None):
        """
        Select best candidate with stable tie-breaking.
        
        Args:
            candidates: List of candidate tuples
            key_func: Function to extract comparison key from candidate
            reverse: If True, select maximum; if False, select minimum
            tolerance: Tolerance for tie detection (default: 1e-6)
        
        Returns:
            Best candidate tuple
        """
        if tolerance is None:
            tolerance = 1e-6
        
        if not candidates:
            return None
        
        # Find best value
        best_val = max(candidates, key=key_func) if reverse else min(candidates, key=key_func)
        best_val = key_func(best_val)
        
        # Find all candidates with best value (within tolerance)
        tied_candidates = [c for c in candidates 
                          if abs(key_func(c) - best_val) <= tolerance]
        
        # Stable tie-breaking: sort by (threshold, feature) for splits
        # Assumes format: (value, feature, threshold, ...)
        if len(tied_candidates) > 1 and len(tied_candidates[0]) >= 3:
            tied_candidates_sorted = sorted(tied_candidates, key=lambda x: (x[2], x[1]))  # threshold, then feature
            return tied_candidates_sorted[0]
        else:
            return tied_candidates[0]
    
    def _compute_variance_reduction(self, parent_idx, left_idx, right_idx):
        """
        Compute reduction in covariate variance from splitting.
        Higher variance = more heterogeneous customers = potentially different tau.
        """
        parent_var = self._compute_covariate_variance(parent_idx)
        left_var = self._compute_covariate_variance(left_idx)
        right_var = self._compute_covariate_variance(right_idx)
        
        n_parent = len(parent_idx)
        if n_parent == 0:
            return 0.0
        
        # Weighted average of child variances
        weight_left = len(left_idx) / n_parent
        weight_right = len(right_idx) / n_parent
        weighted_child_var = weight_left * left_var + weight_right * right_var
        
        return parent_var - weighted_child_var
    
    def _compute_covariate_variance(self, indices):
        """Compute total variance of covariates as heterogeneity measure."""
        if len(indices) < 2:
            return 0.0
        return np.var(self.x[indices], axis=0, ddof=1).sum()
    
    def _check_leaf_constraints(self, indices):
        """Check if leaf has minimum required samples for ALL possible actions."""
        D_sub = self.D[indices]
        
        # STRICT: All actions 0..action_num-1 must each have at least min_leaf_size samples
        for a in range(self.action_num):
            if np.sum(D_sub == a) < self.min_leaf_size:
                return False
        
        return True

    # ============================================================
    # Pruning: Select M most valuable splits
    # ============================================================

    def _compute_pruning_gain(self, node):
        """
        Compute gain from pruning node (higher = less valuable split).
        Pruning gain = parent_value - (left_value + right_value)
        """
        if node.left is None or node.right is None:
            return np.inf
        return node.value - (node.left.value + node.right.value)
    
    def _compute_variance_after_pruning(self, node):
        """
        Compute within-cluster X variance increase if this node is pruned.
        
        Used for tie-breaking when multiple nodes have same pruning gain.
        Lower variance increase = better (more homogeneous clusters).
        
        Returns:
            float: Variance increase from pruning
        """
        if node.left is None or node.right is None:
            return np.inf
        
        # Get indices for merged cluster (after pruning) and child clusters (before pruning)
        merged_indices = node.indices
        left_indices = node.left.indices
        right_indices = node.right.indices
        
        # Compute variances using helper method
        merged_var = self._compute_covariate_variance(merged_indices)
        left_var = self._compute_covariate_variance(left_indices)
        right_var = self._compute_covariate_variance(right_indices)
        
        # Weighted average variance before pruning
        n_total = len(merged_indices)
        if n_total == 0:
            return np.inf
        n_left = len(left_indices)
        n_right = len(right_indices)
        weighted_var_before = (n_left * left_var + n_right * right_var) / n_total
        
        # Variance increase from pruning
        return merged_var - weighted_var_before

    # ============================================================
    # Parameter Estimation
    # ============================================================

    def _fit_segment_and_assign(self, customers, indices, data, segment_id, include_interactions):
        """Estimate segment parameters and assign to customers."""
        X_seg = data['X'][indices]
        D_seg = data['D'][indices]
        Y_seg = data['Y'][indices]
        
        est_tau, est_action = estimate_segment_parameters(X_seg, D_seg, Y_seg, include_interactions, self.action_num)
        segment = SegmentEstimate(est_tau, est_action, segment_id)
        
        for i in indices:
            customers[i].est_segment[self.algo] = segment
        
        return segment

    # ============================================================
    # Tree Traversal Utilities
    # ============================================================
    
    def _get_leaf_nodes(self):
        """Return all leaf nodes in tree."""
        return self._gather_nodes(self.root, lambda n: n.is_leaf)
    
    def _get_internal_nodes_with_leaf_children(self):
        """Return internal nodes where both children are leaves."""
        def condition(n):
            return not n.is_leaf and n.left and n.right and n.left.is_leaf and n.right.is_leaf
        return self._gather_nodes(self.root, condition)
    
    def _gather_nodes(self, node, condition):
        """Recursively collect nodes matching condition."""
        if node is None:
            return []
        
        result = [node] if condition(node) else []
        result += self._gather_nodes(node.left, condition)
        result += self._gather_nodes(node.right, condition)
        
        return result

    # ============================================================
    # Debug Utilities
    # ============================================================
    
    def _estimate_tau(self, indices):
        """Estimate tau (treatment effect) for given indices."""
        if len(indices) == 0:
            return None
        D_sub = self.D[indices]
        Y_sub = self.y[indices]
        unique_actions = np.unique(D_sub)
        
        if len(unique_actions) < 2:
            return None
        
        # Compute mean outcome for each action
        action_means = {}
        for action in unique_actions:
            Y_a = Y_sub[D_sub == action]
            if len(Y_a) > 0:
                action_means[int(action)] = np.mean(Y_a)
        
        if len(action_means) < 2:
            return None
        
        best_action = max(action_means, key=action_means.get)
        baseline_action = 0 if 0 in action_means else min(action_means.keys())
        return action_means[best_action] - action_means[baseline_action]
    
    def _debug_print_node_info(self, node, indices):
        """Print node information for debugging."""
        depth = node.depth
        D_node = self.D[indices]
        Y_node = self.y[indices]
        unique_actions = np.unique(D_node)
        
        if len(unique_actions) >= 2:
            # Compute mean outcome for each action
            action_means = {}
            for action in unique_actions:
                Y_a = Y_node[D_node == action]
                if len(Y_a) > 0:
                    action_means[int(action)] = np.mean(Y_a)
            
            if len(action_means) >= 2:
                best_action = max(action_means, key=action_means.get)
                baseline_action = 0 if 0 in action_means else min(action_means.keys())
                tau_hat = action_means[best_action] - action_means[baseline_action]
                print(f"\n{'  '*depth}📍 Depth {depth}: N={len(indices)}, Value={node.value:.4f}, "
                      f"tau_hat={tau_hat:.4f}, best_action={best_action}")
            else:
                print(f"\n{'  '*depth}📍 Depth {depth}: N={len(indices)}, Value={node.value:.4f}, "
                      f"[insufficient data for multiple actions]")
        else:
            print(f"\n{'  '*depth}📍 Depth {depth}: N={len(indices)}, Value={node.value:.4f}, "
                  f"[insufficient data]")
    
    def _debug_print_split_info(self, depth, gain_splits, var_splits, final_split, criterion, parent_value):
        """Print split evaluation information for debugging."""
        indent = '  ' * depth
        total_candidates = len(self.H[0]) * self.x.shape[1]
        
        print(f"{indent}  🔍 Total candidate splits: {total_candidates}")
        print(f"{indent}  ✓ Valid splits (pass min_leaf_size): {len(gain_splits)}")
        
        if not gain_splits:
            print(f"{indent}  ❌ No valid splits (all violate min_leaf_size={self.min_leaf_size})")
            return
        
        # Print gain criterion with detailed child node info
        print(f"{indent}  📌 Parent DR_value: {parent_value:.4f}")
        best_gain = max(gain_splits, key=lambda x: x[0])[0]
        print(f"{indent}  📊 Gain criterion: best_gain={best_gain:.4f}")
        top_gains = sorted(gain_splits, key=lambda x: x[0], reverse=True)[:5]
        for rank, (g, j, t, left_idx, right_idx) in enumerate(top_gains, 1):
            # Calculate tau_hat and DR values for left and right children
            left_tau = self._estimate_tau(left_idx)
            right_tau = self._estimate_tau(right_idx)
            left_action = int(left_tau >= 0) if left_tau is not None else "?"
            right_action = int(right_tau >= 0) if right_tau is not None else "?"
            
            # Format tau for display
            left_tau_str = f"{left_tau:.4f}" if left_tau is not None else "N/A"
            right_tau_str = f"{right_tau:.4f}" if right_tau is not None else "N/A"
            
            # Calculate actual DR values (the components of gain)
            left_val = compute_node_DR_value(self.y, self.D, self.gamma, left_idx, use_hybrid_method=self.use_hybrid_method, action_num=self.action_num)
            right_val = compute_node_DR_value(self.y, self.D, self.gamma, right_idx, use_hybrid_method=self.use_hybrid_method, action_num=self.action_num)
            recalc_gain = left_val + right_val - parent_value
            
            print(f"{indent}     #{rank}: gain_stored={g:.4f}, gain_recalc={recalc_gain:.4f}, feature={j}, threshold={t:.4f}")
            print(f"{indent}         Left(N={len(left_idx)}): tau={left_tau_str}, action={left_action}, DR_val={left_val:.4f}")
            print(f"{indent}         Right(N={len(right_idx)}): tau={right_tau_str}, action={right_action}, DR_val={right_val:.4f}")
        
        # Print variance criterion
        if var_splits:
            best_var = max(var_splits, key=lambda x: x[0])[0]
            print(f"{indent}  📊 Variance criterion: best_var_reduction={best_var:.4f}")
            top_vars = sorted(var_splits, key=lambda x: x[0], reverse=True)[:3]
            for rank, (v, j, t, _, _) in enumerate(top_vars, 1):
                print(f"{indent}     #{rank}: var_reduction={v:.4f}, feature={j}, threshold={t:.4f}")
        
        # Print decision
        if final_split is None:
            best_gain = max(gain_splits, key=lambda x: x[0])[0] if gain_splits else -np.inf
            best_var = max(var_splits, key=lambda x: x[0])[0] if var_splits else -np.inf
            reason = "No valid splits" if not gain_splits else f"Both criteria failed (gain={best_gain:.4f}, var={best_var:.4f})"
            print(f"{indent}  🍃 → Leaf ({reason})")
        else:
            feature, threshold, left_idx, right_idx = final_split
            criterion_map = {
                "gain": "GAIN-based",
                "gain-variance": "GAIN-based (tie broken by variance)",
                "variance": "VARIANCE-based (fallback)"
            }
            criterion_name = criterion_map.get(criterion, criterion.upper())
            print(f"{indent}  ✅ Using {criterion_name} split")
            print(f"{indent}  ✂️  Split on X[{feature}] <= {threshold:.4f} (criterion={criterion})")
            print(f"{indent}     Left: {len(left_idx)}, Right: {len(right_idx)}")


# ============================================================
# Main Interface Function
# ============================================================

def DAST_segment_and_estimate(pop: PopulationSimulator, n_segments, max_depth, 
                               min_leaf_size,  algo, include_interactions, use_hybrid_method, debug=False):
    """
    Main interface for DAST algorithm.
    
    Returns:
        tree: Trained DASTree object
        val_score: Validation score (average DR value)
        segment_dict: Dictionary mapping segment_id to SegmentEstimate
    """
    # Prepare training data
    data_train = {
        "X": np.array([cust.x for cust in pop.train_customers]),
        "D": np.array([cust.D_i for cust in pop.train_customers]).reshape(-1, 1),
        "Y": np.array([cust.y for cust in pop.train_customers]).reshape(-1, 1),
        "Gamma": pop.gamma_train
    }
    
    # Generate candidate thresholds using quantile-based binning
    # Use B bins to reduce computational cost
    B = 200  # Number of bins 
    H = {}
    for j in range(data_train["X"].shape[1]):
        sorted_values = np.sort(np.unique(data_train["X"][:, j]))
        N_unique = len(sorted_values)
        
        if N_unique <= 1:
            H[j] = sorted_values
        elif N_unique <= B:
            H[j] = (sorted_values[:-1] + sorted_values[1:]) / 2.0
        else:
            quantile_indices = [int(np.floor(k / B * N_unique)) for k in range(1, B)]
            quantile_indices = sorted(set([min(idx, N_unique - 1) for idx in quantile_indices]))
            quantile_values = sorted_values[quantile_indices]
            H[j] = (quantile_values[:-1] + quantile_values[1:]) / 2.0
    
    
    # Build tree
    tree = DASTree(
        x=data_train["X"],
        y=data_train["Y"],
        D=data_train["D"],
        gamma=data_train["Gamma"],
        candidate_thresholds=H,
        min_leaf_size=min_leaf_size,
        max_depth=max_depth,
        algo=algo,
        use_hybrid_method=use_hybrid_method,
        action_num=pop.action_num
    )
    tree.build(debug=debug)
    
    # Prune and estimate
    if debug:
        print(f"\n🔪 Pruning to M={n_segments} segments...")
    
    pruned_segments = tree.prune_to_segments_and_estimate(n_segments, data_train, pop.train_customers, include_interactions, debug=debug)
    pop.est_segments_list[algo] = pruned_segments
    segment_dict = {seg.segment_id: seg for seg in pruned_segments}
    
    # Validation
    val_score = None
    if len(pop.val_customers) > 0:
        tree.predict_segment(pop.val_customers, segment_dict)
        Gamma_val = pop.gamma_val
        val_score = evaluate_on_validation(pop, algo=algo, Gamma_val=Gamma_val)
        if debug:
            print(f"✅ Validation score: {val_score:.4f}")
        
    return tree, val_score, segment_dict
