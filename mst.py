# mst_tree.py

import numpy as np
from ground_truth import SegmentEstimate, PopulationSimulator
from utils import estimate_segment_parameters, evaluate_on_validation, build_design_matrix
from sklearn.linear_model import LinearRegression

import random



def compute_residual_value(X, Y, D, indices, include_interactions):
    # Subset the data
    X_m = X[indices]
    D_m = D[indices].reshape(-1, 1)
    Y_m = Y[indices].reshape(-1, 1)

    # Construct the design matrix: [intercept | X | D]
    X_design = build_design_matrix(X_m, D_m, include_interactions)
    
    # Fit the linear model
    model = LinearRegression(fit_intercept=False).fit(X_design, Y_m)
    
    # Predict
    Y_pred = model.predict(X_design)

    # Compute residuals
    residuals = np.sum((Y_m - Y_pred) ** 2) + np.random.rand() * 1e-6  # small noise to avoid zero residuals

    # return residuals / len(indices)
    return residuals # corrected: return total residuals

class MSTNode:
    def __init__(self, indices, depth=0):
        self.indices = indices            # Indices of customers in this node
        self.depth = depth
        self.split_feature = None
        self.split_threshold = None
        self.left = None
        self.right = None
        self.is_leaf = True
        self.segment_id = None            # Set after tree construction
        self.tau_hat = None
        self.value = None                
    
    def prune(self):
        self.left = None
        self.right = None
        self.is_leaf = True

class MSTree:
    def __init__(self, x, y, D, candidate_thresholds, min_leaf_size, epsilon, max_depth, algo):
        self.x = x
        self.y = y
        self.D = D
        self.H = candidate_thresholds
        self.min_leaf_size = min_leaf_size
        self.epsilon = epsilon
        self.max_depth = max_depth

        self.root = None
        self.leaf_nodes = []
        
        self.algo = algo


    def build(self, include_interactions):
        N = self.x.shape[0]
        all_indices = np.arange(N)
        self.root = self._grow_node(all_indices, depth=0, include_interactions=include_interactions)
        
    
    def prune_to_segments_and_estimate(self, M, data, customers, include_interactions):
        """
        Prune the tree to have exactly M leaf nodes.
        Then fit OLS and assign SegmentEstimate to customers.
        """
        current_leaves = self._get_leaf_nodes()

        # if len(current_leaves) < M:
        #     raise ValueError(f"Current leaves ({len(current_leaves)}) are already less than or equal to M ({M}). No pruning needed.")
        
        while len(current_leaves) > M:
            max_gain = -np.inf
            candidate_node = None
            for node in self._get_internal_nodes_with_leaf_children():
                gain = self._compute_pruning_gain(node)
                if gain > max_gain:
                    max_gain = gain
                    candidate_node = node
            if candidate_node:
                candidate_node.prune()
            current_leaves = self._get_leaf_nodes()

        self.leaf_nodes = current_leaves
        for i, node in enumerate(self.leaf_nodes):
            node.segment_id = i

        final_segments = []
        for node in self.leaf_nodes:
            segment = self._fit_segment_and_assign(customers, node.indices, data, node.segment_id, include_interactions)
            final_segments.append(segment)
        return final_segments

    def predict_segment(self, customers, segment_dict):
        """
        Predicts and assigns segment IDs for a list of new Customer objects based on their covariates.

        Args:
            customers: list of Customer objects
        """
        for cust in customers:
            xi = cust.x
            node = self.root
            while not node.is_leaf:
                if xi[node.split_feature] <= node.split_threshold:
                    node = node.left
                else:
                    node = node.right

            # Assign SegmentEstimate or segment_id (if lookup not needed)
            segment_obj = segment_dict[node.segment_id]
            cust.est_segment[f"{self.algo}"] = segment_obj


    def _grow_node(self, indices, depth, include_interactions):
        node = MSTNode(indices, depth)
        node.value = compute_residual_value(self.x, self.y, self.D, indices, include_interactions)

        if depth == self.max_depth:
            self.leaf_nodes.append(node)
            return node
        best_gain = -np.inf
        best_split = None
        valid_splits = []

        for j in range(self.x.shape[1]):
            for t in self.H[j]:
                left_idx = indices[self.x[indices, j] <= t]
                right_idx = indices[self.x[indices, j] > t]

                if self._check_leaf_constraints(left_idx) and self._check_leaf_constraints(right_idx):
                    left_val = compute_residual_value(self.x, self.y, self.D, left_idx, include_interactions)
                    right_val = compute_residual_value(self.x, self.y, self.D, right_idx, include_interactions)
                    gain = node.value - (left_val + right_val)
                    valid_splits.append((gain, j, t, left_idx, right_idx))

                    if gain >= best_gain:
                        best_gain = gain
                        best_split = (j, t, left_idx, right_idx)

        # If no good split found, fallback to random valid split (if any)
        if best_split is None:
            if valid_splits:
                _, j_rand, t_rand, left_idx, right_idx = random.choice(valid_splits)
                node.is_leaf = False
                node.split_feature = j_rand
                node.split_threshold = t_rand
                node.left = self._grow_node(left_idx, depth + 1, include_interactions)
                node.right = self._grow_node(right_idx, depth + 1, include_interactions)
                return node
            else:
                self.leaf_nodes.append(node)
                return node

        # Use best split
        node.is_leaf = False
        node.split_feature, node.split_threshold, left_idx, right_idx = best_split
        node.left = self._grow_node(left_idx, depth + 1, include_interactions)
        node.right = self._grow_node(right_idx, depth + 1, include_interactions)
        return node

    
    def _check_leaf_constraints(self, indices):
        D_sub = self.D[indices]
        n1 = np.sum(D_sub == 1)
        n0 = np.sum(D_sub == 0)
        return (n1 >= self.min_leaf_size) and (n0 >= self.min_leaf_size)

    
    def _get_leaf_nodes(self):
        return self._gather_nodes(self.root, is_leaf=True)

    def _get_internal_nodes_with_leaf_children(self):
        def condition(node):
            return (
                not node.is_leaf and 
                node.left and node.right and 
                node.left.is_leaf and node.right.is_leaf
            )
        return self._gather_nodes(self.root, condition=condition)

    def _gather_nodes(self, node, is_leaf=False, condition=None):
        if node is None:
            return []
        if condition:
            return (
                [node] if condition(node) else []
            ) + self._gather_nodes(node.left, condition=condition) + self._gather_nodes(node.right, condition=condition)
        if is_leaf and node.is_leaf:
            return [node]
        if not is_leaf and not node.is_leaf:
            return [node]
        return self._gather_nodes(node.left, is_leaf=is_leaf) + self._gather_nodes(node.right, is_leaf=is_leaf)

    def _compute_pruning_gain(self, node):
        """
        Compute the gain of pruning (i.e., merging left and right back to parent).
        
        Pruning gain = parent_value - (left_value + right_value)
        
        For MST, value = residual (lower is better).
        If gain > 0, pruning increases residual (we lose fit quality).
        If gain < 0, keeping the split is better (reduces residual).
        
        We want to prune nodes with the highest gain (least valuable splits).
        """
        left, right = node.left, node.right
        if left is None or right is None:
            return np.inf
        
        # Use cached values instead of recomputing
        # All node values were computed during tree building
        gain = node.value - (left.value + right.value)
        
        return gain

    def _fit_segment_and_assign(self, customers, indices, data, segment_id, include_interactions):
        X_seg = data['X'][indices]
        D_seg = data['D'][indices]
        Y_seg = data['Y'][indices]
        if include_interactions:
            est_alpha, est_beta, est_tau, est_action, est_delta = estimate_segment_parameters(X_seg, D_seg, Y_seg, include_interactions)
            segment = SegmentEstimate(est_alpha, est_beta, est_tau, est_action, segment_id, est_delta=est_delta)
        else:
            est_alpha, est_beta, est_tau, est_action = estimate_segment_parameters(X_seg, D_seg, Y_seg, include_interactions)
            segment = SegmentEstimate(est_alpha, est_beta, est_tau, est_action, segment_id)
        for i in indices:
            customers[i].est_segment[f"{self.algo}"] = segment
        return segment



def MST_segment_and_estimate(pop: PopulationSimulator, n_segments, max_depth, min_leaf_size, epsilon, threshold_grid, algo, include_interactions):
    # Prepare training data
    data_train = {
        "X": np.array([cust.x for cust in pop.train_customers]),
        "D": np.array([cust.D_i for cust in pop.train_customers]).reshape(-1, 1),
        "Y": np.array([cust.y for cust in pop.train_customers]).reshape(-1, 1),
    }
    
    
    # Generate candidate thresholds using quantile-based binning
    # Use B bins to reduce computational cost
    B = threshold_grid if threshold_grid > 0 else 20  # Use threshold_grid as number of bins
    H = {}
    for j in range(data_train["X"].shape[1]):
        sorted_values = np.sort(np.unique(data_train["X"][:, j]))
        N_unique = len(sorted_values)
        
        if N_unique <= B:
            # Few unique values, use them directly as thresholds
            H[j] = sorted_values
        else:
            # Many unique values, use quantile-based binning
            # Define indices for k/B quantiles: l_k = floor(k/B * N_unique)
            quantile_indices = [int(np.floor(k / B * N_unique)) for k in range(1, B)]
            # Remove duplicates and ensure valid indices
            quantile_indices = sorted(set([min(idx, N_unique - 1) for idx in quantile_indices]))
            # Get threshold values at these quantiles
            H[j] = sorted_values[quantile_indices]
    
    # Build MST
    tree = MSTree(
        x=data_train["X"],
        y=data_train["Y"],
        D=data_train["D"],
        candidate_thresholds=H,
        min_leaf_size=min_leaf_size,
        epsilon=epsilon,
        max_depth=max_depth,
        algo=algo,
    )
    tree.build(include_interactions)
    
    # Prune to M leaves and estimate parameters
    pruned_segments = tree.prune_to_segments_and_estimate(n_segments, data_train, pop.train_customers, include_interactions)
    pop.est_segments_list[f"{algo}"] = pruned_segments
    segment_dict = {seg.segment_id: seg for seg in pruned_segments}
    
    
    # Assign each validate customer to estimated segment
    val_score = None
    if len(pop.val_customers) > 0:
        tree.predict_segment(pop.val_customers, segment_dict)
        Gamma_val = pop.gamma_val
        val_score = evaluate_on_validation(pop, algo=algo, Gamma_val=Gamma_val)
        
    return tree, val_score, segment_dict


