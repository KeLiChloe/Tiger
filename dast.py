# dast_tree.py

import numpy as np
from ground_truth import SegmentEstimate, PopulationSimulator
from utils import estimate_segment_parameters, evaluate_on_validation, compute_node_DR_value

import random

class DASTNode:
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
        self.value = None                # Average profit from DR score
    
    def prune(self):
        self.left = None
        self.right = None
        self.is_leaf = True

class DASTree:
    def __init__(self, x, y, D, gamma, candidate_thresholds, min_leaf_size, epsilon, max_depth):
        self.x = x
        self.y = y
        self.D = D
        self.gamma = gamma
        self.H = candidate_thresholds
        self.min_leaf_size = min_leaf_size
        self.epsilon = epsilon
        self.max_depth = max_depth

        self.root = None
        self.leaf_nodes = []


    def build(self):
        N = self.x.shape[0]
        all_indices = np.arange(N)
        self.root = self._grow_node(all_indices, depth=0)
        
    
    def prune_to_segments_and_estimate(self, M, data, customers):
        """
        Prune the tree to have exactly M leaf nodes.
        Then fit OLS and assign SegmentEstimate to customers.
        """
        current_leaves = self._get_leaf_nodes()

        if len(current_leaves) < M:
            raise ValueError(f"Current leaves ({len(current_leaves)}) are already less than or equal to M ({M}). No pruning needed.")
        
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
            segment = self._fit_segment_and_assign(customers, node.indices, data, node.segment_id)
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
            cust.est_segment["dast"] = segment_obj
        


    def _grow_node(self, indices, depth):
        node = DASTNode(indices, depth)
        node.value = compute_node_DR_value(self.y, self.D, self.gamma, indices, buff=True)

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
                    left_val = compute_node_DR_value(self.y, self.D, self.gamma, left_idx, buff=True)
                    right_val = compute_node_DR_value(self.y, self.D, self.gamma, right_idx, buff=True)
                    gain = left_val + right_val - node.value
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
                node.left = self._grow_node(left_idx, depth + 1)
                node.right = self._grow_node(right_idx, depth + 1)
                return node
            else:
                self.leaf_nodes.append(node)
                return node

        # Use best split
        node.is_leaf = False
        node.split_feature, node.split_threshold, left_idx, right_idx = best_split
        node.left = self._grow_node(left_idx, depth + 1)
        node.right = self._grow_node(right_idx, depth + 1)
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
        left, right = node.left, node.right
        if left is None or right is None:
            return np.inf
        gain = (
            compute_node_DR_value(self.y, self.D, self.gamma, node.indices, buff=True) -
            (compute_node_DR_value(self.y, self.D, self.gamma, left.indices, buff=True) +
            compute_node_DR_value(self.y, self.D, self.gamma, right.indices, buff=True))
            
        )
        return gain

    def _fit_segment_and_assign(self, customers, indices, data, segment_id):
        X_seg = data['X'][indices]
        D_seg = data['D'][indices]
        Y_seg = data['Y'][indices]
        est_alpha, est_beta, est_tau, est_action = estimate_segment_parameters(X_seg, D_seg, Y_seg)
        segment = SegmentEstimate(est_alpha, est_beta, est_tau, est_action, segment_id)
        for i in indices:
            customers[i].est_segment["dast"] = segment
        return segment


def DAST_segment_and_estimate(pop: PopulationSimulator, n_segments, max_depth, min_leaf_size, epsilon, threshold_grid):
    # Prepare training data
    data_train = {
        "X": np.array([cust.x for cust in pop.train_customers]),
        "D": np.array([cust.D_i for cust in pop.train_customers]).reshape(-1, 1),
        "Y": np.array([cust.y for cust in pop.train_customers]).reshape(-1, 1),
        "Gamma": pop.gamma[[cust.customer_id for cust in pop.train_customers]]
    }
    
    # candidate thresholds for each feature
    H = {
        j: np.linspace(data_train["X"][:, j].min()-2, data_train["X"][:, j].max()+2, threshold_grid)
        for j in range(data_train["X"].shape[1])
    }
    
    # Build DAST
    tree = DASTree(
        x=data_train["X"],
        y=data_train["Y"],
        D=data_train["D"],
        gamma=data_train["Gamma"],
        candidate_thresholds=H,
        min_leaf_size=min_leaf_size,
        epsilon=epsilon,
        max_depth=max_depth
    )
    tree.build()
    
    # Prune to M leaves and estimate parameters
    pruned_segments = tree.prune_to_segments_and_estimate(n_segments, data_train, pop.train_customers)
    pop.est_segments_list["dast"] = pruned_segments
    segment_dict = {seg.segment_id: seg for seg in pruned_segments}
    
    
    # Assign each validate customer to estimated segment
    val_score = None
    if len(pop.val_customers) > 0:
        tree.predict_segment(pop.val_customers, segment_dict)
        val_score = evaluate_on_validation(pop, algo="dast")
        
    return tree, val_score, segment_dict


