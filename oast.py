# oast_tree.py

import numpy as np
from ground_truth import SegmentEstimate, PopulationSimulator
from utils import estimate_segment_parameters



class OASTNode:
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

class OASTree:
    def __init__(self, x, y, D, gamma, candidate_thresholds, min_leaf_size=5, epsilon=1e-4, max_depth=3):
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
        self.segment_dict = None


    def build(self):
        N = self.x.shape[0]
        all_indices = np.arange(N)
        self.root = self._grow_node(all_indices, depth=0)
        self._assign_segment_ids()
    
    def prune_to_segments_and_estimate(self, M, data, customers):
        """
        Prune the tree to have exactly M leaf nodes.
        Then fit OLS and assign SegmentEstimate to customers.
        """
        current_leaves = self.get_leaf_nodes()

        while len(current_leaves) > M:
            min_gain = np.inf
            candidate_node = None
            for node in self.get_internal_nodes_with_leaf_children():
                gain = self._compute_pruning_gain(node, data['Gamma'], data['D'], data['Y'])
                if gain < min_gain:
                    min_gain = gain
                    candidate_node = node
            if candidate_node:
                candidate_node.prune()
            current_leaves = self.get_leaf_nodes()

        self.leaf_nodes = current_leaves
        self._assign_segment_ids()

        final_segments = []
        for seg_id, node in enumerate(current_leaves):
            segment = self._fit_segment_and_assign(customers, node.indices, data, seg_id)
            final_segments.append(segment)
        return final_segments

    def compute_total_DR_value(self):
        """
        Compute the total DR value across all current leaf nodes (weighted average).
        Returns:
            total_value: float
        """
        leaves = self.get_leaf_nodes()
        total_n = sum(len(node.indices) for node in leaves)

        weighted_value = sum(
            len(node.indices) * node.value
            for node in leaves
            if node.value is not None and np.isfinite(node.value)
        )
        return weighted_value / total_n

    def predict_segment(self, customers):
        """
        Predicts and assigns segment IDs for a list of Customer objects based on their covariates.

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
            seg_id = node.segment_id
            segment_obj = self.segment_dict[seg_id]
            cust.est_segment["oast"] = segment_obj
        
    def _grow_node(self, indices, depth):
        node = OASTNode(indices, depth)
        node.value, node.tau_hat = self._compute_node_value_and_tau(indices)

        if depth == self.max_depth:
            self.leaf_nodes.append(node)
            return node

        best_gain = -np.inf
        best_split = None

        for j in range(self.x.shape[1]):
            for t in self.H[j]:
                left_idx = indices[self.x[indices, j] <= t]
                right_idx = indices[self.x[indices, j] > t]

                if self._check_leaf_constraints(left_idx) and self._check_leaf_constraints(right_idx):
                    left_val, _ = self._compute_node_value_and_tau(left_idx)
                    right_val, _ = self._compute_node_value_and_tau(right_idx)
                    gain = left_val + right_val - node.value

                    if gain > best_gain:
                        best_gain = gain
                        best_split = (j, t, left_idx, right_idx)

        if best_gain <= self.epsilon or best_split is None:
            self.leaf_nodes.append(node)
            return node

        # Make this node internal
        node.is_leaf = False
        node.split_feature, node.split_threshold, left_idx, right_idx = best_split
        node.left = self._grow_node(left_idx, depth + 1)
        node.right = self._grow_node(right_idx, depth + 1)

        return node

    def _compute_node_value_and_tau(self, indices):
        D_m = self.D[indices]
        Y_m = self.y[indices]
        gamma_0_m = self.gamma[indices, 0]
        gamma_1_m = self.gamma[indices, 1]

        n1 = np.sum(D_m == 1)
        n0 = np.sum(D_m == 0)

        if n1 == 0 or n0 == 0:
            raise ValueError("The number of customers of one action is 0!")

        y1 = np.mean(Y_m[D_m == 1])
        y0 = np.mean(Y_m[D_m == 0])
        tau_hat = y1 - y0

        a_i = int(tau_hat > 0)
        V_i = (1 - a_i) * gamma_0_m + a_i * gamma_1_m
        value = np.mean(V_i)
        return value, tau_hat

    def _check_leaf_constraints(self, indices):
        D_sub = self.D[indices]
        n1 = np.sum(D_sub == 1)
        n0 = np.sum(D_sub == 0)
        return (n1 >= self.min_leaf_size) and (n0 >= self.min_leaf_size)

    def _assign_segment_ids(self):
        for i, node in enumerate(self.leaf_nodes):
            node.segment_id = i
    
    def get_leaf_nodes(self):
        return self._gather_nodes(self.root, is_leaf=True)

    def get_internal_nodes_with_leaf_children(self):
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

    def _compute_pruning_gain(self, node, Gamma, D, Y):
        left, right = node.left, node.right
        if left is None or right is None:
            return np.inf
        gain = (
            self._compute_segment_value(left.indices, Gamma, D, Y) +
            self._compute_segment_value(right.indices, Gamma, D, Y) -
            self._compute_segment_value(node.indices, Gamma, D, Y)
        )
        return gain

    def _compute_segment_value(self, indices, Gamma, D, Y):
        if len(indices) == 0:
            return -np.inf
        D_seg = D[indices]
        Y_seg = Y[indices]
        n1 = np.sum(D_seg == 1)
        n0 = np.sum(D_seg == 0)
        if n1 == 0 or n0 == 0:
            return -np.inf
        tau_hat = Y_seg[D_seg == 1].mean() - Y_seg[D_seg == 0].mean()
        a_hat = int(tau_hat > 0)
        V_hat = (1 - a_hat) * Gamma[indices, 0] + a_hat * Gamma[indices, 1]
        return V_hat.mean()

    def _fit_segment_and_assign(self, customers, indices, data, seg_id):
        X_seg = data['X'][indices]
        D_seg = data['D'][indices]
        Y_seg = data['Y'][indices]
        est_alpha, est_beta, est_tau, est_action, _ = estimate_segment_parameters(X_seg, D_seg, Y_seg)
        segment = SegmentEstimate(est_alpha, est_beta, est_tau, est_action, len(indices), segment_id=seg_id)
        for i in indices:
            customers[i].est_segment["oast"] = segment
        return segment

def train_and_prune_and_estimate_oast_tree(train_customers, pop, n_segments, max_depth, min_leaf_size, epsilon, threshold_grid):
    X_train = np.array([cust.x for cust in train_customers])
    D_train = np.array([cust.D_i for cust in train_customers]).reshape(-1, 1)
    Y_train = np.array([cust.y for cust in train_customers]).reshape(-1, 1)
    Gamma_train = pop.gamma[[cust.customer_id for cust in train_customers]]

    data_train = {
        "X": X_train,
        "D": D_train,
        "Y": Y_train,
        "Gamma": Gamma_train
    }

    H = {
        j: np.linspace(X_train[:, j].min(), X_train[:, j].max(), threshold_grid)
        for j in range(X_train.shape[1])
    }

    tree = OASTree(
        x=X_train,
        y=Y_train,
        D=D_train,
        gamma=Gamma_train,
        candidate_thresholds=H,
        min_leaf_size=min_leaf_size,
        epsilon=epsilon,
        max_depth=max_depth
    )
    tree.build()
    pruned_segments = tree.prune_to_segments_and_estimate(n_segments, data_train, train_customers)
    tree.segment_dict = {seg.segment_id: seg for seg in pruned_segments}
    return tree, pruned_segments


def evaluate_oast_on_validation(tree, val_customers, pop):
    Gamma_val = pop.gamma[[cust.customer_id for cust in val_customers]]

    tree.predict_segment(val_customers)
    actions = np.array([
        cust.est_segment["oast"].est_action for cust in val_customers
    ])

    V = (1 - actions) * Gamma_val[:, 0] + actions * Gamma_val[:, 1]
    return V.mean()


def OAST_segment_and_estimate(pop: PopulationSimulator, n_segments, max_depth=3, min_leaf_size=10, epsilon=1e-4, threshold_grid=10):
    
    tree, pruned_segments = train_and_prune_and_estimate_oast_tree(
        train_customers=pop.train_customers,
        pop=pop,
        n_segments=n_segments,
        max_depth=max_depth,
        min_leaf_size=min_leaf_size,
        epsilon=epsilon,
        threshold_grid=threshold_grid
    )
    
    if (len(pruned_segments) != n_segments):
        raise ValueError(f"Expected {n_segments} segments, but got {len(pruned_segments)} after pruning.")
    
    pop.est_segments_list["oast"] = pruned_segments

    val_score = evaluate_oast_on_validation(tree, pop.val_customers, pop)

    return val_score