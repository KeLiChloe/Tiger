from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from ground_truth import PopulationSimulator, SegmentEstimate
from utils import assign_trained_customers_to_segments, estimate_segment_parameters
import numpy as np
from plot import plot_segmentation


def GMM_segment_and_estimate(pop: PopulationSimulator, n_segments: int, random_state=None, pair_up=True):
    """
    Perform GMM-based segmentation and OLS-based estimation per segment.

    Parameters:
        pop: PopulationSimulator object with all simulated data
        n_segments: number of segments to estimate
        random_state: optional random seed for reproducibility
    """

    # Prepare input data for GMM
    x_mat = np.array([cust.x for cust in pop.train_customers])
    D_vec = np.array([cust.D_i for cust in pop.train_customers]).reshape(-1, 1)
    y_vec = np.array([cust.y for cust in pop.train_customers]).reshape(-1, 1)
    gmm_input = np.hstack([x_mat, y_vec])  # shape (N, d+1)

    # Fit GMM
    
    
    if pair_up:
        gmm_double = GaussianMixture(n_components=2*n_segments, random_state=random_state)
        gmm_labels_double = gmm_double.fit_predict(gmm_input)
        # plot_segmentation(gmm_labels_double, df_train, algo="gmm_double", M=2*n_segments)
        
        gmm_labels, _ = pair_segments_by_parameters(x_mat, D_vec, y_vec, gmm_labels_double, n_meta_segments=n_segments)
        # plot_segmentation(gmm_labels, df_train, algo="gmm", M=n_segments)
        
        gmm = GaussianMixture(n_components=n_segments, random_state=random_state)
        gmm.fit_predict(gmm_input)
        bic = gmm.bic(gmm_input)
    else:
        gmm = GaussianMixture(n_components=n_segments, random_state=random_state)
        gmm_labels = gmm.fit_predict(gmm_input)
        bic = gmm.bic(gmm_input)
        
        

    
    # Assign each customer to estimated segment
    pop.est_segments_list["gmm"] = []  # Reset

    # Estimate triplets via OLS per segment
    for m in range(n_segments):
        idx_m = np.where(gmm_labels == m)[0]
        
        if len(idx_m) == 0:
            raise ValueError(f"No customers assigned to segment {m}. ")

        x_m = x_mat[idx_m]
        D_m = D_vec[idx_m]
        y_m = y_vec[idx_m]
        
        est_alpha, est_beta, est_tau, est_action, _ = estimate_segment_parameters(x_m, D_m, y_m)

        # Create estimated segment object
        est_seg = SegmentEstimate(est_alpha, est_beta, est_tau, est_action, len(idx_m), segment_id=m)
        pop.est_segments_list["gmm"].append(est_seg)

    # Link each customer to estimated segment
    assign_trained_customers_to_segments(pop, gmm_labels, "gmm")
    
    x_val = np.array([cust.x for cust in pop.val_customers])
    y_val = np.array([cust.y for cust in pop.val_customers]).reshape(-1, 1)
    val_input = np.hstack([x_val, y_val])
    val_labels = gmm.predict(val_input)
    
    # Assign validation customers
    for cust, seg_id in zip(pop.val_customers, val_labels):
        segment = pop.est_segments_list["gmm"][seg_id]
        cust.est_segment["gmm"] = segment
        
    # get est segment ids for all customers
    df = pop.to_dataframe()
    gmm_labels_plot = df[f'gmm_est_segment_id']
    plot_segmentation(gmm_labels_plot, df, algo="gmm", M=n_segments)
    
    return bic

from scipy.spatial.distance import cdist

def pair_segments_by_parameters(X, D, Y, gmm_labels, n_meta_segments):
    """
    Force pairing of 2M GMM segments into M meta-segments based on parameter similarity.
    Returns:
        meta_labels: array of length N with meta-segment IDs
        paired_params: list of meta-segment (mean) parameters
    """
    unique_labels = np.unique(gmm_labels)
    assert len(unique_labels) == 2 * n_meta_segments, "Must have exactly 2M GMM segments to pair into M"

    param_vectors = []
    segment_param_map = {}

    # Step 1: estimate parameters
    for label in unique_labels:
        idx = (gmm_labels == label)
        _, beta, _, _, _ = estimate_segment_parameters(X[idx], D[idx], Y[idx])
        param_vectors.append(beta)
        segment_param_map[label] = beta

    param_vectors = np.array(param_vectors)
    unpaired_labels = list(unique_labels)
    paired_groups = []
    paired_params = []

    # Step 2: Greedy pairing by closest Euclidean distance
    while unpaired_labels:
        i = unpaired_labels[0]
        rest = unpaired_labels[1:]
        dists = cdist([segment_param_map[i]], [segment_param_map[j] for j in rest])
        j_idx = np.argmin(dists)
        j = rest[j_idx]
        paired_groups.append((i, j))
        avg_params = 0.5 * (segment_param_map[i] + segment_param_map[j])
        paired_params.append(avg_params)
        unpaired_labels.remove(i)
        unpaired_labels.remove(j)

    # Step 3: Map GMM segment to meta-segment
    segment_to_meta = {}
    for meta_id, (i, j) in enumerate(paired_groups):
        segment_to_meta[i] = meta_id
        segment_to_meta[j] = meta_id

    meta_labels = np.array([segment_to_meta[label] for label in gmm_labels])

    return meta_labels, paired_params
