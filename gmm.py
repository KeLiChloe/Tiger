from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from ground_truth import PopulationSimulator, SegmentEstimate
from utils import build_design_matrix
import numpy as np


def GMM_segment_and_estimate(pop: PopulationSimulator, n_segments: int, random_state=None):
    """
    Perform GMM-based segmentation and OLS-based estimation per segment.

    Parameters:
        pop: PopulationSimulator object with simulated data
        n_segments: number of segments to estimate
        random_state: optional random seed for reproducibility
    """

    # Prepare input data for GMM
    df = pop.to_dataframe()
    x_cols = [f'x_{j}' for j in range(pop.d)]
    x_mat = df[x_cols].values
    y_vec = df['outcome'].values.reshape(-1, 1)
    gmm_input = np.hstack([x_mat, y_vec])  # shape (N, d+1)

    # Fit GMM
    gmm = GaussianMixture(n_components=n_segments, random_state=random_state)
    gmm_labels = gmm.fit_predict(gmm_input)
    bic = gmm.bic(gmm_input)

    # Assign each customer to estimated segment
    pop.gmm_est_segments = []
    for seg_id in range(n_segments):
        pop.gmm_est_segments.append(None)  # placeholder for now

    # Estimate triplets via OLS per segment
    for m in range(n_segments):
        idx_m = np.where(gmm_labels == m)[0]
        if len(idx_m) == 0:
            raise ValueError(f"No customers assigned to segment {m}. ")

        x_m = x_mat[idx_m]
        D_m = df.loc[idx_m, 'D_i'].values
        y_m = df.loc[idx_m, 'outcome'].values

        X_design = build_design_matrix(x_m, D_m)  # [1, x, D]
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X_design, y_m)
        theta = reg.coef_.ravel()

        est_alpha = theta[0]
        est_beta = theta[1:-1]
        est_tau = theta[-1]
        # if est_tau < 0: action = 0; if est_tau >= 0: action = 1
        est_action = 1 if est_tau >= 0 else 0

        # Create estimated segment object
        est_seg = SegmentEstimate(est_alpha, est_beta, est_tau, est_action, segment_id=m)
        est_seg.count = len(idx_m)
        pop.gmm_est_segments[m] = est_seg

    # Link each customer to estimated segment
    for i, cust in enumerate(pop.customers):
        m = gmm_labels[i]
        cust.gmm_est_segment = pop.gmm_est_segments[m]
    
    return gmm, bic

