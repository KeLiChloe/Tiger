from sklearn.mixture import GaussianMixture
from ground_truth import PopulationSimulator, SegmentEstimate
from utils import assign_trained_customers_to_segments, evaluate_on_validation, assign_new_customers_to_segments
from sklearn.linear_model import LinearRegression
import numpy as np



def GMM_segment_and_estimate(pop: PopulationSimulator, n_segments: int, x_mat, D_vec, y_vec,algo, random_state=None):
    """
    Perform GMM-based segmentation and OLS-based estimation per segment.

    Parameters:
        pop: PopulationSimulator object with all simulated data
        n_segments: number of segments to estimate
        random_state: optional random seed for reproducibility
    """
    
    gmm_model = GaussianMixture(n_components=n_segments, random_state=random_state)
    gmm_labels = gmm_model.fit_predict(x_mat)
    bic = gmm_model.bic(x_mat)
        
    # Assign each customer to estimated segment
    pop.est_segments_list[f"{algo}"] = []  # Reset!!!!

    # Estimate triplets via OLS per segment
    for m in range(n_segments):
        idx_m = np.where(gmm_labels == m)[0]
        
        if len(idx_m) == 0:
            raise ValueError(f"No customers assigned to segment {m}. ")

        x_m = x_mat[idx_m]
        D_m = D_vec[idx_m]
        y_m = y_vec[idx_m]
        
        est_alpha, est_beta, est_tau, est_action = estimate_segment_parameters(x_m, D_m, y_m)
        est_seg = SegmentEstimate(est_alpha, est_beta, est_tau, est_action, segment_id=m)
        pop.est_segments_list[f"{algo}"].append(est_seg)

    # Link each customer to estimated segment
    assign_trained_customers_to_segments(pop, gmm_labels, f"{algo}")
        
    model_selection = algo.split("-")[-1]
    if model_selection == "standard":
        return bic, gmm_model
    elif model_selection == "da":
        assign_new_customers_to_segments(pop, pop.val_customers, gmm_model, algo)
        Gamma_val = pop.gamma[[cust.customer_id for cust in pop.val_customers]]
        DA_score = evaluate_on_validation(pop, algo=algo, Gamma_val=Gamma_val)
        return  DA_score,gmm_model
    else:
        raise ValueError("model_selection must be either 'standard' or 'da'")
    

def estimate_segment_parameters(X, D, Y):

    X_design = np.hstack((np.ones((X.shape[0], 1)), X))
    
    model = LinearRegression(fit_intercept=False).fit(X_design, Y)
    theta = model.coef_.ravel()
    est_alpha = theta[0]
    est_beta = theta[1:]
    est_tau = np.mean(Y[D==1]) - np.mean(Y[D==0])
    est_action = int(est_tau >= 0)
    return est_alpha, est_beta, est_tau, est_action