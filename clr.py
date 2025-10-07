from builtins import range
import numpy as np
from sklearn.base import BaseEstimator, clone
# from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from ground_truth import PopulationSimulator, SegmentEstimate
from utils import assign_trained_customers_to_segments, estimate_segment_parameters, assign_new_customers_to_segments, evaluate_on_validation


# The expected shape of y is (N,) not (N, 1)!!!!!

class CLRpRegressor(BaseEstimator):
    def __init__(self, num_planes, kmeans_coef, clr_lr=None, max_iter=5, num_tries=8, clf=None):
        self.num_planes = num_planes
        self.kmeans_coef = kmeans_coef
        self.num_tries = num_tries
        self.clr_lr = clr_lr
        self.max_iter = max_iter

        if clf is None:
            self.clf = RandomForestClassifier(n_estimators=10)
        else:
            self.clf = clf

    def fit(self, X_D, y, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.cluster_labels, self.models, _, _ = best_clr(
                    X_D, y, k=self.num_planes, kmeans_coef=self.kmeans_coef,
                    max_iter=self.max_iter, num_tries=self.num_tries,
                    lr=self.clr_lr
        )
        
        if np.unique(self.cluster_labels).shape[0] == 1: # self.clf needs at least 2 classes
            self.cluster_labels[0] = 1 if self.cluster_labels[0] == 0 else 0
        
        # fit classifier to predict cluster labels
        # remove D_all from X when fitting clf
        X_no_D = X_D[:, :-1]
        self.clf.fit(X_no_D, self.cluster_labels)

    def predict(self, X_only):
        check_is_fitted(self, ['cluster_labels', 'models'])
        test_labels = self.clf.predict(X_only)
        return test_labels

def best_clr(X_D, y, k, num_tries=5, **kwargs):
    clr_func = clr
    best_obj = np.inf
    for i in range(num_tries):
        cluster_labels, models, weights, obj = clr_func(X_D, y, k, **kwargs)
        if obj < best_obj:
            best_obj = obj
            best_cluster_labels = cluster_labels
            best_models = models
            best_weights = weights
    return best_cluster_labels, best_models, best_weights, best_obj

def clr(X_D, y, k, kmeans_coef, lr=None, max_iter=10, cluster_labels=None):
  
    if cluster_labels is None:
        cluster_labels = np.random.choice(k, size=X_D.shape[0])
  
    if lr is None:
        # set fit_intercept=True if X do not contain intercept term
        lr = Ridge(alpha=1e-5, fit_intercept=True)
  
    models = [clone(lr) for _ in range(k)]
    scores = np.empty((X_D.shape[0], k))
    preds = np.empty((X_D.shape[0], k))

    for _ in range(max_iter):
    
        # rebuild models
        for cl_idx in range(k):
            if np.sum(cluster_labels == cl_idx) == 0:
                continue
            models[cl_idx].fit(X_D[cluster_labels == cl_idx], y[cluster_labels == cl_idx])
        
        
        # reassign points
        for cl_idx in range(k):
            preds[:, cl_idx] = models[cl_idx].predict(X_D)
            scores[:, cl_idx] = (y - preds[:, cl_idx]) ** 2
        
        # TODO: do something when cluster vanishes?
            if np.sum(cluster_labels == cl_idx) == 0:
                continue
            if kmeans_coef > 0:
                center = np.mean(X_D[cluster_labels == cl_idx], axis=0)
                scores[:, cl_idx] += kmeans_coef * np.asarray(np.sum(np.square(X_D - center), axis=1)).squeeze()
            
        cluster_labels_prev = cluster_labels.copy()
        cluster_labels = np.argmin(scores, axis=1)
        
        if np.allclose(cluster_labels, cluster_labels_prev):
            break
  
    obj = np.mean(scores[np.arange(preds.shape[0]), cluster_labels])
    
    weights = (cluster_labels == np.arange(k)[:,np.newaxis]).sum(axis=1).astype(float)
    weights /= np.sum(weights)
    return cluster_labels, models, weights, obj

def bic_score(X_D, y, cluster_labels, models):
    n, d = X_D.shape
    k = len(models)
    # Predict with each assigned cluster model
    y_hat = np.zeros_like(y, dtype=float)
    for cl_idx in range(k):
        mask = (cluster_labels == cl_idx)
        if np.sum(mask) == 0:
            continue
        y_hat[mask] = models[cl_idx].predict(X_D[mask])
    residuals = y - y_hat
    sigma2 = np.mean(residuals**2)

    # log-likelihood under Gaussian errors
    logL = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)

    # number of parameters
    p = k * (d+1)  # k linear models with d params each + (k-1) for cluster probs

    BIC = -2 * logL + p * np.log(n)
    return BIC


def CLR_segment_and_estimate(pop: PopulationSimulator, n_segments: int, x_mat, D_vec, y_vec, kmeans_coef, num_tries, algo, random_state=None):
    
    
    # Important: y_vec needs to be shape (N,) not (N, 1). 
    y_vec = y_vec.ravel()
    X_D = np.column_stack((x_mat, D_vec))  # add D as a feature
    CLR = CLRpRegressor(num_planes=n_segments, kmeans_coef=kmeans_coef, num_tries=num_tries)
    CLR.fit(X_D, y_vec, seed=random_state)
    clr_labels = CLR.cluster_labels
    bic = bic_score(X_D, y_vec, CLR.cluster_labels, CLR.models)
    
    # Assign each customer to estimated segment
    pop.est_segments_list[f"{algo}"] = []  # Reset!!!!

    # Estimate triplets via OLS per segment
    for m in range(n_segments):
        idx_m = np.where(clr_labels == m)[0]
        
        if len(idx_m) == 0:
            raise ValueError(f"No customers assigned to segment {m}. ")

        x_m = x_mat[idx_m]
        D_m = D_vec[idx_m]
        y_m = y_vec[idx_m]
        
        est_alpha, est_beta, est_tau, est_action = estimate_segment_parameters(x_m, D_m, y_m)
        est_seg = SegmentEstimate(est_alpha, est_beta, est_tau, est_action, segment_id=m)
        pop.est_segments_list[f"{algo}"].append(est_seg)

    # Link each customer to estimated segment
    assign_trained_customers_to_segments(pop, clr_labels, f"{algo}")
    
    model_selection = algo.split("-")[-1]
    if model_selection == "standard":
        return bic, CLR
    elif model_selection == "da":
        assign_new_customers_to_segments(pop, pop.val_customers, CLR, algo)
        DA_score = evaluate_on_validation(pop, algo=algo)
        return  DA_score, CLR
    else:
        raise ValueError("model_selection must be either 'standard' or 'da'")

