from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import entropy
from sklearn.metrics.cluster import contingency_matrix
import numpy as np


def variation_of_information(true_labels, pred_labels):
    """
    Compute Variation of Information (VI) between two clusterings.
    """
    contingency = contingency_matrix(true_labels, pred_labels)
    joint = contingency / np.sum(contingency)
    pi = np.sum(joint, axis=1)
    pj = np.sum(joint, axis=0)

    H_true = entropy(pi)
    H_pred = entropy(pj)
    I = np.sum(joint * np.log((joint + 1e-10) / (pi[:, None] * pj[None, :] + 1e-10)))

    VI = H_true + H_pred - 2 * I
    return VI


def structure_oracle(true_labels, pred_labels):
    """
    Compute structure recovery metrics comparing true vs estimated clusters.

    Parameters:
    - true_labels: ground-truth segment IDs (Z_i)
    - pred_labels: estimated segment IDs (S_i^M)
    - true_K: optional int, known true number of clusters

    Returns:
    - dict with metrics: K-hit, ARI, NMI, VI
    """
    true_labels = np.array(true_labels).reshape(-1)
    pred_labels = np.array(pred_labels).reshape(-1)

    metrics = {}

    # ARI
    metrics['ARI'] = adjusted_rand_score(true_labels, pred_labels)

    # NMI
    metrics['NMI'] = normalized_mutual_info_score(true_labels, pred_labels)

    return metrics


def estimation_oracle(customers, algo):
    """
    Compute estimation accuracy metrics (param-level and outcome-level MSEs).
    
    Parameters:
    - customers: list of Customer objects, each with true and estimated segment
    
    Returns:
    - dict with metrics: MSE_param, MSE_outcome
    """
    d = customers[0].x.shape[0]
    param_errors = []
    outcome_errors = []

    for cust in customers:

        # True and estimated parameters
        true_seg = cust.true_segment
        est_seg = cust.est_segment[algo]
        

        theta_true = np.concatenate(([true_seg.alpha], true_seg.beta, [true_seg.tau]))
        if est_seg.est_action == 404:
            theta_est = theta_true
        else:
            theta_est = np.concatenate(([est_seg.est_alpha], est_seg.est_beta, [est_seg.est_tau]))

        # Parameter-level error
        param_errors.append(np.sum((theta_est - theta_true) ** 2))

        # Outcome-level error (under both treatment assignments a = 0, 1)
        for a in [0, 1]:
            if est_seg.est_action == 404:
                outcome_errors.append(0)
            else:
                y_pred = est_seg.est_alpha + est_seg.est_beta @ cust.x + est_seg.est_tau * a
                y_true = true_seg.alpha + true_seg.beta @ cust.x + true_seg.tau * a
                outcome_errors.append((y_pred - y_true) ** 2)

    N = len(param_errors)
    MSE_param = np.sum(param_errors) / (N * (d + 2))
    MSE_outcome = np.mean(outcome_errors)

    return {
        "MSE_param": MSE_param,
        "MSE_outcome": MSE_outcome
    }

def policy_oracle(customers, algo):
    """
    Compute policy evaluation metrics:
    - Manager profit
    - Oracle profit
    - Regret
    - Mis-treatment rate

    Parameters:
    - customers: list of Customer objects (each linked to true and estimated segments)

    Returns:
    - dict with metrics: manager_profit, oracle_profit, regret, mistreatment_rate
    """
    manager_profit = 0.0
    oracle_profit = 0.0
    mistreated = 0

    for cust in customers:
        x_i = cust.x
        true_seg = cust.true_segment
        est_seg = cust.est_segment[algo]

        # True terms
        baseline = true_seg.alpha + true_seg.beta @ x_i
        oracle_action = 1 if true_seg.tau > 0 else 0
        oracle_profit_i = baseline + (true_seg.tau if oracle_action == 1 else 0)

        # Manager policy: treat if estimated tau > 0
        manager_profit_i = baseline + (true_seg.tau if (est_seg.est_action == 1 or est_seg.est_action == 404) else 0)

        # Mistreatment if manager’s action ≠ oracle
        if est_seg.est_action != oracle_action and est_seg.est_action != 404:
            mistreated += 1

        oracle_profit += oracle_profit_i
        manager_profit += manager_profit_i

    return {
        "manager_profit": manager_profit,
        "oracle_profit": oracle_profit,
        "regret": (oracle_profit - manager_profit), # / len(customers),
        "mistreatment_rate": mistreated # / len(customers)
    }
