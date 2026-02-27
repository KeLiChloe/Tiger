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


def policy_oracle(customers, algo, signal_d=None):
    """
    Compute policy evaluation metrics:
    - Manager profit
    - Oracle profit
    - Regret
    - Mis-treatment rate

    Parameters:
    - customers: list of Customer objects (each linked to true and estimated segments)
    - signal_d: int, number of signal dimensions used in outcome model (pop.signal_d).
                If None, uses full feature dimension (incorrect when disturb_d > 0).

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

        # Determine signal_d: use provided value, else fall back to full dim
        sd = signal_d if signal_d is not None else len(x_i)

        # Oracle action: best action for this segment (pre-computed from true tau)
        oracle_action = true_seg.action
        oracle_profit_i = true_seg.expected_outcome(x_i, oracle_action, signal_d=sd)

        # Manager policy
        if est_seg.est_action == 404:
            manager_profit_i = oracle_profit_i
        else:
            manager_profit_i = true_seg.expected_outcome(x_i, est_seg.est_action, signal_d=sd)

        # Mistreatment if manager’s action ≠ oracle
        if est_seg.est_action != oracle_action and est_seg.est_action != 404:
            mistreated += 1

        oracle_profit += oracle_profit_i
        manager_profit += manager_profit_i

    return {
        "manager_profit": manager_profit,
        "oracle_profit": oracle_profit,
        "regret": (oracle_profit - manager_profit) / len(customers),
        "mistreatment_rate": mistreated / len(customers)
    }
