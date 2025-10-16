
import numpy as np
from ground_truth import PopulationSimulator
from sklearn.linear_model import LinearRegression
import pandas as pd
import plotly.graph_objects as go

def build_design_matrix(x_array, D_array):
    """Add intercept and treatment column to covariates."""
    N, _ = x_array.shape
    intercept = np.ones((N, 1))
    D_col = D_array.reshape(-1, 1)
    return np.hstack([intercept, x_array, D_col])

def assign_trained_customers_to_segments(pop: PopulationSimulator, segment_labels, algo):
    """
    Assign customers to estimated segments based on segment labels.
    
    Parameters:
        pop: PopulationSimulator object with all simulated data
        segment_labels: array of segment labels for each customer
        
    Returns:
        None, modifies pop.customers in-place
    """
    for i, cust in enumerate(pop.train_customers):
        m = segment_labels[i]
        cust.est_segment[algo] = pop.est_segments_list[algo][m]
        assert cust.est_segment[algo].segment_id == m, f"Segment ID mismatch for customer {cust.customer_id}: expected {m}, got {cust.est_segment[algo].segment_id}"
        

def estimate_segment_parameters(X, D, Y):
    """Fit OLS model Y ~ x + D and return parameters ."""
    if len(X) < X.shape[1] + 1:
        # print("Warning: Not enough data to fit OLS.")
            
        return 404, np.ones(X.shape[1])*404, 404, 404
    
    X_design = build_design_matrix(X, D)
    
    model = LinearRegression(fit_intercept=False).fit(X_design, Y)
    theta = model.coef_.ravel()
    est_alpha = theta[0]
    est_beta = theta[1:-1]
    # check if D_vec contains only 0s or 1s
    if np.all(D == 0) or np.all(D == 1):
        print("Warning: D_i contains only one treatment assignment.")
        return 404, np.ones(X.shape[1])*404, 404, 404
        
    else:
        est_tau = theta[-1]
        est_action = int(est_tau >= 0)
        return est_alpha, est_beta, est_tau, est_action


def plot_segment_sankey(original, pruned):
    df = pd.DataFrame({'orig': original, 'pruned': pruned})
    flow = df.groupby(['orig', 'pruned']).size().reset_index(name='count')

    # Create node labels
    orig_labels = sorted(df['orig'].unique())
    pruned_labels = sorted(df['pruned'].unique())
    all_labels = [f'Orig {i}' for i in orig_labels] + [f'Pruned {i}' for i in pruned_labels]

    label_to_index = {label: idx for idx, label in enumerate(all_labels)}

    # Build Sankey diagram
    source = [label_to_index[f'Orig {o}'] for o in flow['orig']]
    target = [label_to_index[f'Pruned {p}'] for p in flow['pruned']]
    value = flow['count'].tolist()

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=all_labels),
        link=dict(source=source, target=target, value=value)
    )])
    fig.update_layout(title_text="Segment Merge Flow (Original → Pruned)", font_size=12)
    fig.show()
    


# estimated total profits of a segment after applying the learnt policy to the customers in that segment
def compute_node_DR_value(Y, D, gamma, indices, buff):
    D_m = D[indices]
    Y_m = Y[indices]

    n1 = np.sum(D_m == 1)
    n0 = np.sum(D_m == 0)

    # TODO: what is a reasonable return value here?
    if n1 == 0 or n0 == 0:
        return 0 
        # raise ValueError("The number of customers of an action is 0!")

    y1 = np.mean(Y_m[D_m == 1])
    y0 = np.mean(Y_m[D_m == 0])
    tau_hat = y1 - y0
    a_i = int(tau_hat >= 0)
    
    
    if buff:
        # Method 1: Direct + Gamma
        # If D_m[i] = a_i, then we get profit Y_m[i]
        # If D_m[i] != a_i, then we get profit gamma_a_i_m[i]
        gamma_a_i_m = gamma[indices, a_i].reshape(-1, 1)
        value = np.sum(Y_m[D_m == a_i]) + np.sum(gamma_a_i_m[D_m != a_i])
    else:
        # Method 2: Gamma only
        gamma_0_m = gamma[indices, 0]
        gamma_1_m = gamma[indices, 1]
        V_i = (1 - a_i) * gamma_0_m + a_i * gamma_1_m
        value = np.sum(V_i)
    
    return value


def compute_residual_value(X, Y, D, indices):
    # Subset the data
    X_m = X[indices]
    D_m = D[indices].reshape(-1, 1)
    Y_m = Y[indices].reshape(-1, 1)

    # Construct the design matrix: [intercept | X | D]
    X_design = build_design_matrix(X_m, D_m)
    
    # Fit the linear model
    model = LinearRegression(fit_intercept=False).fit(X_design, Y_m)
    
    # Predict
    Y_pred = model.predict(X_design)

    # Compute residuals
    residuals = np.sum((Y_m - Y_pred) ** 2)

    return residuals/len(indices)
    # return residuals

def evaluate_on_validation(pop: PopulationSimulator, algo):
    # the validation customers need to have already been assigned to estimated segments
    
    if len(pop.val_customers) == 0:
        return None
    
    Gamma_val = pop.gamma[[cust.customer_id for cust in pop.val_customers]]

    V = 0
    for i, cust in enumerate(pop.val_customers):
        assigned_action = cust.est_segment[algo].est_action
        if assigned_action == 404 and algo != "mst" and algo != "policy_tree" and algo != "policy_tree-buff":
            # print("404!!!!")
            assigned_action = cust.true_segment.action
            cust.est_segment[algo].est_action = cust.true_segment.action
        else:
            # randomly assign action
            assigned_action = np.random.choice([0, 1])
            cust.est_segment[algo].est_action = assigned_action
        
        if int(assigned_action) == int(cust.D_i):
            V += cust.y
        else:
            V += Gamma_val[i, assigned_action]

    return V / len(pop.val_customers)

def assign_new_customers_to_segments(pop: PopulationSimulator, customers, model, algo):
    """
    Assign new customers to segments based on the trained model.
    
    Parameters:
        pop: PopulationSimulator object with all simulated data
        algo: algorithm used for segmentation
    Returns:
        None, modifies pop.customers in-place
    """
    
    x_new = np.array([cust.x for cust in customers])
    if x_new.shape[0] > 0:
        labels = model.predict(x_new)
        for cust, seg_id in zip(customers, labels):
            segment = pop.est_segments_list[f"{algo}"][seg_id]
            cust.est_segment[f"{algo}"] = segment


def pick_M_for_algo(algo, df_results_M):
    max_val = ["gmm-da", "kmeans-da", "clr-da", "policy_tree", "policy_tree-buff", "dast", "mst", "kmeans-standard"]
    min_val = ["gmm-standard", "clr-standard"]
    if algo in max_val:
        algo_picked_M = {
            f'{algo}_picked_M': df_results_M.at[df_results_M[f'{algo}_val'].idxmax(), 'M'],
        }
    elif algo in min_val: 
        algo_picked_M = {
            f'{algo}_picked_M': df_results_M.at[df_results_M[f'{algo}_val'].idxmin(), 'M'],
        }
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    return algo_picked_M