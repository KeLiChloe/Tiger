
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
        

def estimate_segment_parameters(X, D, Y):
    """Fit OLS model Y ~ x + D and return parameters ."""
    # if len(X) < X.shape[1] + 1:
    #     raise ValueError("Too few samples to estimate parameters reliably.")

    X_design = build_design_matrix(X, D)
    
    model = LinearRegression(fit_intercept=False).fit(X_design, Y)
    theta = model.coef_.ravel()
    est_alpha = theta[0]
    est_beta = theta[1:-1]
    est_tau = theta[-1]
    est_action = int(est_tau >= 0)
    
    return est_alpha, est_beta, est_tau, est_action, theta


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
    
    
def print_oast_tree(node, feature_names=None, indent=""):
    """
    Recursively print the structure of an OAST tree.

    Parameters:
        node (OASTNode): The current node to print.
        feature_names (list): Optional list of feature names for clarity.
        indent (str): Current indentation (used in recursion).
    """
    if node.is_leaf:
        print(f"{indent}Leaf [segment_id={node.segment_id}, n={len(node.indices)}] "
              f"value={node.value:.4f}, tau_hat={node.tau_hat:.4f}")
    else:
        feat = (feature_names[node.split_feature]
                if feature_names else f"x_{node.split_feature}")
        print(f"{indent}Split: {feat} <= {node.split_threshold:.4f} "
              f"(value={node.value:.4f}, tau_hat={node.tau_hat:.4f})")
        print_oast_tree(node.left, feature_names, indent + "  ")
        print_oast_tree(node.right, feature_names, indent + "  ")

