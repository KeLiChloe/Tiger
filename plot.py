import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

# optional parameters
import matplotlib.pyplot as plt
import numpy as np

def plot_segmentation(labels, X, y_vec, D_vec, algo, M=None, tree=None):
    """
    Visualize segmentation in 2D (x, y) with:
        - Colors by segment
        - Markers by treatment group
        - Decision boundaries for tree-based methods (DAST, MST)
    
    Args:
        tree: For DAST/MST, the tree object to extract split boundaries
    """

    # what is X is multi-dimensional? In this case, we only plot the first dimension
    X = np.ravel(X[:, 0])
    y_vec = np.ravel(y_vec)
    D_vec = np.ravel(D_vec)

    unique_labels = np.unique(labels)
    label_to_color_idx = {label: i for i, label in enumerate(unique_labels)}
    cmap = plt.cm.get_cmap("Set1", len(unique_labels))
    markers = {0: 'o', 1: 'x'}

    # Axis limits
    x_min, x_max = X.min(), X.max()
    y_min, y_max = y_vec.min(), y_vec.max()

    plt.figure(figsize=(8, 6))

    # Scatter plot
    for label in unique_labels:
        for D in [0, 1]:
            idx = (labels == label) & (D_vec == D)
            color = cmap(label_to_color_idx[label])
            plt.scatter(
                X[idx],
                y_vec[idx],
                color=color,
                marker=markers[D],
                alpha=0.7,
                s=20,
                label=f"Seg {label}, D={D}"
            )

    # Draw decision boundaries for tree-based methods
    if tree is not None and algo in ["dast", "mst"]:
        splits = extract_tree_splits(tree)
        # splits = [4,9,11.0]
        for split_x in splits:
            plt.axvline(x=split_x, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Decision boundary')
        # Remove duplicate labels in legend
        handles, labels_legend = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels_legend, handles))
        plt.legend(by_label.values(), by_label.keys())
    else:
        plt.legend()

    plt.xlabel("x_1")
    plt.ylabel("outcome")
    plt.title(f"{algo.upper()}-Based Segmentation")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True)
    plt.tight_layout()
    # print(f"figures/{algo}_segmentation_{M}.png")
    plt.savefig(f"figures/{algo}_segmentation_{M}.png", dpi=300)
    plt.close()  # Close the figure to free memory


def extract_tree_splits(tree):
    """
    Extract all split thresholds from a tree (for plotting decision boundaries).
    
    Args:
        tree: DASTree or MSTree object
        
    Returns:
        List of split thresholds (x values where splits occur)
    """
    splits = []
    
    def traverse(node):
        if node is None or node.is_leaf:
            return
        # Only extract splits on the first feature (dimension 0) for 1D plotting
        if node.split_feature == 0:
            splits.append(node.split_threshold)
        traverse(node.left)
        traverse(node.right)
    
    traverse(tree.root)
    return sorted(splits)
    # return [7, 12]



def plot_ground_truth(df, title="Ground-Truth Segmentation", segment_col='true_segment_id', x_col='x_0', y_col='outcome', D_col='D_i'):
    """
    Plot outcome Y_i vs. covariate x_i in 1D simulation.
    
    Parameters:
    - df: pandas DataFrame with columns x_col, y_col, segment_col, D_col
    - title: plot title
    - segment_col: column indicating true segment membership
    - x_col: covariate column name (should be scalar)
    - y_col: outcome column name
    - D_col: binary treatment indicator column name
    """
    plt.figure(figsize=(8, 6
                        ))
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown']  # extendable for >3 segments
    markers = {0: 'o', 1: 'x'}

    segments = sorted(df[segment_col].unique())
    for k in segments:
        for D_i in [0, 1]:
            subset = df[(df[segment_col] == k) & (df[D_col] == D_i)]
            plt.scatter(
                subset[x_col],
                subset[y_col],
                color=colors[k % len(colors)],
                marker=markers[D_i],
                alpha=0.7,
                label=f"Segment {k}, D={D_i}"
            )

    plt.title(title, fontsize=20)
    plt.xlabel(f"${x_col}$", fontsize=16)
    plt.ylabel(f"${y_col}$", fontsize=16)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/ground_truth_plot.png", dpi=300)
    plt.close()  # Close the figure to free memory


def plot_implementation_clustering(implement_customers, algo, title=None):
    """
    Plot implementation customers with their clustering results and assigned treatment.
    
    Left plot: True segments with potential outcomes under BOTH treatments (counterfactual)
    Right plot: Estimated segments with algorithm-assigned treatment (factual)
    
    Parameters:
    - implement_customers: list of Customer_implement objects
    - algo: algorithm name (e.g., 'dast', 'kmeans-da', etc.)
    - title: optional custom title
    """
    # Extract data from customers
    X = np.array([cust.x[0] if len(cust.x) > 0 else 0 for cust in implement_customers])
    true_segment_ids = np.array([cust.true_segment.segment_id for cust in implement_customers])
    est_segment_ids = np.array([cust.est_segment[algo].segment_id if cust.est_segment[algo] is not None else -1 for cust in implement_customers])
    assigned_treatments = np.array([cust.implement_action for cust in implement_customers])  # Treatment assigned by algo
    
    # Generate ground truth potential outcomes for BOTH treatments
    Y_0 = np.array([cust.true_segment.generate_outcome(cust.x, 0, 0, cust.signal_d) for cust in implement_customers])
    Y_1 = np.array([cust.true_segment.generate_outcome(cust.x, 1, 0, cust.signal_d) for cust in implement_customers])
    
    # Actual observed outcome (based on assigned treatment)
    Y_observed = np.array([cust.y for cust in implement_customers])
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Define colors and markers
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown']
    markers = {0: 'o', 1: 'x'}
    marker_sizes = {0: 60, 1: 80}  # Make 'x' larger for better visibility
    
    # Left subplot: True segments with BOTH potential outcomes (counterfactual)
    ax1 = axes[0]
    unique_true_segments = sorted(np.unique(true_segment_ids))
    for seg_id in unique_true_segments:
        for treatment in [0, 1]:
            mask = (true_segment_ids == seg_id)
            if np.any(mask):
                Y_treatment = Y_0[mask] if treatment == 0 else Y_1[mask]
                # For 'o' marker (filled), use edgecolors; for 'x' marker (unfilled), don't use edgecolors
                if treatment == 0:
                    ax1.scatter(
                        X[mask],
                        Y_treatment,
                        color=colors[seg_id % len(colors)],
                        marker=markers[treatment],
                        alpha=0.6,
                        s=marker_sizes[treatment],
                        linewidths=1,
                        label=f"True Seg {seg_id}, T={treatment} (potential)"
                    )
                else:  # treatment == 1, 'x' marker
                    ax1.scatter(
                        X[mask],
                        Y_treatment,
                        color=colors[seg_id % len(colors)],
                        marker=markers[treatment],
                        alpha=0.6,
                        s=marker_sizes[treatment],
                        linewidths=2,  # Make 'x' marker thicker
                        label=f"True Seg {seg_id}, T={treatment} (potential)"
                    )
    
    ax1.set_xlabel("Covariate $x_0$", fontsize=14)
    ax1.set_ylabel("Potential Outcome $y$", fontsize=14)
    ax1.set_title(f"True Segments + Potential Outcomes Under Both Treatments\n(Ground Truth Counterfactuals)", fontsize=15, fontweight='bold')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Right subplot: Estimated segments with algo-assigned treatment (factual)
    ax2 = axes[1]
    unique_est_segments = sorted(np.unique(est_segment_ids[est_segment_ids >= 0]))
    for seg_id in unique_est_segments:
        for treatment in [0, 1]:
            mask = (est_segment_ids == seg_id) & (assigned_treatments == treatment)
            if np.any(mask):
                # For 'o' marker (filled), use edgecolors; for 'x' marker (unfilled), don't use edgecolors
                if treatment == 0:
                    ax2.scatter(
                        X[mask],
                        Y_observed[mask],
                        color=colors[seg_id % len(colors)],
                        marker=markers[treatment],
                        alpha=0.7,
                        s=marker_sizes[treatment],
                        linewidths=1,
                        label=f"Est Seg {seg_id}, Assigned T={treatment}"
                    )
                else:  # treatment == 1, 'x' marker
                    ax2.scatter(
                        X[mask],
                        Y_observed[mask],
                        color=colors[seg_id % len(colors)],
                        marker=markers[treatment],
                        alpha=0.7,
                        s=marker_sizes[treatment],
                        linewidths=2,  # Make 'x' marker thicker
                        label=f"Est Seg {seg_id}, Assigned T={treatment}"
                    )
    
    ax2.set_xlabel("Covariate $x_0$", fontsize=14)
    ax2.set_ylabel("Observed Outcome $y$", fontsize=14)
    ax2.set_title(f"Estimated Segments + Algorithm-Assigned Treatment\n({algo.upper()} Policy)", fontsize=15, fontweight='bold')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.legend(loc='best', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"figures/implementation_{algo}_clustering.png", dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory


