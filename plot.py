import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

# optional parameters
import matplotlib.pyplot as plt
import numpy as np

def plot_segmentation(labels, X, y_vec, D_vec, algo, M=None):
    """
    Visualize segmentation in 2D (x, y) with:
        - Colors by segment
        - Markers by treatment group
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

    plt.figure(figsize=(10, 6))

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

    plt.xlabel("x_1")
    plt.ylabel("outcome")
    plt.title(f"{algo.upper()}-Based Segmentation")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # print(f"figures/{algo}_segmentation_{M}.png")
    plt.savefig(f"figures/{algo}_segmentation_{M}.png", dpi=300)



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
    plt.figure(figsize=(10, 6
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


