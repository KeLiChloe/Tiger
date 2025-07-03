import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

# optional parameters
import matplotlib.pyplot as plt
import numpy as np

def plot_segmentation(labels, df, x_col='x_0', y_col='outcome', D_col='D_i', algo='gmm', M=None):
    """
    Visualize segmentation in 2D (x, y) with:
        - Colors by segment
        - Markers by treatment group
    """
    # Data
    X_plot = df[[x_col, y_col]].values
    unique_labels = np.unique(labels)
    label_to_color_idx = {label: i for i, label in enumerate(unique_labels)}
    cmap = plt.cm.get_cmap("Set1", len(unique_labels))
    markers = {0: 'o', 1: 'x'}

    # Axis limits
    x_min, x_max = X_plot[:, 0].min(), X_plot[:, 0].max()
    y_min, y_max = X_plot[:, 1].min(), X_plot[:, 1].max()

    plt.figure(figsize=(10, 6))

    # Scatter plot
    for label in unique_labels:
        for D in [0, 1]:
            idx = (labels == label) & (df[D_col] == D)
            color = cmap(label_to_color_idx[label])
            plt.scatter(
                X_plot[idx, 0],
                X_plot[idx, 1],
                color=color,
                marker=markers[D],
                alpha=0.7,
                s=20,
                label=f"Seg {label}, D={D}"
            )

    plt.xlabel(f"${x_col}$")
    plt.ylabel(f"${y_col}$")
    plt.title(f"{algo.upper()}-Based Segmentation")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
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


