import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

# optional parameters
def plot_segmentation(pop, df=None, x_col='x_0', y_col='outcome', D_col='D_i', algo='gmm', gmm=None):
    """
    Visualize GMM segmentation in 2D (x_0, y) with:
        - Colored customer points by segment
        - Markers for D_i (o for control, x for treated)
        - Soft-colored ellipses from GMM covariances
        - Styled to match ground-truth plot

    Parameters:
    - pop: PopulationSimulator
    - df: optional DataFrame (calls pop.to_dataframe() if None)
    - x_col, y_col: plot axes
    - D_col: treatment indicator
    """
    if df is None:
        df = pop.to_dataframe()

    # Data and metadata
    X_plot = df[[x_col, y_col]].values
    # labels = np.array([cust.est_segment.segment_id if cust.est_segment else -1 for cust in pop.customers])
    labels = df[f'{algo}_est_segment_id'].values
    M = np.max(labels) + 1
    colors = plt.cm.get_cmap("Set1", M)
    markers = {0: 'o', 1: 'x'}

    # Fix axis limits for consistency with ground-truth
    x_min, x_max = X_plot[:, 0].min(), X_plot[:, 0].max()
    y_min, y_max = X_plot[:, 1].min(), X_plot[:, 1].max()

    plt.figure(figsize=(10, 6))

    # Scatter: by segment + treatment
    for m in range(M):
        for D in [0, 1]:
            idx = (labels == m) & (df[D_col] == D)
            plt.scatter(
                X_plot[idx, 0],
                X_plot[idx, 1],
                color=colors(m),
                marker=markers[D],
                alpha=0.7,
                s=20,
                label=f"Seg {m}, D={D}"
            )

    # Ellipses: soft Gaussian shading
    if algo == "gmm":
        if gmm is None:
            raise ValueError("GMM object must be provided for GMM-based segmentation.")
        for m in range(M):
            mean = gmm.means_[m]

            if gmm.covariance_type == 'full':
                cov = gmm.covariances_[m]
            elif gmm.covariance_type == 'tied':
                cov = gmm.covariances_
            elif gmm.covariance_type == 'diag':
                cov = np.diag(gmm.covariances_[m])
            elif gmm.covariance_type == 'spherical':
                cov = np.eye(2) * gmm.covariances_[m]
            else:
                raise ValueError(f"Unsupported covariance type: {gmm.covariance_type}")

            vals, vecs = np.linalg.eigh(cov)
            order = np.argsort(vals)[::-1]
            vals, vecs = vals[order], vecs[:, order]

            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            width, height = 2 * np.sqrt(vals)

            ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                            alpha=0.2, color=colors(m), zorder=0)
            plt.gca().add_patch(ellipse)

    # Final styling
    plt.xlabel(f"${x_col}$")
    plt.ylabel(f"${y_col}$")
    plt.title(f"{algo.upper()}-Based Segmentation")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{algo}_segmentation.png", dpi=300)


def plot_ground_truth(df, title="Outcome $Y_i$ vs. Covariate $x_i$", segment_col='true_segment_id', x_col='x_0', y_col='outcome', D_col='D_i'):
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

    plt.title(title)
    plt.xlabel(f"${x_col}$")
    plt.ylabel(f"${y_col}$")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ground_truth_plot.png", dpi=300)


