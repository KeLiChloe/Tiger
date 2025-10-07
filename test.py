# Generate N random input data points
import numpy as np
from clr import CLRpRegressor, bic_score
import matplotlib.pyplot as plt

import numpy as np

import os

if "R_SESSION_TMPDIR" in os.environ:
    del os.environ["R_SESSION_TMPDIR"]


def generate_ground_truth(N_per_cluster, clusters, noise_std=1.0, seed=None):
    """
    Generate ground-truth data with cluster-specific parameters.
    
    Parameters:
    - N_per_cluster: number of samples per cluster
    - clusters: list of dicts, each with keys {"x_range": (low, high), "alpha": float, "beta": float, "tau": float}
    - noise_std: std deviation of Gaussian noise
    - seed: random seed
    
    Returns:
    - X: np.array of shape (N_total,)
    - D: np.array of binary treatments
    - y: np.array of outcomes
    - labels: cluster assignment for each point
    """
    if seed is not None:
        np.random.seed(seed)
    
    X_all, D_all, y_all, labels = [], [], [], []
    
    for k, params in enumerate(clusters):
        low, high = params["x_range"]
        X = np.random.uniform(low, high, N_per_cluster)
        D = np.random.binomial(1, 0.5, N_per_cluster)  # treatment assignment
        eps = np.random.normal(0, noise_std, N_per_cluster)
        
        y = params["alpha"] + params["beta"] * X + params["tau"] * D + eps
        
        X_all.append(X)
        D_all.append(D)
        y_all.append(y)
        labels.append(np.full(N_per_cluster, k))
    
    return (
        np.concatenate(X_all),
        np.concatenate(D_all),
        np.concatenate(y_all),
        np.concatenate(labels),
    )


def plot_ground_truth(X_all, y_all, labels, clusters):
    plt.figure(figsize=(8, 6))

    for cid, params in enumerate(clusters):
        idx = labels == cid
        plt.scatter(X_all[idx], y_all[idx], label=f"Cluster {cid}", alpha=0.6)

        # x range for this cluster
        low, high = params["x_range"]
        r = np.linspace(low, high, 100)

        # Control line (D=0)
        y_control = params["alpha"] + params["beta"] * r
        plt.plot(r, y_control, color="black", linestyle="--", alpha=0.7)

        # Treatment line (D=1)
        y_treatment = params["alpha"] + params["beta"] * r + params["tau"]
        plt.plot(r, y_treatment, color="red", linestyle="-", alpha=0.7)

    plt.title("Ground Truth: Clusters with Control & Treatment Lines")
    plt.legend()
    plt.show()

def plot_clustering_results(X_all, y_all, labels_pred):
    plt.figure(figsize=(8, 6))

    for cid in np.unique(labels_pred):
        idx = labels_pred == cid
        plt.scatter(X_all[idx], y_all[idx], label=f"Pred Cluster {cid}", alpha=0.6)

    plt.title("Estimated Clusters from CLR")
    plt.legend()
    plt.show()


# clusters = [
#     {"x_range": (-5, 10), "alpha": 2.0, "beta": 1.5, "tau": 15.0},
#     {"x_range": (0, 20), "alpha": -5.0, "beta": 7.8, "tau": 13.0},
#     {"x_range": (5, 15), "alpha": 0.5, "beta": -2.0, "tau": 17.0},
# ]

# set seed for reproducibility
seed = np.random.randint(0, 10000)
np.random.seed(7659)
print("Random seed:", seed)


# generate clusters randomly
clusters = [
    {"x_range": (-5, 5), "alpha": np.random.uniform(-5, 5), "beta": np.random.uniform(-10, 10), "tau": np.random.uniform(-50, 50)},
    {"x_range": (3, 15), "alpha": np.random.uniform(-10, 10), "beta": np.random.uniform(-10, 10), "tau": np.random.uniform(-50, 50)},
    {"x_range": (10, 25), "alpha": np.random.uniform(5, 25), "beta": np.random.uniform(-10, 10), "tau": np.random.uniform(-50, 50)},
]

X_all, D_all, y_all, labels = generate_ground_truth(
    N_per_cluster=50,
    clusters=clusters,
    noise_std=3.0,
    seed=42
)

# aug_X_all = np.column_stack((X_all, D_all))
aug_X_all = np.column_stack(( X_all, D_all))  # add intercept term

# After generating ground truth
plot_ground_truth(X_all, y_all, labels, clusters)

kmeans_coef = 0.3

bic_scores = []
labels_pred_list = []
CLR_list = []
for M in range(2, 8):
    print(f"Fitting CLR with M = {M} clusters...")
    CLR = CLRpRegressor(num_planes=M, kmeans_coef=kmeans_coef, num_tries=10)

    CLR.fit(aug_X_all, y_all)

    # bic_score
    bic = bic_score(aug_X_all, y_all, CLR.cluster_labels, CLR.models)
    bic_scores.append(bic)
    labels_pred_list.append(CLR.cluster_labels)
    CLR_list.append(CLR)

print("Optimal M by BIC:", np.argmin(bic_scores) + 2)

# After CLR.fit(...)
optimal_M = np.argmin(bic_scores)
labels_pred = labels_pred_list[optimal_M]
CLR = CLR_list[optimal_M]
plot_clustering_results(X_all, y_all, labels_pred)


# predict cluster for new data points
X_new = np.array([[ 20]])
pred_labels = CLR.predict(X_new)
print("Predicted clusters for new points:", pred_labels)