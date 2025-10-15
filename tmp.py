import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

# ---------- 参数 ----------
np.random.seed(42)

n_clusters = 2
n_samples_per_cluster = 200
d_total = 50
partial_x = 0.1  # signal 占比
d_signal = int(d_total * partial_x)
d_noise = d_total - d_signal

# ---------- signal centers ----------
signal_centers = np.array([[1]*d_signal, [-1]*d_signal]) * 5
# ---------- noise centers (deliberately flipped) ----------
noise_centers = np.array([[-1]*d_noise, [1]*d_noise]) * 8  # flipped structure

# ---------- 生成数据 ----------
X = []
true_labels = []

for k in range(n_clusters):
    # signal: cluster-dependent
    x_signal = np.random.randn(n_samples_per_cluster, d_signal) + signal_centers[k]
    # noise: cluster *inversely* dependent (flip index)
    x_noise = np.random.randn(n_samples_per_cluster, d_noise) + noise_centers[1 - k]
    # combine
    x_full = np.hstack([x_signal, x_noise])
    X.append(x_full)
    true_labels += [k] * n_samples_per_cluster

X = np.vstack(X)
true_labels = np.array(true_labels)

# ---------- y generation (only from signal) ----------
betas = np.random.randn(n_clusters, d_signal)
Y = np.zeros(len(X))
for i in range(len(X)):
    k = true_labels[i]
    Y[i] = betas[k] @ X[i, :d_signal] + np.random.normal(0, 1)

# ---------- KMeans on signal-only ----------
kmeans_signal = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
labels_signal = kmeans_signal.fit_predict(X[:, :d_signal])
ari_signal = adjusted_rand_score(true_labels, labels_signal)

# ---------- KMeans on full X ----------
kmeans_full = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
labels_full = kmeans_full.fit_predict(X)
ari_full = adjusted_rand_score(true_labels, labels_full)

# ---------- 回归性能 ----------
def fit_cluster_models(pred_labels, X_signal, Y, name=""):
    results = []
    for k in range(n_clusters):
        idx = (pred_labels == k)
        if np.sum(idx) > 0:
            model = LinearRegression().fit(X_signal[idx], Y[idx])
            y_pred = model.predict(X_signal[idx])
            mse = mean_squared_error(Y[idx], y_pred)
            results.append({
                "Cluster": k,
                f"N_Points_{name}": np.sum(idx),
                f"MSE_{name}": mse
            })
        else:
            results.append({
                "Cluster": k,
                f"N_Points_{name}": 0,
                f"MSE_{name}": np.nan
            })
    return pd.DataFrame(results).set_index("Cluster")

df_signal = fit_cluster_models(labels_signal, X[:, :d_signal], Y, name="Signal")
df_full = fit_cluster_models(labels_full, X[:, :d_signal], Y, name="Full")

summary = df_signal.join(df_full, lsuffix="_S", rsuffix="_F")
summary.loc["Total"] = {
    "N_Points_S": len(X),
    "MSE_S": np.nan,
    "N_Points_F": len(X),
    "MSE_F": np.nan,
    "ARI_Signal": ari_signal,
    "ARI_Full": ari_full
}

print(summary.round(4))
