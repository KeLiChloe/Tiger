import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted

class CLRcRegressor(BaseEstimator, RegressorMixin):
    """
    Clusterwise Linear Regression.

    Iteratively fits k separate linear models on clusters of the data.

    Parameters
    ----------
    n_clusters : int
        Number of clusters (planes) to fit.
    init : {'kmeans', 'random'} (default='kmeans')
        Initialization method for cluster assignments.
    alpha : float, default=1e-5
        Regularization strength for Ridge regression.
    max_iter : int, default=10
        Maximum number of EM iterations.
    tol : float, default=1e-4
        Convergence threshold on cluster assignments.
    random_state : int or None
        Random seed for reproducibility.
    """
    def __init__(self, n_clusters=3, init='kmeans', alpha=1e-5,
                 max_iter=10, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit the clusterwise linear regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self : returns an instance of self.
        """
        n_samples, n_features = X.shape

        # Initialize cluster labels
        if self.init == 'kmeans':
            km = KMeans(n_clusters=self.n_clusters,
                        random_state=self.random_state)
            labels = km.fit_predict(X)
        else:
            rng = np.random.RandomState(self.random_state)
            labels = rng.randint(self.n_clusters, size=n_samples)

        # Initialize Ridge regressors
        # fit_intercept=False bebause we already add intercept manually to design matrix
        base_estimator = Ridge(alpha=self.alpha, fit_intercept=False)
        self.models_ = [clone(base_estimator) for _ in range(self.n_clusters)]

        for iteration in range(self.max_iter):
            labels_old = labels.copy()

            # Fit one linear model per cluster
            for k in range(self.n_clusters):
                mask = labels == k
                if np.any(mask):
                    self.models_[k].fit(X[mask], y[mask])
                else:
                    # Reinitialize empty cluster center by random sample
                    labels[np.random.choice(n_samples)] = k

            # Compute squared errors for each model and assign clusters
            errors = np.column_stack([
                self.models_[k].predict(X) for k in range(self.n_clusters)
            ])
            errors = (y[:, np.newaxis] - errors) ** 2
            labels = np.argmin(errors, axis=1)

            # Check convergence of labels
            changes = np.sum(labels != labels_old)
            if changes / n_samples < self.tol:
                break

        # After convergence, store labels and compute weights
        self.labels_ = labels
        counts = np.bincount(labels, minlength=self.n_clusters)
        self.cluster_weights_ = counts / n_samples
        return self

    def predict(self, X):  # noqa: D102
        check_is_fitted(self, 'models_')

        # Predict per-cluster outputs and choose minimal error
        preds = np.column_stack([
            model.predict(X) for model in self.models_
        ])

        # Cannot compute true error without y; return mixture or cluster-specific
        # Here we choose the cluster with highest prior weight
        cluster = np.argmax(self.cluster_weights_)
        return preds[:, cluster]

    def score(self, X, y):  # noqa: D102
        # R^2 score aggregated over clusters
        labels = self.predict_clusters(X)
        y_pred = np.empty_like(y)
        for k, model in enumerate(self.models_):
            mask = labels == k
            if np.any(mask):
                y_pred[mask] = model.predict(X[mask])
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u / v

    def predict_clusters(self, X):  # noqa: D102
        check_is_fitted(self, 'models_')
        X = check_array(X)
        errors = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        errors = (errors - errors.mean(axis=1, keepdims=True))  # dummy transform
        return np.argmin((errors - errors.mean(axis=1, keepdims=True))**2, axis=1)


