import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from econml.dr import DRLearner
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, default_converter
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
grf = importr('grf')
policytree = importr('policytree')
ro.r('library(policytree)')

class SegmentTrue:
    def __init__(self, alpha, beta, tau, x_mean, segment_id):
        self.alpha = alpha
        self.beta = beta  # np.array of shape (d,)
        self.tau = tau
        self.x_mean = x_mean
        self.segment_id = segment_id
        self.count = 0

    def generate_outcome(self, x, D_i, noise_std):
        noise = np.random.normal(0, noise_std)
        return self.alpha + self.beta @ x + self.tau * D_i + noise

        
class SegmentEstimate:
    def __init__(self, est_alpha, est_beta, est_tau, est_action, count, segment_id=None):
        self.est_alpha = est_alpha
        self.est_beta = est_beta  # np.array of shape (d,)
        self.est_tau = est_tau
        self.est_action = est_action  # 0 or 1
        self.segment_id = segment_id
        self.count = 0


class Customer:
    def __init__(self, x, true_segment: SegmentTrue, D_i, noise_std=1.0, customer_id=None):
        self.x = x
        self.D_i = D_i
        self.true_segment = true_segment
        self.est_segment = {
            "gmm": None,
            "oast": None,
            "policy_tree": None
        }

        self.y = true_segment.generate_outcome(x, D_i, noise_std)
        self.customer_id = customer_id
        
# ----------------------------------------
# Population Simulator
# ----------------------------------------

class PopulationSimulator:
    def __init__(self, N, d, K, covariate_noise, param_range, noise_std, DR_generation_method="forest"):
        self.N = N
        self.d = d
        self.K = K
        assert isinstance(param_range, dict), "param_range must be a dictionary with keys: alpha, beta, tau, x_mean"
        self.param_range = param_range
        self.noise_std = noise_std
        self.covariate_noise = covariate_noise  # Controls how similar x_i are within a segment

        self.true_segments = self._init_true_segments()
        self.customers = self._generate_customers()
        
        self.est_segments_list = {
            "gmm": [],
            "policy_tree": [],
            "oast": []
        }
        
        if DR_generation_method == "forest":
            self.gamma = self._generate_gamma_matrix_forest()
        elif DR_generation_method == "reg":
            self.gamma = self._generate_gamma_matrix_reg()
        
        self.train_customers, self.val_customers, self.train_indices, self.val_indices = None, None, None, None
        
            
    def _init_true_segments(self):
        pr = self.param_range  # alias for convenience
        true_segments = []
        while True:
            beta_force = []
            for k in range(self.K):
                alpha = np.random.uniform(*pr["alpha"])
                beta = np.random.uniform(*pr["beta"], size=self.d)
                tau = np.random.uniform(*pr["tau"])
                x_mean = np.random.uniform(*pr["x_mean"], size=self.d)
                true_segments.append(SegmentTrue(alpha, beta, tau, x_mean, segment_id=k))
                beta_force.append(beta)
            # Check if the beta vectors are sufficiently distinct
            beta_force = np.array(beta_force)
            if np.any(beta_force < 0) and np.any(beta_force > 0):
                break
        
        return true_segments



    def _generate_customers(self):
        customers = []
        cov = np.eye(self.d) * (self.covariate_noise ** 2)
        for i in range(self.N):
            segment = np.random.choice(self.true_segments)
            x = np.random.multivariate_normal(mean=segment.x_mean, cov=cov)
            D_i = np.random.binomial(1, 0.5)
            cust = Customer(x, segment, D_i, self.noise_std, customer_id=i)
            segment.count += 1
            customers.append(cust)
        return customers


    def _generate_gamma_matrix_reg(self):
        """
        Generate the doubly robust (DR) score matrix Gamma ∈ R^{N x 2} for each customer.
        Gamma[i, 0] = DR estimate under control (a=0)
        Gamma[i, 1] = DR estimate under treatment (a=1)
        """
        X = np.array([cust.x for cust in self.customers])
        D = np.array([cust.D_i for cust in self.customers])
        Y = np.array([cust.y for cust in self.customers])

        # Step 1: Fit separate outcome models μ̂₀(x), μ̂₁(x)
        model_0 = LinearRegression()
        model_1 = LinearRegression()

        X0, Y0 = X[D == 0], Y[D == 0]
        X1, Y1 = X[D == 1], Y[D == 1]
        model_0.fit(X0, Y0)
        model_1.fit(X1, Y1)

        mu_0_hat = model_0.predict(X)
        mu_1_hat = model_1.predict(X)

        # Step 2: Apply DR formula
        e = 0.5  # Propensity score

        gamma_1 = mu_1_hat + (D / e) * (Y - mu_1_hat)
        gamma_0 = mu_0_hat + ((1 - D) / (1 - e)) * (Y - mu_0_hat)

        Gamma = np.stack([gamma_0, gamma_1], axis=1)
        return Gamma


    def _generate_gamma_matrix_forest(self):
        """
        Generate the doubly robust (DR) score matrix Gamma ∈ R^{N x 2} using econml,
        assuming known propensity score e=0.5 (randomized treatment).
        Gamma[i, 0] = DR estimate under control (a=0)
        Gamma[i, 1] = DR estimate under treatment (a=1)
        """

        # Step 1: Extract data
        X = np.array([cust.x for cust in self.customers])
        D = np.array([cust.D_i for cust in self.customers]).reshape(-1,1)
        Y = np.array([cust.y for cust in self.customers]).reshape(-1,1)
        
        with localconverter(default_converter + numpy2ri.converter):
            X_r = ro.conversion.py2rpy(np.asarray(X))
            Y_r = ro.conversion.py2rpy(np.asarray(Y))
            D_r = ro.conversion.py2rpy(np.asarray(D))

        cforest = grf.causal_forest(X_r, Y_r, D_r)
        Gamma = policytree.double_robust_scores(cforest)
        Gamma = np.array(Gamma)

        return Gamma


    def split_customers_into_train_and_validate(self, train_frac=0.7, seed=None):
        np.random.seed(seed)
        indices = np.arange(self.N)
        
        np.random.shuffle(indices)

        split = int(self.N * train_frac)
        train_idx, val_idx = indices[:split], indices[split:]

        train_customers = [self.customers[i] for i in train_idx]
        val_customers = [self.customers[i] for i in val_idx]
        return train_customers, val_customers, train_idx, val_idx

    def to_dataframe(self):
        data = []
        for cust in self.customers:
            row = {
                'customer_id': cust.customer_id,
                'true_segment_id': cust.true_segment.segment_id,
                'gmm_est_segment_id': cust.est_segment['gmm'].segment_id if cust.est_segment['gmm'] else None,
                'policy_tree_est_segment_id': cust.est_segment['policy_tree'].segment_id if cust.est_segment['policy_tree'] else None,
                'oast_est_segment_id': cust.est_segment['oast'].segment_id if cust.est_segment['oast'] else None,
                'D_i': cust.D_i,
                'outcome': cust.y,
            }
            for j in range(self.d):
                row[f'x_{j}'] = cust.x[j]
            data.append(row)
        return pd.DataFrame(data)

    
    def get_true_parameters(self):
        return {
            seg.segment_id: {
                'alpha': seg.alpha,
                'beta': seg.beta,
                'tau': seg.tau,
            }
            for seg in self.true_segments
        }


