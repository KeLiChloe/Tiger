import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from econml.dr import DRLearner
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, default_converter
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from scipy.spatial.distance import pdist


ALGORITHMS = ["gmm-standard", "gmm-da", "dast", "policy_tree", "mst", "kmeans-standard", "kmeans-da", "clr-standard", "clr-da"]

grf = importr('grf')
policytree = importr('policytree')
ro.r('library(policytree)')

class SegmentTrue:
    def __init__(self, alpha, beta, tau, segment_id, x_mean=None):
        self.alpha = alpha
        self.beta = beta  # np.array of shape (d,)
        self.tau = tau
        self.x_mean = x_mean
        self.segment_id = segment_id
        self.action = int(tau>0)

    def generate_outcome(self, x, D_i, noise_std):
        noise = np.random.normal(0, noise_std)
        return self.alpha + self.beta @ x + self.tau * D_i + noise
        

        
class SegmentEstimate:
    def __init__(self, est_alpha, est_beta, est_tau, est_action, segment_id=None):
        self.est_alpha = est_alpha
        self.est_beta = est_beta  # np.array of shape (d,)
        self.est_tau = est_tau
        self.est_action = est_action  # 0 or 1
        self.segment_id = segment_id


class Customer_pilot:
    def __init__(self, x, D_i, y, true_segment: SegmentTrue, customer_id=None):
        self.x = x
        self.D_i = D_i
        self.true_segment = true_segment
        self.y = y
        self.est_segment = {algo: None for algo in ALGORITHMS}

        
        self.customer_id = customer_id

class Customer_implement:
    def __init__(self, x, true_segment: SegmentTrue, noise_std):
        self.x = x
        self.true_segment = true_segment
        self.noise_std = noise_std
        
        self.est_segment = {algo: None for algo in ALGORITHMS}
        
    def evaluate_profits(self, algo):
        # evaluate the profit for a single customer under algo
        if self.est_segment[algo].est_action != 404:
            self.implement_action = self.est_segment[algo].est_action 
        else: # 404 case : only one treatment
            self.implement_action = self.true_segment.action
        
        self.y = self.true_segment.generate_outcome(self.x, self.implement_action, self.noise_std)
        return self.y
        
# ----------------------------------------
# Population Simulator
# ----------------------------------------

class PopulationSimulator:
    def __init__(self, N_total_pilot_customers, N_total_implement_customers, d, K, covariate_noise, param_range, noise_std, DR_generation_method):
        self.N_total_pilot_customers = N_total_pilot_customers
        self.N_total_implement_customers = N_total_implement_customers
        self.d = d
        self.K = K
        
        self.param_range = param_range
        self.noise_std = noise_std
        self.covariate_noise = covariate_noise  # Controls how similar x_i are within a segment

        self.true_segments = self._init_true_segments()
        self.pilot_customers = self._generate_pilot_customers()
        self.implement_customers = self._generate_implement_customers()
        
        self.est_segments_list = {algo: [] for algo in ALGORITHMS}
        
        self.gamma = self._generate_gamma_matrix(DR_generation_method)
        
        self.train_customers, self.val_customers, self.train_indices, self.val_indices = None, None, None, None
        
            
    def _init_true_segments(self):
        pr = self.param_range  # alias for convenience
        true_segments = []
        for k in range(self.K):
            alpha = np.random.uniform(*pr["alpha"])
            beta = np.random.uniform(*pr["beta"], size=self.d)
            tau = np.random.uniform(*pr["tau"])
            x_mean = np.random.uniform(*pr["x_mean"], size=self.d)
            true_segments.append(SegmentTrue(alpha, beta, tau, segment_id=k, x_mean=x_mean))
        
        return true_segments

        # while True:
        #     beta_check = []
        #     tau_check = []
        #     true_segments = []
        #     for k in range(self.K):
        #         alpha = np.random.uniform(*pr["alpha"])
        #         beta = np.random.uniform(*pr["beta"], size=self.d)
        #         tau = np.random.uniform(*pr["tau"])
        #         x_mean = np.random.uniform(*pr["x_mean"], size=self.d)
        #         true_segments.append(SegmentTrue(alpha, beta, tau, segment_id=k, x_mean=x_mean))
        #         beta_check.append(beta)
        #         tau_check.append(tau)
            
        #     # Check if the beta vectors are sufficiently distinct and not all tau are positive nor negative
        #     beta_check = np.array(beta_check)
        #     tau_check = np.array(tau_check)
        #     if np.min(pdist(beta_check, metric="euclidean")) > 3 and not np.all(tau_check > 0) and not np.all(tau_check < 0):
        #         return true_segments


    def _generate_pilot_customers(self):
        pilot_customers = []
        cov = np.eye(self.d) * (self.covariate_noise ** 2)
        for i in range(self.N_total_pilot_customers):
            segment = np.random.choice(self.true_segments)
            x = np.random.multivariate_normal(mean=segment.x_mean, cov=cov)
            D_i = np.random.binomial(1, 0.5)
            y = segment.generate_outcome(x, D_i, self.noise_std)
            cust = Customer_pilot(x, D_i, y, segment, customer_id=i)
            pilot_customers.append(cust)
        return pilot_customers
    
    def _generate_implement_customers(self):
        implement_customers = []
        cov = np.eye(self.d) * (self.covariate_noise ** 2)
        for _ in range(self.N_total_implement_customers):
            segment = np.random.choice(self.true_segments)
            x = np.random.multivariate_normal(mean=segment.x_mean, cov=cov)
            cust = Customer_implement(x, segment, self.noise_std)
            implement_customers.append(cust)
        return implement_customers
    
    # ---------- NEW: overlap function ----------
    def compute_covariate_overlap(self):
        """Compute average Bhattacharyya coefficient of X distributions across true segments."""
        Sigma = np.eye(self.d) * (self.covariate_noise ** 2)   # dxd covariance matrix
        Sigma_inv = np.linalg.inv(Sigma)                       # its inverse
        K = self.K

        bcs = []
        for k in range(K):
            for l in range(k+1, K):
                mu_k = self.true_segments[k].x_mean
                mu_l = self.true_segments[l].x_mean
                dm = mu_k - mu_l                               # difference vector
                exponent = -0.125 * float(dm.T @ Sigma_inv @ dm)   # scalar
                bc = np.exp(exponent)                          # should be <= 1
                bc = float(np.clip(bc, 0.0, 1.0))              # enforce [0,1]
                bcs.append(bc)

        return 2 * np.mean(bcs) if bcs else 1.0


    def compute_joint_overlap(self):
        """Compute average Bhattacharyya coefficient of (x,y) distributions across true segments."""
        Sigma = np.eye(self.d) * (self.covariate_noise ** 2)
        sigma = self.noise_std
        K = self.K

        def bhattacharyya_gaussian(m1, S1, m2, S2):
            S = 0.5 * (S1 + S2)
            dm = (m1 - m2).reshape(-1, 1)
            eps = 1e-9
            S_inv = np.linalg.inv(S + eps * np.eye(S.shape[0]))
            term1 = 0.125 * float(dm.T @ S_inv @ dm)
            term2 = 0.5 * np.log(np.linalg.det(S + eps*np.eye(S.shape[0])) /
                                 np.sqrt(np.linalg.det(S1 + eps*np.eye(S1.shape[0])) * np.linalg.det(S2 + eps*np.eye(S2.shape[0]))))
            BD = term1 + term2
            return float(np.exp(-BD))  # Bhattacharyya coefficient

        def mean_cov_for(seg, D):
            mu = seg.x_mean
            beta = seg.beta
            alpha = seg.alpha
            tau = seg.tau
            m = np.concatenate([mu, [alpha + beta @ mu + tau * D]])
            S = np.block([
                [Sigma,             Sigma @ beta.reshape(-1,1)],
                [beta.reshape(1,-1) @ Sigma,  beta @ Sigma @ beta + sigma**2]
            ])
            return m, S

        bcs = []
        for k in range(K):
            for l in range(k+1, K):
                bc_Ds = []
                for D in (0, 1):
                    m1, S1 = mean_cov_for(self.true_segments[k], D)
                    m2, S2 = mean_cov_for(self.true_segments[l], D)
                    bc_Ds.append(bhattacharyya_gaussian(m1, S1, m2, S2))
                bcs.append(0.5 * sum(bc_Ds))  # average over D
        return 2 * np.mean(bcs) if bcs else 1.0
    

    def compute_assignment_ambiguity(self):
        """
        Compute normalized assignment ambiguity score (posterior entropy over segments).
        Returns a scalar in [0,1].
        
        X: array (N,d) covariates
        D: array (N,) treatments {0,1}
        Y: array (N,) outcomes
        """
        X = np.array([cust.x for cust in self.pilot_customers])
        D = np.array([cust.D_i for cust in self.pilot_customers])
        Y = np.array([cust.y for cust in self.pilot_customers])
        
        N, d = X.shape
        K = self.K
        Sigma = np.eye(d) * (self.covariate_noise ** 2)
        Sigma_inv = np.linalg.inv(Sigma)
        logdet_Sigma = np.log(np.linalg.det(Sigma))
        sigma2 = self.noise_std ** 2

        # Precompute segment params
        mus = [seg.x_mean for seg in self.true_segments]
        alphas = [seg.alpha for seg in self.true_segments]
        betas = [seg.beta for seg in self.true_segments]
        taus = [seg.tau for seg in self.true_segments]

        total_entropy = 0.0
        for i in range(N):
            x = X[i]
            d_i = D[i]
            y = Y[i]

            logliks = []
            for k in range(K):
                mu = mus[k]
                alpha = alphas[k]
                beta = betas[k]
                tau = taus[k]

                # log p(x | Z=k)
                dx = x - mu
                llx = -0.5 * (d * np.log(2*np.pi) + logdet_Sigma + dx @ Sigma_inv @ dx)

                # log p(y | x, D, Z=k)
                mean_y = alpha + beta @ x + tau * d_i
                lly = -0.5 * (np.log(2*np.pi*sigma2) + (y - mean_y)**2 / sigma2)

                logliks.append(llx + lly)  # equal priors

            # Normalize with log-sum-exp trick
            M = max(logliks)
            w = np.exp(np.array(logliks) - M)
            probs = w / w.sum()

            # entropy of p(Z | x,y,D)
            entropy_i = -(probs * np.log(probs + 1e-12)).sum()
            total_entropy += entropy_i

        avg_entropy = total_entropy / N
        norm_entropy = avg_entropy / np.log(K)  # normalized to [0,1]
        return norm_entropy


    def _generate_gamma_matrix(self, method):
        """
        Generate the doubly robust (DR) score matrix Gamma ∈ R^{N x 2} for each customer.
        Gamma[i, 0] = DR estimate under control (a=0)
        Gamma[i, 1] = DR estimate under treatment (a=1)
        """
        X = np.array([cust.x for cust in self.pilot_customers])
        D = np.array([cust.D_i for cust in self.pilot_customers])
        Y = np.array([cust.y for cust in self.pilot_customers])
        
        X0, Y0 = X[D == 0], Y[D == 0]
        X1, Y1 = X[D == 1], Y[D == 1]

        if method == "reg":
            model_0 = LinearRegression()
            model_1 = LinearRegression()

            
            model_0.fit(X0, Y0)
            model_1.fit(X1, Y1)

            mu_0_hat = model_0.predict(X)
            mu_1_hat = model_1.predict(X)

            e = 0.5  # Propensity score

            gamma_1 = mu_1_hat + (D / e) * (Y - mu_1_hat)
            gamma_0 = mu_0_hat + ((1 - D) / (1 - e)) * (Y - mu_0_hat)

            Gamma = np.stack([gamma_0, gamma_1], axis=1)
        
        if method == "forest":
            with localconverter(default_converter + numpy2ri.converter):
                X_r = ro.conversion.py2rpy(X)
                Y_r = ro.conversion.py2rpy(Y)
                D_r = ro.conversion.py2rpy(D)
            ro.globalenv['Y'] = Y_r
            ro.globalenv['D'] = D_r
            Y_r = ro.r('as.numeric(Y)')
            D_r = ro.r('as.numeric(D)')

            cforest = grf.causal_forest(X_r, Y_r, D_r)
            Gamma_r = policytree.double_robust_scores(cforest)

            with localconverter(default_converter + numpy2ri.converter):
                Gamma = ro.conversion.rpy2py(Gamma_r)

        
        if method == "mlp":
            
            model_0 = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                max_iter=10000,
                random_state=42
            )
            model_1 = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                max_iter=10000,
                random_state=42
            )
            
            model_0.fit(X0, Y0)
            model_1.fit(X1, Y1)

            mu_0_hat = model_0.predict(X)
            mu_1_hat = model_1.predict(X)
            
            e = 0.5  # Propensity score

            gamma_1 = mu_1_hat + (D / e) * (Y - mu_1_hat)
            gamma_0 = mu_0_hat + ((1 - D) / (1 - e)) * (Y - mu_0_hat)

            Gamma = np.stack([gamma_0, gamma_1], axis=1)
            
        return Gamma


    def split_pilot_customers_into_train_and_validate(self, train_frac=0.8):
        indices = np.arange(self.N_total_pilot_customers)

        split = int(self.N_total_pilot_customers * train_frac)
        self.train_idx, self.val_idx = indices[:split], indices[split:]

        self.train_customers = [self.pilot_customers[i] for i in self.train_idx]
        self.val_customers = [self.pilot_customers[i] for i in self.val_idx]

    def to_dataframe(self):
        data = []
        for cust in self.pilot_customers:
            row = {
                "customer_id": cust.customer_id,
                "true_segment_id": cust.true_segment.segment_id,
                "D_i": cust.D_i,
                "outcome": cust.y,
            }

            # 动态添加每个算法的 est_segment_id
            for algo in ALGORITHMS:
                row[f"{algo}_est_segment_id"] = (
                    cust.est_segment[algo].segment_id if cust.est_segment[algo] is not None else None
                )
            # 添加特征向量 x
            for j in range(self.d):
                row[f"x_{j}"] = cust.x[j]

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








