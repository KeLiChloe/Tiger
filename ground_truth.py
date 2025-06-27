import numpy as np
import pandas as pd


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
    def __init__(self, est_alpha, est_beta, est_tau, est_action, segment_id=None):
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

        # self.gmm_est_segment = None
        # self.policy_tree_est_segment = None
        # self.oast_est_segment = None
        self.y = true_segment.generate_outcome(x, D_i, noise_std)
        self.customer_id = customer_id
        
# ----------------------------------------
# Population Simulator
# ----------------------------------------

class PopulationSimulator:
    def __init__(self, N, d, K, covariate_noise, param_range, noise_std):
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
        # self.gmm_est_segments = None
        # self.policy_tree_est_segments = None
        # self.oast_est_segments = None  
        
        

    def _init_true_segments(self):
        pr = self.param_range  # alias for convenience
        true_segments = []
        for k in range(self.K):
            alpha = np.random.uniform(*pr["alpha"])
            beta = np.random.uniform(*pr["beta"], size=self.d)
            tau = np.random.uniform(*pr["tau"])
            x_mean = np.random.uniform(*pr["x_mean"], size=self.d)
            true_segments.append(SegmentTrue(alpha, beta, tau, x_mean, segment_id=k))
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


