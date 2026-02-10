import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from econml.dr import DRLearner
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, default_converter
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from scipy.spatial.distance import pdist, squareform


ALGORITHMS = ["kmeans-standard", "kmeans-da", 
              "gmm-standard", "gmm-da", 
              "clr-standard", "clr-da",
              "policy_tree", 
              "mst", 
              "dast", 
              "t_learner",
              "x_learner",
              "s_learner",
              "dr_learner"]

grf = importr('grf')
policytree = importr('policytree')
ro.r('library(policytree)')

class SegmentTrue:
    def __init__(self, alpha, beta, tau, segment_id, x_mean=None, delta=None):
        self.alpha = alpha
        self.beta = beta  # np.array of shape (d,)
        self.tau = tau
        self.delta = delta  # np.array of shape (d,) for interaction terms
        self.x_mean = x_mean
        self.segment_id = segment_id
        self.action = int(tau>=0)

    def generate_outcome(self, x, D_i, noise_std, signal_d):
        noise = np.random.normal(0, noise_std)
        # Base outcome: y = alpha + beta @ x + tau * D + delta @ x * D + noise
        base = self.alpha + self.beta[:signal_d] @ x[:signal_d] + self.tau * D_i
        
        # Add interaction terms if delta is provided
        if self.delta is not None:
            interaction = (self.delta[:signal_d] * x[:signal_d]).sum() * D_i
            return base + interaction + noise
        return base + noise
        

        
class SegmentEstimate:
    def __init__(self, est_alpha, est_beta, est_tau, est_action, segment_id=None, est_delta=None):
        self.est_alpha = est_alpha
        self.est_beta = est_beta  # np.array of shape (d,)
        self.est_tau = est_tau
        self.est_delta = est_delta  # np.array of shape (d,) for interaction terms
        self.est_action = est_action  # 0 or 1
        self.segment_id = segment_id


class Customer_pilot:
    def __init__(self, x, D_i, y, true_segment: SegmentTrue,  customer_id=None):
        self.x = x
        self.D_i = D_i
        self.true_segment = true_segment
        self.y = y
        self.est_segment = {algo: None for algo in ALGORITHMS}

        self.customer_id = customer_id

class Customer_implement:
    def __init__(self, x, true_segment: SegmentTrue, noise_std, signal_d):
        self.x = x
        self.true_segment = true_segment
        self.noise_std = noise_std
        self.signal_d = signal_d
        
        self.est_segment = {algo: None for algo in ALGORITHMS}
        
    def evaluate_profits(self, algo, implement_action=None):
        
        # evaluate the profit for a single customer under algo
        if implement_action is None:
            if self.est_segment[algo].est_action != 404:
                self.implement_action = self.est_segment[algo].est_action 
            
            else: # 404 case
                self.implement_action = self.true_segment.action
        else:
            # for evaluating the meta-learner performance
            self.implement_action = implement_action
        
        # Option 1: No noise for deterministic profit evaluation (current)
        noise_std = 0 # set noise to 0 for profit evaluation
        # Option 2: Use actual noise (uncomment line below to enable)
        # noise_std = self.noise_std  
        self.y = self.true_segment.generate_outcome(self.x, self.implement_action, noise_std, self.signal_d)
        return self.y
        
# ----------------------------------------
# Population Simulator
# ----------------------------------------

class PopulationSimulator:
    def __init__(self, N_total_pilot_customers, N_total_implement_customers, d, K, disturb_covariate_noise, param_range, DR_generation_method, partial_x, X_mean_vectors=None, X_noise_std_scale=None, Y_noise_std_scale=None, disallowed_ball_radius=None):
        self.N_total_pilot_customers = N_total_pilot_customers
        self.N_total_implement_customers = N_total_implement_customers
        self.d = d
        self.K = K
        
        self.param_range = param_range
        self.disturb_covariate_noise = disturb_covariate_noise
        self.signal_d = d if partial_x == 0 else max(1, int(d * partial_x))  # number of features used in outcome generation
        self.disturb_d = d - self.signal_d  # number of features NOT used in outcome generation
        self.disallowed_ball_radius = disallowed_ball_radius if disallowed_ball_radius is not None else 0

        self.true_segments = self._init_true_segments(X_mean_vectors)
        
        # Compute signal_covariate_noise based on X_noise_std_scale (required parameter)
        if X_noise_std_scale is None:
            raise ValueError("X_noise_std_scale is required. Please provide a scale factor for within-cluster covariate noise.")
        
        if self.K <= 1:
            raise ValueError(f"Cannot use X_noise_std_scale with K={self.K}. Need at least 2 clusters to compute average distance.")
        
        # Extract signal dimensions of mean vectors (only first signal_d dimensions)
        mean_vectors_signal = np.array([seg.x_mean[:self.signal_d] for seg in self.true_segments])
        # Compute pairwise distances
        pairwise_distances = pdist(mean_vectors_signal, metric='euclidean')
        
        if len(pairwise_distances) == 0:
            raise ValueError(f"No pairwise distances computed for K={self.K} clusters. This should not happen. Please check the data.")
        
        # Average distance
        avg_distance = np.mean(pairwise_distances)
        # Set signal_covariate_noise as scale times average distance
        self.signal_covariate_noise = X_noise_std_scale * avg_distance
        print(f"Computed X_covariate_noise: {self.signal_covariate_noise:.4f} (scale={X_noise_std_scale}, avg_distance={avg_distance:.4f})")
        
        # NOW adjust tau for adjacent clusters (after signal_covariate_noise is set)
        self._adjust_adjacent_cluster_tau()
        
        # Compute noise_std based on Y_noise_std_scale (required parameter)
        if Y_noise_std_scale is None:
            raise ValueError("Y_noise_std_scale is required. Please provide a scale factor for outcome noise.")
        
        # Compute average treatment effect magnitude across segments
        tau_values = np.array([seg.tau for seg in self.true_segments])
        avg_tau_magnitude = np.mean(np.abs(tau_values))
        # Set noise_std as scale times average |tau|
        self.noise_std = Y_noise_std_scale * avg_tau_magnitude
        print(f"Computed Ynoise_std: {self.noise_std:.4f} (scale={Y_noise_std_scale}, avg_|tau|={avg_tau_magnitude:.4f})")
        
        self.pilot_customers = self._generate_pilot_customers()
        self.implement_customers = self._generate_implement_customers()
        
        self.est_segments_list = {algo: [] for algo in ALGORITHMS}
        
        self.gamma = self._generate_gamma_matrix(DR_generation_method)
        
        self.train_customers, self.val_customers, self.train_indices, self.val_indices = None, None, None, None
        
            
    def _init_true_segments(self, X_mean_vectors):
        pr = self.param_range  # alias for convenience
        true_segments = []
        
        # If generating mean vectors, use minimum distance constraint
        if X_mean_vectors is None:
            # Calculate a reasonable minimum distance based on space size
            space_range = pr["x_mean"][1] - pr["x_mean"][0]
            # Heuristic: min distance should be at least space_range / (K^(1/d))
            # This ensures clusters can be reasonably separated
            min_distance = (space_range / (self.K ** (1.0 / self.d))) * self.disallowed_ball_radius
            print(f"Generating mean vectors with minimum distance: {min_distance:.2f}")
            
            generated_means = []
            for k in range(self.K):
                alpha = np.random.uniform(*pr["alpha"])
                beta = np.random.uniform(*pr["beta"], size=self.d)
                tau = np.random.uniform(*pr["tau"])
                # Generate delta (interaction coefficients) if specified in param_range
                delta = np.random.uniform(*pr["delta"], size=self.d) if pr["delta"] is not None else None
                
                # Generate x_mean with minimum distance constraint
                max_attempts = 100
                for attempt in range(max_attempts):
                    x_mean_candidate = np.random.uniform(*pr["x_mean"], size=self.d)
                    
                    # Check distance to all existing means
                    if len(generated_means) == 0:
                        x_mean = x_mean_candidate
                        break
                    
                    distances = [np.linalg.norm(x_mean_candidate - existing) 
                                for existing in generated_means]
                    min_dist_to_existing = min(distances)
                    
                    if min_dist_to_existing >= min_distance:
                        x_mean = x_mean_candidate
                        break
                    
                    if attempt == max_attempts - 1:
                        print(f"⚠️  Warning: Could not find mean vector for cluster {k} satisfying min distance.")
                        print(f"   Using best candidate with distance {min_dist_to_existing:.2f}")
                        x_mean = x_mean_candidate
                
                generated_means.append(x_mean)
                true_segments.append(SegmentTrue(alpha, beta, tau, segment_id=k, x_mean=x_mean, delta=delta))
        else:
            # Use provided mean vectors
            for k in range(self.K):
                alpha = np.random.uniform(*pr["alpha"])
                beta = np.random.uniform(*pr["beta"], size=self.d)
                tau = np.random.uniform(*pr["tau"])
                # Generate delta (interaction coefficients) if specified in param_range
                delta = np.random.uniform(*pr["delta"], size=self.d) if pr["delta"] is not None else None
                x_mean = X_mean_vectors[k]
                true_segments.append(SegmentTrue(alpha, beta, tau, segment_id=k, x_mean=x_mean, delta=delta))
        
        return true_segments
    
    def _adjust_adjacent_cluster_tau(self):
        """
        Make sure "adjacent but not overlapping" clusters have opposite treatment effect signs.
        Adjacent means: distance ≈ 1-3 sigma (touching at boundaries, not fully overlapping).
        This creates a challenging scenario for algorithms.
        """
        if self.K < 2:
            return
        
        # Extract mean vectors (only signal dimensions matter for clustering)
        mean_vectors_signal = np.array([seg.x_mean[:self.signal_d] for seg in self.true_segments])
        
        # Compute pairwise distances in signal space
        dist_matrix = squareform(pdist(mean_vectors_signal, metric='euclidean'))
        
        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Define "adjacent" as distance in range [lower_bound, upper_bound]
        # This means clusters touch at boundaries but don't heavily overlap
        sigma = self.signal_covariate_noise
        lower_bound = 1.0 * sigma  # Closer than this = too much overlap
        upper_bound = 4.0 * sigma  # Farther than this = well separated
        
        # Find pairs in the "adjacent" range
        adjacent_pairs = []
        for i in range(self.K):
            for j in range(i+1, self.K):
                dist = dist_matrix[i, j]
                if lower_bound <= dist <= upper_bound:
                    adjacent_pairs.append((i, j, dist))
        
        if adjacent_pairs:
            # Pick the pair with distance closest to 2*sigma (sweet spot for "touching")
            target_dist = 2.0 * sigma
            best_pair = min(adjacent_pairs, key=lambda x: abs(x[2] - target_dist))
            idx1, idx2, dist = best_pair
            
            tau1 = self.true_segments[idx1].tau
            tau2 = self.true_segments[idx2].tau
            action1_before = self.true_segments[idx1].action
            action2_before = self.true_segments[idx2].action
            
            # If they have the same sign, flip one of them
            if tau1 * tau2 > 0:  # Same sign (both positive or both negative)
                # Flip the sign of tau2
                self.true_segments[idx2].tau = -tau2
                self.true_segments[idx2].action = int(self.true_segments[idx2].tau >= 0)
                
                action2_after = self.true_segments[idx2].action
                
                print(f"⚠️  Adjacent clusters (segments {idx1} and {idx2}) had same tau sign.")
                print(f"   Distance: {dist:.2f}, sigma={sigma:.2f}")
                print(f"   BEFORE flip: Seg{idx1} tau={tau1:+7.2f} action={action1_before}, Seg{idx2} tau={tau2:+7.2f} action={action2_before}")
                print(f"   AFTER  flip: Seg{idx1} tau={tau1:+7.2f} action={action1_before}, Seg{idx2} tau={-tau2:+7.2f} action={action2_after}")
            else:
                print(f"✓ Adjacent clusters (segments {idx1} and {idx2}) already have opposite tau signs.")
                print(f"   Distance: {dist:.2f}, sigma={sigma:.2f}")
                print(f"   Seg{idx1}: tau={tau1:+7.2f}, Seg{idx2}: tau={tau2:+7.2f}")

    def _generate_pilot_customers(self):
        pilot_customers = []

        cov_signal = np.eye(self.signal_d) * (self.signal_covariate_noise ** 2) 
        if self.disturb_d > 0:
            # === 1️⃣ Generate adversarial noise clusters ===
            cov_noise = np.eye(self.disturb_d) * (self.disturb_covariate_noise ** 2)
            N_noise_clusters = max(1, int(np.random.uniform(self.K - 5, self.K + 5)))
            noise_cluster_means = [
                np.random.uniform(*self.param_range["x_mean"], size=self.disturb_d)
                for _ in range(N_noise_clusters)
            ]
            noise_cluster_labels = np.random.randint(
                0, N_noise_clusters, size=self.N_total_pilot_customers
            )
        for i in range(self.N_total_pilot_customers):
            # 2️⃣ Choose true segment for signal part
            segment = np.random.choice(self.true_segments)

            # 3️⃣ Signal features: cluster-dependent
            x_signal = np.random.multivariate_normal(mean=segment.x_mean, cov=cov_signal)

            # 4️⃣ Disturbing features: come from random noise cluster
            if self.disturb_d > 0:
                w = noise_cluster_labels[i]
                x_noise = np.random.multivariate_normal(mean=noise_cluster_means[w], cov=cov_noise)
                x_full = np.concatenate([x_signal, x_noise])

            else:
                x_full = x_signal

            # 6️⃣ Outcome
            D_i = np.random.binomial(1, 0.5)
            y = segment.generate_outcome(x_full, D_i, self.noise_std, self.signal_d)

            # 7️⃣ Save
            cust = Customer_pilot(x_full, D_i, y, segment, customer_id=i)
            pilot_customers.append(cust)

        return pilot_customers


    
        
    def _generate_implement_customers(self):
        implement_customers = []

        # --- Signal and noise covariance ---
        cov_signal = np.eye(self.signal_d) * (self.signal_covariate_noise ** 2)
        

        # === 1️⃣  noise clusters ===
        if self.disturb_d > 0:
            cov_noise = np.eye(self.disturb_d) * (self.disturb_covariate_noise ** 2)
            N_noise_clusters = max(1, int(np.random.uniform(self.K - 5, self.K + 5)))

            # Randomly position these noise cluster means
            noise_cluster_means = [
                np.random.uniform(*self.param_range["x_mean"], size=self.disturb_d)
                for _ in range(N_noise_clusters)
            ]

            # Assign each implement customer to a random noise cluster
            noise_cluster_labels = np.random.randint(
                0, N_noise_clusters, size=self.N_total_implement_customers
            )

        for i in range(self.N_total_implement_customers):
            # 2️⃣ Choose true signal segment (same as before)
            segment = np.random.choice(self.true_segments)

            # 3️⃣ Signal part: segment-dependent
            x_signal = np.random.multivariate_normal(mean=segment.x_mean, cov=cov_signal)

            # 4️⃣ Noise part: from an unrelated latent noise cluster
            if self.disturb_d > 0:
                w = noise_cluster_labels[i]
                x_noise = np.random.multivariate_normal(mean=noise_cluster_means[w], cov=cov_noise)

                # 5️⃣ Combine signal + noise
                x_full = np.concatenate([x_signal, x_noise])
            else:
                x_full = x_signal

            # 6️⃣ Create Customer_implement object
            cust = Customer_implement(x_full, segment, self.noise_std, self.signal_d)
            implement_customers.append(cust)

        return implement_customers




    # ---------- NEW: overlap function ----------
    def compute_covariate_overlap(self):
        """Compute average Bhattacharyya coefficient of X distributions across true segments."""
        Sigma = np.eye(self.d) * (self.signal_covariate_noise ** 2)   # dxd covariance matrix
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

        return np.mean(bcs) if bcs else 1.0

    def compute_outcome_overlap(self):
        """Compute average Bhattacharyya coefficient of Y distributions across true segments."""
        Sigma = np.eye(self.d) * (self.signal_covariate_noise ** 2)
        sigma = self.noise_std
        K = self.K

        def y_params(seg, D):
            mu = seg.x_mean
            beta = seg.beta
            alpha = seg.alpha
            tau = seg.tau
            delta = seg.delta if seg.delta is not None else np.zeros(self.d)
            # Mean includes interaction term: alpha + beta @ mu + tau * D + delta @ mu * D
            m_y = alpha + beta @ mu + tau * D + (delta @ mu) * D
            # Variance includes interaction term contributions
            v_y = beta @ Sigma @ beta + (delta @ Sigma @ delta) * (D**2) + 2 * D * (beta @ Sigma @ delta) + sigma**2
            return m_y, v_y

        def bc_univariate(m1, v1, m2, v2):
            term1 = 0.25 * np.log(0.25 * ((v1/v2) + (v2/v1) + 2))
            term2 = 0.25 * ((m1 - m2)**2) / (v1 + v2)
            BD = term1 + term2
            BC = float(np.exp(-BD))
            return float(np.clip(BC, 0.0, 1.0))  # ensure in [0,1]

        bcs = []
        for k in range(K):
            for l in range(k+1, K):
                bc_Ds = []
                for D in (0, 1):
                    m1, v1 = y_params(self.true_segments[k], D)
                    m2, v2 = y_params(self.true_segments[l], D)
                    bc_Ds.append(bc_univariate(m1, v1, m2, v2))
                bcs.append(0.5 * sum(bc_Ds))  # average over D
        return np.mean(bcs) if bcs else 1.0


    def compute_joint_overlap(self):
        """Compute average Bhattacharyya coefficient of (x,y) distributions across true segments."""
        Sigma = np.eye(self.d) * (self.signal_covariate_noise ** 2)
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
            delta = seg.delta if seg.delta is not None else np.zeros(self.d)
            # Mean of Y includes interaction: alpha + beta @ mu + tau * D + delta @ mu * D
            m_y = alpha + beta @ mu + tau * D + (delta @ mu) * D
            m = np.concatenate([mu, [m_y]])
            # Covariance: cov(X, Y) = Sigma @ (beta + D * delta)
            cov_xy = Sigma @ (beta + D * delta)
            # Variance of Y: var(Y) = (beta + D*delta)^T @ Sigma @ (beta + D*delta) + sigma^2
            var_y = (beta + D * delta) @ Sigma @ (beta + D * delta) + sigma**2
            S = np.block([
                [Sigma,                     cov_xy.reshape(-1,1)],
                [cov_xy.reshape(1,-1),      var_y]
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
        return np.mean(bcs) if bcs else 1.0
    

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
        Sigma = np.eye(d) * (self.signal_covariate_noise ** 2)
        Sigma_inv = np.linalg.inv(Sigma)
        logdet_Sigma = np.log(np.linalg.det(Sigma))
        sigma2 = self.noise_std ** 2

        # Precompute segment params
        mus = [seg.x_mean for seg in self.true_segments]
        alphas = [seg.alpha for seg in self.true_segments]
        betas = [seg.beta for seg in self.true_segments]
        taus = [seg.tau for seg in self.true_segments]
        deltas = [seg.delta if seg.delta is not None else np.zeros(d) for seg in self.true_segments]

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
                delta = deltas[k]

                # log p(x | Z=k)
                dx = x - mu
                llx = -0.5 * (d * np.log(2*np.pi) + logdet_Sigma + dx @ Sigma_inv @ dx)

                # log p(y | x, D, Z=k) - includes interaction term
                mean_y = alpha + beta @ x + tau * d_i + (delta @ x) * d_i
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
            )
            model_1 = MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                max_iter=10000,
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
                'delta': seg.delta if seg.delta is not None else np.zeros(self.d),
            }
            for seg in self.true_segments
        }








