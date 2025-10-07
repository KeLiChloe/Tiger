class PopulationConstructor:
    # TODO:Only works for 1-D now!!!!
    def __init__(self, x_vec, D_vec, y_vec, alpha, beta, tau, cluster_sizes, DR_generation_method):
        assert x_vec.shape[0] == D_vec.shape[0] and x_vec.shape[0] == y_vec.shape[0] 
        assert len(alpha) == len(beta) and len(alpha) == len(tau)
        assert sum(cluster_sizes) == x_vec.shape[0]
        self.N_total_pilot_customers = x_vec.shape[0]
        self.d = x_vec.shape[1]  # dimensionality of covariates
        self.K = len(alpha)  # number of segments
        self.cluster_sizes = cluster_sizes

        self.true_segments = self._init_true_segments(alpha, beta, tau)
        self.pilot_customers = self._generate_pilot_customers(x_vec, D_vec, y_vec)
        
        self.est_segments_list = {
            "gmm": [],
            "policy_tree": [],
            "dast": []
        }
        
        self.gamma = self._generate_gamma_matrix(DR_generation_method)
        
        self.train_customers, self.val_customers, self.train_indices, self.val_indices = None, None, None, None
        
            
    def _init_true_segments(self, alpha, beta, tau):
        true_segments = []
        for k in range(self.K):
            true_segments.append(SegmentTrue(alpha[k], beta[k], tau[k], segment_id=k))
        return true_segments

    def _generate_pilot_customers(self, x_vec, D_vec, y_vec):
        pilot_customers = []
        start = 0

        for k, size in enumerate(self.cluster_sizes):
            end = start + size
            for i in range(start, end):
                x_i = x_vec[i]
                D_i = D_vec[i]
                y_i = y_vec[i]
                segment = self.true_segments[k]  # deterministically assigned
                cust = Customer_pilot(x_i, D_i, y_i, segment, customer_id=i)
                pilot_customers.append(cust)
            start = end

        return pilot_customers


    def split_pilot_customers_into_train_and_validate(self, train_frac=0.8, seed=None):
        np.random.seed(seed)
        indices = np.arange(self.N_total_pilot_customers)

        split = int(self.N_total_pilot_customers * train_frac)
        self.train_idx, self.val_idx = indices[:split], indices[split:]

        self.train_customers = [self.pilot_customers[i] for i in self.train_idx]
        self.val_customers = [self.pilot_customers[i] for i in self.val_idx]

    def to_dataframe(self):

        data = []
        for cust in self.pilot_customers:
            row = {
                'customer_id': cust.customer_id,
                'true_segment_id': cust.true_segment.segment_id,
                'gmm_est_segment_id': cust.est_segment['gmm'].segment_id if cust.est_segment['gmm'] else None,
                'policy_tree_est_segment_id': cust.est_segment['policy_tree'].segment_id if cust.est_segment['policy_tree'] else None,
                'dast_est_segment_id': cust.est_segment['dast'].segment_id if cust.est_segment['dast'] else None,
                'D_i': cust.D_i,
                'outcome': cust.y,
            }
            for j in range(self.d):
                row[f'x_{j}'] = cust.x[j]
            data.append(row)
        return pd.DataFrame(data)
