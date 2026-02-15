import numpy as np
from utils import assign_trained_customers_to_segments
from ground_truth import PopulationSimulator, SegmentEstimate
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression

def T_learner(implement_customers, x_mat, D_vec, y_vec):
    """
    T-Learner implementation using neural networks.
    Supports multiple actions (multi-arm treatment).
    
    Fits separate outcome models for each action arm:
    μ̂_a(x) = E[Y | X=x, D=a] for each action a
    
    Parameters:
    implement_customers : list
        List of implementation customers.
    x_mat : array-like, shape (n_samples, n_features)
        Feature matrix from pilot/training data.
    D_vec : array-like, shape (n_samples,)
        Action/treatment assignment vector (can be any integer: 0, 1, 2, ..., k-1).
    y_vec : array-like, shape (n_samples,)
        Outcome vector from pilot/training data.
        
    Returns:
    seg_labels_impl : array-like, shape (n_impl_samples,)
        Recommended action for each implementation customer (argmax of predicted outcomes).
    action_identity : array-like, shape (n_actions,)
        Action identity mapping: action_identity[i] = i (segment i recommends action i).
    """
    import numpy as np
    from sklearn.neural_network import MLPRegressor
    
    # Identify all unique actions in the data
    unique_actions = np.unique(D_vec)
    n_actions = len(unique_actions)
    
    
    # Train a separate model for each action
    mu_pilot_models = {}
    for action in unique_actions:
        # Get data for this action
        X_a = x_mat[D_vec == action]
        Y_a = y_vec[D_vec == action]
        
        if len(X_a) == 0:
            print(f"Warning: No samples for action {action}, skipping")
            continue
        
        # Train model for this action
        model_a = MLPRegressor(
            hidden_layer_sizes=(32, ),
            activation='relu',
             
            max_iter=2000,
            early_stopping=True,
        )
        model_a.fit(X_a, Y_a)
        mu_pilot_models[int(action)] = model_a
    
    # Predict outcomes for implementation customers under all actions
    X_impl = np.array([cust.x for cust in implement_customers])
    mu_mat_impl = _build_mu_matrix(mu_pilot_models, X_impl)  # shape: (n_impl_customers, n_actions)
    
    # Recommend action with highest predicted outcome
    seg_labels_impl = np.argmax(mu_mat_impl, axis=1).astype(int)
    # Use unique_actions instead of np.arange to handle non-consecutive actions
    action_identity = unique_actions.astype(int)
    
    return seg_labels_impl, action_identity
    

def _build_mu_matrix(mu_models, X_impl):
    """
    Build matrix of predicted outcomes for each action.
    
    Returns:
    mu_mat : array, shape (n, n_actions)
        mu_mat[:, i] contains predictions for action i (using enumeration index, not action value)
    """
    n = X_impl.shape[0]
    n_actions = len(mu_models)
    mu_mat = np.zeros((n, n_actions), dtype=float)
    
    # Sort actions to ensure consistent column ordering
    sorted_actions = sorted(mu_models.keys())
    for i, action in enumerate(sorted_actions):
        model = mu_models[action]
        pred = model.predict(X_impl)
        mu_mat[:, i] = pred

    return mu_mat


def X_learner(implement_customers, x_mat, D_vec, y_vec):
    """
    X-Learner implementation using neural networks.
    Supports multiple actions (multi-arm treatment).
    
    The X-learner is particularly effective when action groups are imbalanced.
    
    Multi-action strategy:
    - Uses action 0 as the baseline/reference action
    - For each other action a, estimates treatment effect relative to baseline
    - Predicts outcome under each action and recommends the best one
    
    Steps for each action a vs baseline 0:
    1. Fit outcome models: μ̂ₐ(x) and μ̂₀(x)
    2. Construct pseudo-treatment effects:
       - τ̃ₐ = Y - μ̂₀(X) for D=a (observed - baseline expectation)
       - τ̃₀ₐ = μ̂ₐ(X) - Y for D=0 (action a expectation - observed baseline)
    3. Fit effect models: τ̂ₐ(x) and τ̂₀ₐ(x)
    4. Combine with weighting
    
    Parameters:
    implement_customers : list
        List of implementation customers.
    x_mat : array-like, shape (n_samples, n_features)
        Feature matrix from pilot/training data.
    D_vec : array-like, shape (n_samples,)
        Action assignment vector (0, 1, 2, ..., k-1) from pilot/training data.
    y_vec : array-like, shape (n_samples,)
        Outcome vector from pilot/training data.
    propensity_scores : dict, optional
        Dictionary mapping action -> propensity scores. If None, uses equal weighting.
        
    Returns:
    seg_labels_impl : array-like, shape (n_impl_samples,)
        Recommended action for each implementation customer.
    action_identity : array-like, shape (n_actions,)
        Action identity mapping
    """
    import numpy as np
    from sklearn.neural_network import MLPRegressor
    
    # Identify all unique actions
    unique_actions = np.unique(D_vec)
    n_actions = len(unique_actions)
    
    # Ensure action 0 (baseline) exists
    if 0 not in unique_actions:
        raise ValueError("X-learner requires action 0 as baseline/reference action")
    
    
    # Get baseline (action 0) data
    X_baseline = x_mat[D_vec == 0]
    Y_baseline = y_vec[D_vec == 0]
    
    # ========== Stage 1: Fit outcome model for baseline ==========
    
    model_mu_baseline = MLPRegressor(
        hidden_layer_sizes=(32,),
        activation='relu',
        max_iter=2000,
        early_stopping=True,
    )
    model_mu_baseline.fit(X_baseline, Y_baseline)
    
    # Store treatment effect models for each action vs baseline
    tau_models_from_action = {}  # Models trained on action a data
    tau_models_from_baseline = {}  # Models trained on baseline data
    
    # ========== For each non-baseline action, estimate effect vs baseline ==========
    for action in unique_actions:
        if action == 0:
            continue  # Skip baseline
        
        
        # Get data for this action
        X_action = x_mat[D_vec == action]
        Y_action = y_vec[D_vec == action]
        
        if len(X_action) == 0:
            print(f"    Warning: No samples for action {action}, skipping")
            continue
        
        # ========== Stage 1: Fit outcome model for this action ==========
        
        model_mu_action = MLPRegressor(
            hidden_layer_sizes=(32,),
            activation='relu',
             
            max_iter=2000,
            early_stopping=True,
        )
        model_mu_action.fit(X_action, Y_action)
        
        # ========== Stage 2: Compute pseudo-treatment effects ==========
        
        # For action a units: τ̃ₐ = Y - μ̂₀(X)
        # (observed under action a minus predicted baseline)
        mu_baseline_on_action = model_mu_baseline.predict(X_action)
        tau_tilde_action = Y_action - mu_baseline_on_action
        
        # For baseline units: τ̃₀ₐ = μ̂ₐ(X) - Y
        # (predicted under action a minus observed baseline)
        mu_action_on_baseline = model_mu_action.predict(X_baseline)
        tau_tilde_baseline_to_action = mu_action_on_baseline - Y_baseline
        
        # ========== Stage 3: Fit treatment effect models ==========
        
        # Model trained on action a data
        model_tau_from_action = MLPRegressor(
            hidden_layer_sizes=(32,),
            activation='relu',
             
            max_iter=2000,
            early_stopping=True,
        )
        model_tau_from_action.fit(X_action, tau_tilde_action)
        tau_models_from_action[action] = model_tau_from_action
        
        # Model trained on baseline data
        model_tau_from_baseline = MLPRegressor(
            hidden_layer_sizes=(32,),
            activation='relu',
             
            max_iter=2000,
            early_stopping=True,
        )
        model_tau_from_baseline.fit(X_baseline, tau_tilde_baseline_to_action)
        tau_models_from_baseline[action] = model_tau_from_baseline
    
    # ========== Stage 4: Predict for implementation customers ==========
    
    X_impl = np.array([cust.x for cust in implement_customers])
    n_impl = X_impl.shape[0]
    
    # Predict baseline outcome
    mu_baseline_impl = model_mu_baseline.predict(X_impl)
    
    # Build outcome matrix: predict Y for each (customer, action) pair
    outcome_matrix = np.zeros((n_impl, n_actions))
    outcome_matrix[:, 0] = mu_baseline_impl  # Baseline outcomes
    
    # For each non-baseline action, predict outcome
    for i, action in enumerate(unique_actions):
        if action == 0:
            continue
        
        if action not in tau_models_from_action:
            # No model for this action, use baseline
            outcome_matrix[:, i] = mu_baseline_impl
            continue
        
        # Predict treatment effect from both models
        tau_from_action = tau_models_from_action[action].predict(X_impl)
        tau_from_baseline = tau_models_from_baseline[action].predict(X_impl)
        
        # Weighted combination (using equal weights for simplicity)
        # In practice, could use propensity scores
        weight = 1/n_actions
        tau_combined = weight * tau_from_baseline + (1 - weight) * tau_from_action
        
        # Predicted outcome under action a = baseline outcome + treatment effect
        outcome_matrix[:, i] = mu_baseline_impl + tau_combined
    
    # Recommend action with highest predicted outcome
    seg_labels_impl = np.argmax(outcome_matrix, axis=1).astype(int)
    # Use unique_actions instead of np.arange to handle non-consecutive actions
    action_identity = unique_actions.astype(int)
    
    
    return seg_labels_impl, action_identity


def S_learner(implement_customers, x_mat, D_vec, y_vec):
    """
    S-Learner implementation using neural networks.
    Supports multiple actions (multi-arm treatment).
    
    Fits a single outcome model that takes both covariates and action as input: Y = f(X, D)
    For multi-action case, uses one-hot encoding of actions.
    
    Parameters:
    implement_customers : list
        List of implementation customers.
    x_mat : array-like, shape (n_samples, n_features)
        Feature matrix from pilot/training data.
    D_vec : array-like, shape (n_samples,)
        Action/treatment assignment vector (can be any integer: 0, 1, 2, ..., k-1).
    y_vec : array-like, shape (n_samples,)
        Outcome vector from pilot/training data.
        
    Returns:
    seg_labels_impl : array-like, shape (n_impl_samples,)
        Recommended action for each implementation customer (argmax of predicted outcomes).
    action_identity : array-like, shape (n_actions,)
        Action identity mapping: action_identity[i] = i (segment i recommends action i).
    """
    import numpy as np
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import OneHotEncoder
    
    # Identify all unique actions in the data
    unique_actions = np.unique(D_vec)
    n_actions = len(unique_actions)
    
    
    # Use one-hot encoding for actions if more than 2 actions
    # For binary case, keep it simple as scalar
    if n_actions == 2:
        # Binary case: use scalar representation (0/1)
        D_encoded = D_vec.reshape(-1, 1)
    else:
        # Multi-action case: use one-hot encoding
        encoder = OneHotEncoder(sparse_output=False, categories=[unique_actions])
        D_encoded = encoder.fit_transform(D_vec.reshape(-1, 1))
    
    # Create augmented feature matrix [X, D_encoded]
    X_D = np.hstack([x_mat, D_encoded])
    
    # Fit a single model on combined data
    model = MLPRegressor(
        hidden_layer_sizes=(32,),
        activation='relu',
         
        max_iter=2000,
        early_stopping=True,
    )
    
    model.fit(X_D, y_vec)
    
    # For implementation customers, predict outcomes under all possible actions
    X_impl = np.array([cust.x for cust in implement_customers])
    n_impl = X_impl.shape[0]
    
    # Build outcome matrix: predict Y for each (customer, action) pair
    mu_mat_impl = np.zeros((n_impl, n_actions), dtype=float)
    
    for i, action in enumerate(unique_actions):
        if n_actions == 2:
            # Binary case: scalar representation
            D_action = np.full((n_impl, 1), action)
        else:
            # Multi-action case: one-hot encoding
            D_action = np.zeros((n_impl, n_actions))
            D_action[:, i] = 1
        
        # Augment features with action
        X_D_action = np.hstack([X_impl, D_action])
        
        # Predict outcomes under this action
        mu_mat_impl[:, i] = model.predict(X_D_action)
    
    # Recommend action with highest predicted outcome
    seg_labels_impl = np.argmax(mu_mat_impl, axis=1).astype(int)
    # Use unique_actions instead of np.arange to handle non-consecutive actions
    action_identity = unique_actions.astype(int)
    
    return seg_labels_impl, action_identity


def DR_learner(implement_customers, x_mat, D_vec, y_vec):
    """
    DR-Learner (Doubly Robust Learner) implementation.
    Supports multiple actions (multi-arm treatment).
    
    DR-learner combines outcome regression and propensity score weighting to create
    a doubly robust estimator. It's robust to misspecification of either the outcome
    model or the propensity model (but not both).
    
    Algorithm (for binary treatment):
    Step 1: Nuisance training
        (a) Estimate propensity scores π̂(x) = P(D=1|X=x)
        (b) Estimate outcome models μ̂_a(x) = E[Y|X=x, D=a] for each action a
    
    Step 2: Pseudo-outcome regression
        Construct pseudo-outcome: φ̂(Z) = [A - π̂(X)] / [π̂(X){1 - π̂(X)}] {Y - μ̂_A(X)} + μ̂_1(X) - μ̂_0(X)
        Estimate CATE: τ̂(x) = E[φ̂(Z) | X=x]
    
    For multi-arm treatment:
    - Uses action 0 as baseline
    - For each action a, estimates τ̂_a(x) = E[Y(a) - Y(0) | X=x]
    
    Parameters:
    implement_customers : list
        List of implementation customers.
    x_mat : array-like, shape (n_samples, n_features)
        Feature matrix from training data.
    D_vec : array-like, shape (n_samples,)
        Action/treatment assignment vector.
    y_vec : array-like, shape (n_samples,)
        Outcome vector from training data.
        
    Returns:
    seg_labels_impl : array-like, shape (n_impl_samples,)
        Recommended action for each implementation customer.
    action_identity : array-like, shape (n_actions,)
        Action identity mapping.
    """
    
    # Identify unique actions
    unique_actions = np.unique(D_vec)
    n_actions = len(unique_actions)    
    # Get implementation customer features
    X_impl = np.array([cust.x for cust in implement_customers])
    n_impl = X_impl.shape[0]
    
    if n_actions == 2:
        
        # Step 1a: Estimate propensity scores π̂(x) = P(D=1|X=x)
        propensity_model = LogisticRegression(max_iter=1000, random_state=42)
        propensity_model.fit(x_mat, D_vec)
        pi_hat = propensity_model.predict_proba(x_mat)[:, 1]  # P(D=1|X)
        
        # Clip propensity scores to avoid division by zero
        pi_hat = np.clip(pi_hat, 0.01, 0.99)
        
        # Step 1b: Estimate outcome models μ̂_a(x) for each action
        mu_models = {}
        for action in unique_actions:
            X_a = x_mat[D_vec == action]
            Y_a = y_vec[D_vec == action]
            
            if len(X_a) == 0:
                print(f"    Warning: No samples for action {action}")
                continue
            
            model_a = MLPRegressor(
                hidden_layer_sizes=(32,),
                activation='relu',
                max_iter=2000,
                early_stopping=True,
                random_state=42
            )
            model_a.fit(X_a, Y_a)
            mu_models[int(action)] = model_a
        
        # Predict μ̂_0(x) and μ̂_1(x) for training data
        mu_0 = mu_models[0].predict(x_mat)
        mu_1 = mu_models[1].predict(x_mat)
        
        # Get μ̂_A(x) - the predicted outcome under observed action
        mu_A = np.where(D_vec == 1, mu_1, mu_0)
        
        # Step 2: Construct pseudo-outcome φ̂(Z)
        # φ̂(Z) = [A - π̂(X)] / [π̂(X){1 - π̂(X)}] {Y - μ̂_A(X)} + μ̂_1(X) - μ̂_0(X)
        ipw_weight = (D_vec - pi_hat) / (pi_hat * (1 - pi_hat))
        pseudo_outcome = ipw_weight * (y_vec - mu_A) + (mu_1 - mu_0)
        
        # Step 3: Regress pseudo-outcome on X to get CATE estimate
        cate_model = MLPRegressor(
            hidden_layer_sizes=(32,),
            activation='relu',
            max_iter=2000,
            early_stopping=True,
            random_state=42
        )
        cate_model.fit(x_mat, pseudo_outcome)
        
        # Predict treatment effect for implementation customers
        tau_hat = cate_model.predict(X_impl)
        
        # Recommend action: treatment if τ̂(x) ≥ 0, control otherwise
        seg_labels_impl = (tau_hat >= 0).astype(int)
        action_identity = unique_actions.astype(int)

    else:
        # ========== Multi-arm treatment case ==========
        
        # Check if action 0 exists (needed as baseline)
        if 0 not in unique_actions:
            raise ValueError("DR-learner requires action 0 as baseline for multi-arm treatment")
        
        outcome_matrix = np.zeros((n_impl, n_actions))
        
        # Step 1b: Train outcome models for all actions
        mu_models = {}
        for action in unique_actions:
            X_a = x_mat[D_vec == action]
            Y_a = y_vec[D_vec == action]
            
            if len(X_a) == 0:
                print(f"    Warning: No samples for action {action}")
                continue
            
            model_a = MLPRegressor(
                hidden_layer_sizes=(32,),
                activation='relu',
                max_iter=2000,
                early_stopping=True,
                random_state=42
            )
            model_a.fit(X_a, Y_a)
            mu_models[int(action)] = model_a
        
        # Predict baseline outcome for implementation customers
        mu_0_impl = mu_models[0].predict(X_impl)
        outcome_matrix[:, 0] = mu_0_impl
        
        # For each non-baseline action, estimate treatment effect using DR
        for i, action in enumerate(unique_actions):
            if action == 0:
                continue  # Already handled baseline
            
            # Create binary problem: action a vs baseline 0
            mask = (D_vec == action) | (D_vec == 0)
            X_binary = x_mat[mask]
            Y_binary = y_vec[mask]
            D_binary = (D_vec[mask] == action).astype(float)
            
            if len(X_binary) == 0 or np.sum(D_binary == 1) == 0:
                print(f"    Warning: No samples for action {action}, using baseline")
                outcome_matrix[:, i] = outcome_matrix[:, 0]
                continue
            
            # Step 1a: Estimate propensity scores for this binary problem
            propensity_model_a = LogisticRegression(max_iter=1000, random_state=42)
            propensity_model_a.fit(X_binary, D_binary)
            pi_hat_a = propensity_model_a.predict_proba(X_binary)[:, 1]
            pi_hat_a = np.clip(pi_hat_a, 0.01, 0.99)
            
            # Predict outcomes under both actions for this subset
            mu_0_binary = mu_models[0].predict(X_binary)
            mu_a_binary = mu_models[int(action)].predict(X_binary)
            
            # Get μ̂_D(x) - predicted outcome under observed action
            mu_D = np.where(D_binary == 1, mu_a_binary, mu_0_binary)
            
            # Step 2: Construct pseudo-outcome
            ipw_weight = (D_binary - pi_hat_a) / (pi_hat_a * (1 - pi_hat_a))
            pseudo_outcome_a = ipw_weight * (Y_binary - mu_D) + (mu_a_binary - mu_0_binary)
            
            # Step 3: Regress pseudo-outcome on X
            cate_model_a = MLPRegressor(
                hidden_layer_sizes=(32,),
                activation='relu',
                max_iter=2000,
                early_stopping=True,
                random_state=42
            )
            cate_model_a.fit(X_binary, pseudo_outcome_a)
            
            # Predict treatment effect for implementation customers
            tau_a_impl = cate_model_a.predict(X_impl)
            
            # Predicted outcome = baseline + treatment effect
            outcome_matrix[:, i] = mu_0_impl + tau_a_impl
        
        # Recommend action with highest predicted outcome
        seg_labels_impl = np.argmax(outcome_matrix, axis=1).astype(int)
        action_identity = unique_actions.astype(int)
        
    return seg_labels_impl, action_identity