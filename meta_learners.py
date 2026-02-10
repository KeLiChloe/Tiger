import numpy as np
from utils import assign_trained_customers_to_segments
from ground_truth import PopulationSimulator, SegmentEstimate

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
    action_identity = np.arange(len(mu_pilot_models), dtype=int)
    
    return seg_labels_impl, action_identity
    

def _build_mu_matrix(mu_models, X_impl):
    n = X_impl.shape[0]
    mu_mat = np.zeros((n, len(mu_models)), dtype=float)
    for a, model in mu_models.items():
        a_int = int(a)
        pred = model.predict(X_impl)
        mu_mat[:, a_int] = pred

    return mu_mat


def X_learner(implement_customers, x_mat, D_vec, y_vec, propensity_scores=None):
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
    action_identity = np.arange(n_actions, dtype=int)
    
    
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
    action_identity = np.arange(n_actions, dtype=int)
    
    return seg_labels_impl, action_identity