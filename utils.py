
import numpy as np

def build_design_matrix(x_array, D_array):
    """Add intercept and treatment column to covariates."""
    N, _ = x_array.shape
    intercept = np.ones((N, 1))
    D_col = D_array.reshape(-1, 1)
    return np.hstack([intercept, x_array, D_col])