import numpy as np
from scipy.optimize import minimize_scalar

def optimize_b(z, w, x, X_neighbors, o, lambda_reg):
    """
    Optimize b_i to minimize lambda * sum_j |z_ij| * theta_ij
    subject to theta_ij >= -b_i (w_i^T (x_j - x_i) + b_i), theta_ij >= 0, and o_i /10 <|b_i| >= 2o_i / 3.
    
    Parameters:
    - w: Normal vector (d,), assumed unit norm
    - x: Center point (d,)
    - X_neighbors: Neighbor points (d x n)
    - z: Weights for each neighbor (n,)
    - o: 中心点x与其最近邻居的距离
    - lambda_reg: Regularization parameter
    
    Returns:
    - b: Optimized scalar
    - keth: Optimized thresholds (n,)
    """
    n = X_neighbors.shape[1]
    nx = X_neighbors - x[:, np.newaxis] # nx[j] = x_j - x_i
    lowerbound = o/10
    uperbound = o# Constraint threshold for |b_i|
    
    def objective(b):
        """Compute objective function for given b_i."""
        a = w @ nx + b  # w_i^T z_j + b_i
        keth = np.maximum(0, -b * a)
        # keth >= 0 implicitly satisfied
        return lambda_reg * np.sum(np.abs(z) * keth)
    
    # Optimize for positive b_i (>= o / 3)
    result_pos = minimize_scalar(objective, bounds=(lowerbound, uperbound), method='bounded')
    obj_pos = result_pos.fun
    b_pos = result_pos.x
    
    # Optimize for negative b_i (<= -o_i / 3)
    result_neg = minimize_scalar(objective, bounds=(-uperbound, -lowerbound), method='bounded')
    obj_neg = result_neg.fun
    b_neg = result_neg.x
    
    # Choose the solution with lower objective value
    if obj_pos <= obj_neg:
        b_opt = b_pos
    else:
        b_opt = b_neg
    
    # Compute theta_ij for optimal b_i
    a = w @ nx + b_opt
    keth = np.maximum(0, -b_opt * a)

    
    return b_opt, keth