"""Utilities for toy dataset analysis and visualization."""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def compute_bayes_optimal_boundary(
    overlap: float = 0.8,
    x_range: Tuple[float, float] = (-4, 4),
    y_range: Tuple[float, float] = (-4, 4),
    n_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Bayes optimal decision boundary for two overlapping Gaussians.
    
    For equal priors P(y=0) = P(y=1) = 0.5, the optimal boundary is where:
    log P(x|y=0) = log P(x|y=1)
    
    Which simplifies to:
    (x - μ₀)ᵀ Σ₀⁻¹ (x - μ₀) - log|Σ₀| = (x - μ₁)ᵀ Σ₁⁻¹ (x - μ₁) - log|Σ₁|
    
    Args:
        overlap: Overlap parameter used in dataset generation (must match generate_toy_dataset)
        x_range: (x_min, x_max) for grid
        y_range: (y_min, y_max) for grid
        n_points: Number of points in each dimension for the grid
    
    Returns:
        boundary_points: Array of shape (n_boundary_points, 2) with points on the boundary
        grid_x, grid_y: Meshgrid for plotting
        bayes_probs: Array of shape (n_points, n_points) with P(y=1|x) for each grid point
    """
    # Define the distributions (matching generate_toy_dataset)
    mean_0 = np.array([-1.0, -1.0])
    cov_0 = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    mean_1 = np.array([1.0, 1.0])
    cov_1 = np.array([[1.0 + overlap, 0.0], [0.0, 1.0 + overlap]])
    
    # Create grid
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    grid_x, grid_y = np.meshgrid(x, y)
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    
    # Compute log-likelihoods for each class
    log_likelihood_0 = multivariate_normal.logpdf(grid_points, mean=mean_0, cov=cov_0)
    log_likelihood_1 = multivariate_normal.logpdf(grid_points, mean=mean_1, cov=cov_1)
    
    # For equal priors, posterior is proportional to likelihood
    # P(y=1|x) = P(x|y=1) / (P(x|y=0) + P(x|y=1))
    # Using log-space for numerical stability
    log_likelihood_0_2d = log_likelihood_0.reshape(grid_x.shape)
    log_likelihood_1_2d = log_likelihood_1.reshape(grid_x.shape)
    
    # Compute posterior probabilities using log-sum-exp trick
    max_log = np.maximum(log_likelihood_0_2d, log_likelihood_1_2d)
    log_sum = max_log + np.log(
        np.exp(log_likelihood_0_2d - max_log) + np.exp(log_likelihood_1_2d - max_log)
    )
    log_posterior_1 = log_likelihood_1_2d - log_sum
    bayes_probs = np.exp(log_posterior_1)
    
    # Find boundary points (where P(y=1|x) = 0.5, or equivalently log_likelihood_0 = log_likelihood_1)
    log_diff = log_likelihood_1_2d - log_likelihood_0_2d
    
    # Use matplotlib's contour to find the boundary
    fig = plt.figure(figsize=(1, 1))
    cs = plt.contour(grid_x, grid_y, log_diff, levels=[0], colors='red')
    plt.close(fig)
    
    # Extract boundary points from contour
    boundary_points = []
    if len(cs.allsegs) > 0 and len(cs.allsegs[0]) > 0:
        for seg in cs.allsegs[0]:
            boundary_points.append(seg)
        if boundary_points:
            boundary_points = np.vstack(boundary_points)
        else:
            boundary_points = np.array([]).reshape(0, 2)
    else:
        boundary_points = np.array([]).reshape(0, 2)
    
    return boundary_points, grid_x, grid_y, bayes_probs

