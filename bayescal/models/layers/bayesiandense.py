"""Bayesian Dense layer implementation for JAX."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn


class BayesianDense(nn.Module):
    """
    Bayesian dense layer using Bayes by Backprop with Local Reparameterization Trick.

    This layer maintains a distribution over weights rather than point estimates.
    Uses the Local Reparameterization Trick (LRT) for efficient training:
    instead of sampling weights, we directly sample output activations.
    Uses Flax's compact pattern to infer input dimension from first call.
    """

    features: int
    prior_std: float = 1.0
    posterior_std_init: float = 0.1

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        rng: Any,
        training: bool = True,
        sample: bool = True,
    ) -> jnp.ndarray:
        """
        Forward pass with Local Reparameterization Trick (LRT).

        Args:
            inputs: Input tensor of shape (batch, input_dim)
            rng: Random number generator
            training: Whether in training mode
            sample: Whether to sample from the weight distribution (True) or use mean weights (False)

        Returns:
            Output tensor of shape (batch, features)
        """
        input_dim = inputs.shape[-1]
        weight_shape = (input_dim, self.features)
        
        # Define parameters (Flax will initialize on first call)
        mean = self.param(
            "mean",
            nn.initializers.normal(stddev=0.1),
            weight_shape,
        )
        log_std = self.param(
            "log_std",
            lambda rng, shape: jnp.full(shape, jnp.log(self.posterior_std_init)),
            weight_shape,
        )
        
        # Use jax.lax.cond for JIT-compatible conditional execution
        def training_forward(inputs, rng, mean, log_std):
            """
            Training mode: Local Reparameterization Trick (LRT) for dense layers.
            
            Instead of sampling weights and then multiplying, we directly sample
            the output activations. This is more memory efficient and faster.
            
            For dense(x, W) where W ~ N(μ, σ²) with independent weights:
            - Mean output: x @ μ
            - Variance output: x² @ σ² (element-wise square of inputs, element-wise square of std)
            - Sample: mean + sqrt(variance) * ε
            """
            std = jnp.exp(log_std)
            std_sq = std ** 2  # Variance of weights
            
            # Compute mean output: inputs @ mean_weights
            mean_output = inputs @ mean
            
            # Compute variance output: (inputs²) @ (std²)
            # For dense layer: Var[y] = sum(x² * σ²) where x² is element-wise square
            inputs_sq = inputs ** 2
            var_output = inputs_sq @ std_sq
            
            # Sample output activations: mean + sqrt(variance) * epsilon
            std_output = jnp.sqrt(var_output + 1e-8)  # Add small epsilon for numerical stability
            output_shape = mean_output.shape
            eps = jax.random.normal(rng, output_shape)
            sampled_output = mean_output + std_output * eps
            
            return sampled_output

        def mean_forward(inputs, mean):
            """Use mean weights (deterministic)."""
            return inputs @ mean

        # Use jax.lax.cond for JIT-compatible conditional execution
        return jax.lax.cond(
            sample,
            lambda: training_forward(inputs, rng, mean, log_std),
            lambda: mean_forward(inputs, mean),
        )

