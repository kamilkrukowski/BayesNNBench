"""Custom Bayesian layer implementation for JAX."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn


class BayesianDense(nn.Module):
    """
    Bayesian dense layer using Bayes by Backprop.

    This layer maintains a distribution over weights rather than point estimates.
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
    ) -> jnp.ndarray:
        """
        Forward pass with reparameterization trick.

        Args:
            inputs: Input tensor of shape (batch, input_dim)
            rng: Random number generator
            training: Whether in training mode

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
        
        if training:
            # Sample weights from posterior using reparameterization trick
            std = jnp.exp(log_std)
            eps = jax.random.normal(rng, weight_shape)
            weights = mean + std * eps
        else:
            # Use mean weights during inference
            weights = mean

        return inputs @ weights

