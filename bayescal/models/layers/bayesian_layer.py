"""Custom Bayesian layer implementation for JAX."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn


class BayesianDense(nn.Module):
    """
    Bayesian dense layer using Bayes by Backprop.

    This layer maintains a distribution over weights rather than point estimates.
    """

    features: int
    prior_std: float = 1.0
    posterior_std_init: float = 0.1

    def setup(self) -> None:
        """Initialize layer parameters."""
        # Posterior parameters (mean and log std)
        self.mean = self.param(
            "mean",
            nn.initializers.normal(stddev=0.1),
            (self.features,),
        )
        self.log_std = self.param(
            "log_std",
            lambda rng, shape: jnp.full(shape, jnp.log(self.posterior_std_init)),
            (self.features,),
        )

    def __call__(
        self,
        inputs: jnp.ndarray,
        rng: Any,
        training: bool = True,
    ) -> jnp.ndarray:
        """
        Forward pass with reparameterization trick.

        Args:
            inputs: Input tensor
            rng: Random number generator
            training: Whether in training mode

        Returns:
            Output tensor
        """
        if training:
            # Sample weights from posterior
            std = jnp.exp(self.log_std)
            eps = jax.random.normal(rng, self.mean.shape)
            weights = self.mean + std * eps
        else:
            # Use mean weights during inference
            weights = self.mean

        return inputs @ weights

