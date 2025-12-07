"""Bayesian Neural Network using Bayes by Backprop."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

class BayesianMLP(nn.Module):
    """
    Bayesian Multi-Layer Perceptron using Bayes by Backprop.

    This model maintains distributions over weights and uses variational inference.
    """

    hidden_dims: tuple[int, ...] = (128, 128)
    num_classes: int = 10
    prior_std: float = 1.0
    posterior_std_init: float = 0.1

    def setup(self) -> None:
        """Initialize model layers."""
        self.layers = [
            nn.Dense(dim) for dim in self.hidden_dims
        ]
        self.output_layer = nn.Dense(self.num_classes)

    def __call__(
        self,
        inputs: jnp.ndarray,
        rng: Any,
        training: bool = True,
    ) -> jnp.ndarray:
        """
        Forward pass through the Bayesian MLP.

        Args:
            inputs: Input data
            rng: Random number generator
            training: Whether in training mode

        Returns:
            Class probabilities of shape (batch_size, num_classes)
        """
        x = inputs
        for layer in self.layers:
            x = layer(x)
            x = nn.relu(x)
        logits = self.output_layer(x)
        probs = nn.softmax(logits)
        return probs

    def init_params(
        self,
        rng: Any,
        input_shape: tuple[int, ...],
    ) -> dict[str, Any]:
        """
        Initialize model parameters.

        Args:
            rng: Random number generator
            input_shape: Shape of input data

        Returns:
            Initialized parameters
        """
        dummy_input = jnp.ones((1, *input_shape))
        return self.init(rng, dummy_input, training=True)

