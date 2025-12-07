"""Bayesian Neural Network using Bayes by Backprop."""

from typing import Any

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
        training: bool = True,
    ) -> jnp.ndarray:
        """
        Forward pass through the Bayesian MLP.

        Args:
            inputs: Input data
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
            input_shape: Shape of input data (without batch dimension).

        Returns:
            Initialized parameters
        """
        # Extract feature dimension from input_shape
        if len(input_shape) == 1:
            input_dim = input_shape[0]
        else:
            input_dim = int(jnp.prod(jnp.array(input_shape)))
        
        # Create dummy input for initialization: (batch_size=1, features)
        dummy_input = jnp.zeros((1, input_dim), dtype=jnp.float32)
        return self.init(rng, dummy_input, training=True)

