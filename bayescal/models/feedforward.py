"""Feedforward Neural Network with Dropout."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn


class FFN(nn.Module):
    """
    Feedforward Neural Network with Dropout.

    This model uses point estimates for weights and dropout for regularization.
    """

    hidden_dims: tuple[int, ...] = (128, 128)
    num_classes: int = 10
    dropout_rate: float = 0.5

    def setup(self) -> None:
        """Initialize model layers."""
        self.dense_layers = [
            nn.Dense(dim) for dim in self.hidden_dims
        ]
        self.dropout_layers = [
            nn.Dropout(rate=self.dropout_rate) for _ in self.hidden_dims
        ]
        self.output_layer = nn.Dense(self.num_classes)

    def __call__(
        self,
        inputs: jnp.ndarray,
        rng: Any,
        training: bool = True,
    ) -> jnp.ndarray:
        """
        Forward pass through the feedforward network.

        Args:
            inputs: Input data
            rng: Random number generator
            training: Whether in training mode

        Returns:
            Class probabilities of shape (batch_size, num_classes)
        """
        x = inputs
        rngs = jax.random.split(rng, len(self.dense_layers))
        
        for dense, dropout, rng_key in zip(
            self.dense_layers, self.dropout_layers, rngs
        ):
            x = dense(x)
            x = nn.relu(x)
            if training:
                x = dropout(x, rng=rng_key)
        
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

