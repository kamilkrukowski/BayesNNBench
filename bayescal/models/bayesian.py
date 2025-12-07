"""Bayesian Neural Network using Bayes by Backprop."""

from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from bayescal.models.layers.bayesian_layer import BayesianDense


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
            BayesianDense(
                features=dim,
                prior_std=self.prior_std,
                posterior_std_init=self.posterior_std_init,
            )
            for dim in self.hidden_dims
        ]
        self.output_layer = nn.Dense(self.num_classes)

    def _forward_single(
        self,
        inputs: jnp.ndarray,
        rng: Any,
        training: bool = True,
    ) -> jnp.ndarray:
        """
        Single forward pass through the Bayesian MLP.

        Args:
            inputs: Input data of shape (batch_size, input_dim)
            rng: Random number generator for Bayesian layer sampling
            training: Whether in training mode

        Returns:
            Class probabilities of shape (batch_size, num_classes)
        """
        x = inputs
        rngs = jax.random.split(rng, len(self.layers))
        
        for layer, layer_rng in zip(self.layers, rngs):
            x = layer(x, layer_rng, training=training)
            x = nn.relu(x)
        
        logits = self.output_layer(x)
        probs = nn.softmax(logits)
        return probs

    def __call__(
        self,
        inputs: jnp.ndarray,
        rng: Any,
        training: bool = True,
        n_samples: int = 1,
    ) -> jnp.ndarray:
        """
        Forward pass through the Bayesian MLP with Monte Carlo sampling.

        Args:
            inputs: Input data of shape (batch_size, input_dim)
            rng: Random number generator for Bayesian layer sampling
            training: Whether in training mode
            n_samples: Number of Monte Carlo samples to draw. For training, use 1.
                       For inference, use >1 to get better uncertainty estimates.

        Returns:
            Class probabilities of shape (batch_size, num_classes).
            If n_samples > 1, returns mean probabilities across samples.
        """
        if n_samples == 1:
            return self._forward_single(inputs, rng, training=training)
        
        # Multiple samples (for inference/uncertainty estimation)
        sample_rngs = jax.random.split(rng, n_samples)
        all_probs = [
            self._forward_single(inputs, sample_rng, training=training)
            for sample_rng in sample_rngs
        ]
        
        # Stack and average: (n_samples, batch_size, num_classes) -> (batch_size, num_classes)
        all_probs = jnp.stack(all_probs)  # (n_samples, batch_size, num_classes)
        mean_probs = jnp.mean(all_probs, axis=0)  # (batch_size, num_classes)
        return mean_probs

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
        rng1, rng2 = jax.random.split(rng)
        return self.init(rng1, dummy_input, rng2, training=True)

