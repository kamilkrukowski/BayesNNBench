"""Training loop implementation."""

from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from flax import linen as nn


def train_step(
    model: nn.Module,
    params: dict[str, Any],
    opt_state: optax.OptState,
    batch: tuple[jnp.ndarray, jnp.ndarray],
    rng: Any,
    optimizer: optax.GradientTransformation,
) -> tuple[dict[str, Any], optax.OptState, dict[str, float]]:
    """
    Single training step.

    Args:
        model: Model instance
        params: Model parameters
        opt_state: Optimizer state
        batch: Training batch (inputs, labels)
        rng: Random number generator
        optimizer: Optimizer

    Returns:
        Updated parameters, optimizer state, and metrics
    """
    inputs, labels = batch
    rng, step_rng = jax.random.split(rng)

    def loss_fn(p: dict[str, Any]) -> tuple[jnp.ndarray, dict[str, float]]:
        probs = model.apply(p, inputs, step_rng, training=True)
        # Cross-entropy loss: -sum(y * log(p))
        log_probs = jnp.log(probs + 1e-8)  # Add small epsilon for numerical stability
        loss = -jnp.sum(labels * log_probs, axis=-1).mean()
        accuracy = (probs.argmax(axis=-1) == labels.argmax(axis=-1)).mean()
        return loss, {"accuracy": float(accuracy)}

    (loss, metrics), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    metrics["loss"] = float(loss)
    return params, opt_state, metrics


def train_epoch(
    model: nn.Module,
    params: dict[str, Any],
    opt_state: optax.OptState,
    train_loader: Any,
    rng: Any,
    optimizer: optax.GradientTransformation,
) -> tuple[dict[str, Any], optax.OptState, dict[str, float]]:
    """
    Train for one epoch.

    Args:
        model: Model instance
        params: Model parameters
        opt_state: Optimizer state
        train_loader: Training data loader
        rng: Random number generator
        optimizer: Optimizer

    Returns:
        Updated parameters, optimizer state, and epoch metrics
    """
    epoch_metrics = {"loss": 0.0, "accuracy": 0.0}
    num_batches = 0

    for batch in tqdm(train_loader, desc="Training"):
        params, opt_state, batch_metrics = train_step(
            model, params, opt_state, batch, rng, optimizer
        )
        for key in epoch_metrics:
            epoch_metrics[key] += batch_metrics[key]
        num_batches += 1
        rng, _ = jax.random.split(rng)

    # Average metrics
    for key in epoch_metrics:
        epoch_metrics[key] /= num_batches

    return params, opt_state, epoch_metrics

