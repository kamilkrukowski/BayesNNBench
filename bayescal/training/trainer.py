"""Training loop implementation."""

from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from flax import linen as nn

from bayescal.evaluation import metrics as eval_metrics


# Create a cached JIT-compiled function for gradient computation
# This will be compiled once per unique function signature
_grad_fn_cache = {}


def train_step(
    model: nn.Module,
    params: dict[str, Any],
    opt_state: optax.OptState,
    batch: tuple[jnp.ndarray, jnp.ndarray],
    rng: Any,
    optimizer: optax.GradientTransformation,
    beta: float = 1.0,
    n_vi_samples: int = 1,
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
        beta: Beta parameter for beta-VI (only used for Bayesian models)
        n_vi_samples: Number of samples for variational inference during training.
                     Used for Bayesian models to get more stable gradient estimates.

    Returns:
        Updated parameters, optimizer state, and metrics
    """
    inputs, labels = batch
    rng, step_rng = jax.random.split(rng)

    # Determine n_vi_samples: use provided value for Bayesian models, 1 otherwise
    is_bayesian = hasattr(model, "compute_kl_divergence") and hasattr(model, "beta")
    n_samples = n_vi_samples if is_bayesian else 1

    # Create a loss function that captures the model's get_loss method
    # We'll JIT-compile the gradient computation
    def loss_fn(p: dict[str, Any]) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        return model.get_loss(p, inputs=inputs, labels=labels, rng=step_rng, n_vi_samples=n_samples)

    # JIT-compile the value_and_grad computation
    # Use a cache key based on the model type to avoid recompiling unnecessarily
    cache_key = (type(model).__name__, is_bayesian, n_samples)
    if cache_key not in _grad_fn_cache:
        # Create a template function that will be JIT-compiled
        # The actual model.get_loss will be called inside, but JAX will trace through it
        # Mark n_samples_val as static so the Python if statement in get_loss can be evaluated
        def grad_fn_template(p, inputs, labels, rng_key, n_samples_val):
            def inner_loss(pp):
                # This will call model.get_loss, which JAX can trace through
                return model.get_loss(pp, inputs=inputs, labels=labels, rng=rng_key, n_vi_samples=n_samples_val)
            return jax.value_and_grad(inner_loss, has_aux=True)(p)
        
        # Mark n_samples_val (argument index 4) as static so Python conditionals work
        _grad_fn_cache[cache_key] = jax.jit(grad_fn_template, static_argnums=(4,))
    
    grad_fn = _grad_fn_cache[cache_key]
    (loss, metrics), grads = grad_fn(params, inputs, labels, step_rng, n_samples)
    
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    # Convert JAX arrays to Python floats after gradient computation
    metrics_float = {k: float(v) for k, v in metrics.items()}
    metrics_float["loss"] = float(loss)
    return params, opt_state, metrics_float


def train_epoch(
    model: nn.Module,
    params: dict[str, Any],
    opt_state: optax.OptState,
    train_loader: Any,
    rng: Any,
    optimizer: optax.GradientTransformation,
    beta: float = 1.0,
    n_vi_samples: int = 1,
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
        beta: Beta parameter for beta-VI (only used for Bayesian models)
        n_vi_samples: Number of samples for variational inference during training.
                     Used for Bayesian models to get more stable gradient estimates.

    Returns:
        Updated parameters, optimizer state, and epoch metrics
    """
    epoch_metrics = {}
    num_batches = 0
    
    # Collect predictions and labels for macro metrics
    all_predictions = []
    all_labels = []

    for batch in tqdm(train_loader, desc="Training"):
        inputs, labels = batch
        params, opt_state, batch_metrics = train_step(
            model, params, opt_state, batch, rng, optimizer, beta=beta, n_vi_samples=n_vi_samples
        )
        # Accumulate all metrics from batch_metrics
        for key, value in batch_metrics.items():
            if key not in epoch_metrics:
                epoch_metrics[key] = 0.0
            epoch_metrics[key] += value
        num_batches += 1
        rng, batch_rng = jax.random.split(rng)
        
        # Get predictions for macro metrics (using current params)
        probs = model.apply(params, inputs=inputs, rng=batch_rng, training=False, n_samples=1)
        all_predictions.append(probs)
        all_labels.append(labels)

    # Average batch metrics (skip macro metrics which are computed separately)
    macro_keys = {"macro_auroc", "macro_f1"}
    for key in epoch_metrics:
        if key not in macro_keys:
            epoch_metrics[key] /= num_batches
    
    # Compute macro metrics across entire epoch
    if all_predictions:
        predictions = jnp.concatenate(all_predictions, axis=0)
        labels = jnp.concatenate(all_labels, axis=0)
        
        epoch_metrics["macro_auroc"] = eval_metrics.macro_auroc(predictions, labels)
        epoch_metrics["macro_f1"] = eval_metrics.macro_f1(predictions, labels)
    else:
        epoch_metrics["macro_auroc"] = 0.0
        epoch_metrics["macro_f1"] = 0.0

    return params, opt_state, epoch_metrics

