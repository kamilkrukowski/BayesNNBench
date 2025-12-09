"""Training and evaluation utilities for toy dataset experiments."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from bayescal.evaluation import calibration
from bayescal.training import optimizers, trainer


def train_model(
    model: Any,
    model_name: str,
    X_train: np.ndarray,
    y_train_onehot: np.ndarray,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 0.01,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    """
    Train a model on the toy dataset.

    Args:
        model: Model instance to train
        model_name: Name of the model (for logging)
        X_train: Training features
        y_train_onehot: Training labels (one-hot encoded)
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        max_grad_norm: Maximum gradient norm for clipping
        seed: Random seed
        verbose: Whether to print training progress

    Returns:
        Tuple of (trained_params, training_history)
    """
    rng = jax.random.PRNGKey(seed)

    # Initialize model
    input_shape = (X_train.shape[1],)
    rng, init_rng = jax.random.split(rng)
    params = model.init_params(init_rng, input_shape=input_shape)

    # Create optimizer with gradient clipping for stability
    optimizer = optimizers.get_optimizer(
        learning_rate=lr, optimizer_type="adam", max_grad_norm=max_grad_norm
    )
    opt_state = optimizer.init(params)

    # Create batches
    n_batches = (len(X_train) + batch_size - 1) // batch_size
    batches = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X_train))
        batch_X = jnp.array(X_train[start_idx:end_idx])
        batch_y = jnp.array(y_train_onehot[start_idx:end_idx])
        batches.append((batch_X, batch_y))

    # Training loop
    history = {"loss": [], "accuracy": []}
    rng, train_rng = jax.random.split(rng)

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        for batch_X, batch_y in batches:
            rng, batch_rng = jax.random.split(rng)

            # Determine n_vi_samples for Bayesian models
            is_bayesian = hasattr(model, "compute_kl_divergence") and hasattr(
                model, "beta"
            )
            n_vi_samples = 1 if is_bayesian else 1
            beta = getattr(model, "beta", 1.0) if is_bayesian else 1.0

            params, opt_state, metrics = trainer.train_step(
                model,
                params,
                opt_state,
                (batch_X, batch_y),
                batch_rng,
                optimizer,
                beta=beta,
                n_vi_samples=n_vi_samples,
            )

            epoch_loss += metrics["loss"]
            epoch_acc += metrics["accuracy"]
            num_batches += 1

        history["loss"].append(epoch_loss / num_batches)
        history["accuracy"].append(epoch_acc / num_batches)

        if verbose and (epoch + 1) % 50 == 0:
            print(
                f"{model_name} - Epoch {epoch+1}/{epochs}: "
                f"Loss={history['loss'][-1]:.4f}, Acc={history['accuracy'][-1]:.4f}"
            )

    return params, history


def evaluate_model(
    model: Any,
    params: dict[str, Any],
    X_test: np.ndarray,
    y_test_onehot: np.ndarray,
    n_samples: int = 1,
    model_name: str = "",
    seed: int = 42,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate model and return predictions, labels, and metrics.

    Args:
        model: Model instance
        params: Model parameters
        X_test: Test features
        y_test_onehot: Test labels (one-hot encoded)
        n_samples: Number of MC samples for evaluation
        model_name: Name of the model (for logging, optional)
        seed: Random seed

    Returns:
        Tuple of (predictions, labels, metrics_dict, fraction_of_positives, mean_predicted_value)
    """
    rng = jax.random.PRNGKey(seed)
    rng, eval_rng = jax.random.split(rng)

    X_test_jax = jnp.array(X_test)
    y_test_jax = jnp.array(y_test_onehot)

    # Get predictions
    probs = model.apply(
        params, inputs=X_test_jax, rng=eval_rng, training=False, n_samples=n_samples
    )

    # Compute metrics
    predicted_classes = jnp.argmax(probs, axis=-1)
    true_classes = jnp.argmax(y_test_jax, axis=-1)
    accuracy = (predicted_classes == true_classes).mean()

    ece = calibration.expected_calibration_error(probs, y_test_jax)
    mce = calibration.maximum_calibration_error(probs, y_test_jax)
    brier = calibration.brier_score(probs, y_test_jax)

    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration.calibration_curve(
        probs, y_test_jax, prune_small_bins=False
    )

    metrics = {
        "accuracy": float(accuracy),
        "ece": ece,
        "mce": mce,
        "brier": brier,
    }

    return probs, y_test_jax, metrics, fraction_of_positives, mean_predicted_value

