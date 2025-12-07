"""Optimizer configurations."""

import optax


def get_optimizer(
    learning_rate: float = 1e-3,
    optimizer_type: str = "adam",
) -> optax.GradientTransformation:
    """
    Get optimizer based on type.

    Args:
        learning_rate: Learning rate
        optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw')

    Returns:
        Optimizer instance
    """
    if optimizer_type.lower() == "adam":
        return optax.adam(learning_rate)
    elif optimizer_type.lower() == "sgd":
        return optax.sgd(learning_rate, momentum=0.9)
    elif optimizer_type.lower() == "adamw":
        return optax.adamw(learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

