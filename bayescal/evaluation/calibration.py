"""Calibration metrics: ECE, Brier score, and calibration curves."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def expected_calibration_error(
    predictions: jnp.ndarray,
    labels: jnp.ndarray,
    num_bins: int = 10,
) -> float:
    """
    Calculate Expected Calibration Error (ECE) for multiclass classification.
    
    Uses the top-label (confidence-based) approach for multiclass:
    - Confidence = max predicted probability (top-1 prediction confidence)
    - Accuracy = whether top prediction matches true label
    - Bins predictions by confidence level
    
    This is the standard multiclass ECE definition from:
    "On Calibration of Modern Neural Networks" (Guo et al., 2017)

    Args:
        predictions: Predicted probabilities of shape (n_samples, n_classes)
        labels: True labels (one-hot encoded) of shape (n_samples, n_classes)
        num_bins: Number of bins for calibration

    Returns:
        ECE score: Weighted average of |confidence - accuracy| across bins
    """
    # For multiclass: use top-1 confidence and check if top prediction is correct
    confidences = jnp.max(predictions, axis=1)  # Top-1 confidence
    predicted_classes = jnp.argmax(predictions, axis=1)
    true_classes = jnp.argmax(labels, axis=1)
    accuracies = predicted_classes == true_classes  # Top-1 accuracy

    bin_boundaries = jnp.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        # Only include bins with >= 1% samples
        if prop_in_bin >= 0.01:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += jnp.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def maximum_calibration_error(
    predictions: jnp.ndarray,
    labels: jnp.ndarray,
    num_bins: int = 10,
) -> float:
    """
    Calculate Maximum Calibration Error (MCE).
    
    Unlike ECE, MCE is not weighted by bin size, so it captures
    the worst-case calibration error across all bins.

    Args:
        predictions: Predicted probabilities of shape (n_samples, n_classes)
        labels: True labels (one-hot encoded) of shape (n_samples, n_classes)
        num_bins: Number of bins for calibration

    Returns:
        MCE score
    """
    confidences = jnp.max(predictions, axis=1)
    predicted_classes = jnp.argmax(predictions, axis=1)
    true_classes = jnp.argmax(labels, axis=1)
    accuracies = predicted_classes == true_classes

    bin_boundaries = jnp.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    mce = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        # Only consider bins with >= 1% samples
        if prop_in_bin >= 0.01:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            bin_error = jnp.abs(avg_confidence_in_bin - accuracy_in_bin)
            mce = jnp.maximum(mce, bin_error)

    return float(mce)


def calibration_bin_statistics(
    predictions: jnp.ndarray,
    labels: jnp.ndarray,
    num_bins: int = 10,
) -> dict[str, np.ndarray]:
    """
    Get detailed statistics for each calibration bin.
    
    Useful for diagnosing why ECE might be low despite visible miscalibration.

    Args:
        predictions: Predicted probabilities of shape (n_samples, n_classes)
        labels: True labels (one-hot encoded) of shape (n_samples, n_classes)
        num_bins: Number of bins for calibration

    Returns:
        Dictionary with arrays for each bin:
        - 'bin_centers': Center of each bin
        - 'proportions': Proportion of samples in each bin
        - 'accuracies': Accuracy in each bin
        - 'confidences': Average confidence in each bin
        - 'errors': Calibration error (|confidence - accuracy|) in each bin
    """
    confidences = jnp.max(predictions, axis=1)
    predicted_classes = jnp.argmax(predictions, axis=1)
    true_classes = jnp.argmax(labels, axis=1)
    accuracies = predicted_classes == true_classes

    bin_boundaries = jnp.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    proportions = []
    bin_accuracies = []
    bin_confidences = []
    bin_errors = []
    bin_centers_filtered = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        # Only include bins with >= 1% samples
        if prop_in_bin >= 0.01:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            bin_error = jnp.abs(avg_confidence_in_bin - accuracy_in_bin)
            
            bin_centers_filtered.append((bin_lower + bin_upper) / 2)
            proportions.append(float(prop_in_bin))
            bin_accuracies.append(float(accuracy_in_bin))
            bin_confidences.append(float(avg_confidence_in_bin))
            bin_errors.append(float(bin_error))

    return {
        "bin_centers": np.array(bin_centers_filtered),
        "proportions": np.array(proportions),
        "accuracies": np.array(bin_accuracies),
        "confidences": np.array(bin_confidences),
        "errors": np.array(bin_errors),
    }


def brier_score(
    predictions: jnp.ndarray,
    labels: jnp.ndarray,
) -> float:
    """
    Calculate Brier score.

    Args:
        predictions: Predicted probabilities of shape (n_samples, n_classes)
        labels: True labels (one-hot encoded) of shape (n_samples, n_classes)

    Returns:
        Brier score
    """
    return float(jnp.mean(jnp.sum((predictions - labels) ** 2, axis=1)))


def calibration_curve(
    predictions: jnp.ndarray,
    labels: jnp.ndarray,
    num_bins: int = 10,
    prune_small_bins: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate calibration curve.

    Args:
        predictions: Predicted probabilities of shape (n_samples, n_classes)
        labels: True labels (one-hot encoded) of shape (n_samples, n_classes)
        num_bins: Number of bins for calibration
        prune_small_bins: If True, only include bins with >= 1% samples. If False, include all bins.

    Returns:
        Tuple of (fraction_of_positives, mean_predicted_value) arrays
    """
    confidences = jnp.max(predictions, axis=1)
    predicted_classes = jnp.argmax(predictions, axis=1)
    true_classes = jnp.argmax(labels, axis=1)
    accuracies = (predicted_classes == true_classes).astype(jnp.float32)

    bin_boundaries = jnp.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    fraction_of_positives = []
    mean_predicted_value = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()  # Use mean for proportion

        # Include bin based on pruning setting
        if not prune_small_bins or prop_in_bin >= 0.01:
            # For empty bins, use NaN or skip
            if prop_in_bin > 0:
                fraction_of_positives.append(float(accuracies[in_bin].mean()))
                mean_predicted_value.append(float(confidences[in_bin].mean()))
            elif not prune_small_bins:
                # Include empty bins as NaN for unpruned version
                fraction_of_positives.append(float('nan'))
                mean_predicted_value.append(float((bin_lower + bin_upper) / 2))

    return (
        np.array(fraction_of_positives),
        np.array(mean_predicted_value),
    )

