"""Visualization utilities for toy dataset experiments."""

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split


def plot_single_sample_predictions(
    model: Any,
    params: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    n_samples_to_show: int = 5,
    n_plot_points: int = 2000,
    seed: int = 42,
    figures_dir: Path | None = None,
) -> None:
    """
    Plot decision boundaries from individual MC samples (not averaged).

    Args:
        model: Model instance
        params: Model parameters
        X_test: Test features
        y_test: Test labels (not one-hot)
        model_name: Name of the model
        n_samples_to_show: Number of individual samples to plot
        n_plot_points: Number of test points to plot (subsampled if larger)
        seed: Random seed
        figures_dir: Directory to save figures (optional)
    """
    # Subsample test set for visualization
    if len(X_test) > n_plot_points:
        _, X_test_plot, _, y_test_plot = train_test_split(
            X_test,
            y_test,
            test_size=n_plot_points / len(X_test),
            random_state=seed,
            stratify=y_test,
        )
    else:
        X_test_plot, y_test_plot = X_test, y_test

    # Create a grid
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_jax = jnp.array(grid_points)

    rng = jax.random.PRNGKey(seed)

    # Get individual sample predictions (not averaged)
    fig, axes = plt.subplots(1, n_samples_to_show, figsize=(5 * n_samples_to_show, 5))
    if n_samples_to_show == 1:
        axes = [axes]

    for i in range(n_samples_to_show):
        rng, eval_rng = jax.random.split(rng)
        # Single sample, no averaging
        probs = model.apply(
            params, inputs=grid_jax, rng=eval_rng, training=False, n_samples=1
        )
        probs_class1 = np.array(probs[:, 1]).reshape(xx.shape)

        contour = axes[i].contourf(
            xx, yy, probs_class1, levels=20, cmap="coolwarm", alpha=0.8
        )
        axes[i].contour(xx, yy, probs_class1, levels=[0.5], colors="black", linewidths=2)
        axes[i].scatter(
            X_test_plot[:, 0],
            X_test_plot[:, 1],
            c=y_test_plot,
            cmap="coolwarm",
            edgecolors="black",
            linewidths=0.3,
            s=15,
            alpha=0.5,
        )
        axes[i].set_title(f"{model_name} - Sample {i+1}", fontsize=10)
        axes[i].set_xlabel("Feature 1", fontsize=9)
        axes[i].set_ylabel("Feature 2", fontsize=9)

    plt.suptitle(
        f"{model_name}: Individual MC Samples (n_samples=1 each)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    if figures_dir:
        model_name_safe = model_name.lower().replace(" ", "_")
        plt.savefig(
            figures_dir / f"{model_name_safe}_individual_samples.png",
            dpi=150,
            bbox_inches="tight",
        )
    plt.show()

    # Also plot the averaged version for comparison
    print(f"\n{model_name} - Averaged over 100 samples:")
    rng_avg = jax.random.PRNGKey(seed)
    rng_avg, eval_rng_avg = jax.random.split(rng_avg)
    probs_avg = model.apply(
        params, inputs=grid_jax, rng=eval_rng_avg, training=False, n_samples=100
    )
    probs_class1_avg = np.array(probs_avg[:, 1]).reshape(xx.shape)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    contour = ax.contourf(
        xx, yy, probs_class1_avg, levels=20, cmap="coolwarm", alpha=0.8
    )
    ax.contour(xx, yy, probs_class1_avg, levels=[0.5], colors="black", linewidths=2)
    ax.scatter(
        X_test_plot[:, 0],
        X_test_plot[:, 1],
        c=y_test_plot,
        cmap="coolwarm",
        edgecolors="black",
        linewidths=0.3,
        s=15,
        alpha=0.5,
    )
    ax.set_xlabel("Feature 1", fontsize=12)
    ax.set_ylabel("Feature 2", fontsize=12)
    ax.set_title(
        f"{model_name}: Averaged Decision Boundary (MC=100)",
        fontsize=12,
        fontweight="bold",
    )
    plt.colorbar(contour, ax=ax)
    plt.tight_layout()
    if figures_dir:
        model_name_safe = model_name.lower().replace(" ", "_")
        plt.savefig(
            figures_dir / f"{model_name_safe}_averaged_boundary_mc100.png",
            dpi=150,
            bbox_inches="tight",
        )
    plt.show()


def plot_predictive_posterior(
    model: Any,
    params: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    n_samples: int = 1,
    n_plot_points: int = 2000,
    seed: int = 42,
    figures_dir: Path | None = None,
) -> None:
    """
    Plot predictive posterior over the input space.

    Args:
        model: Model instance
        params: Model parameters
        X_test: Test features
        y_test: Test labels (not one-hot)
        model_name: Name of the model
        n_samples: Number of MC samples for prediction
        n_plot_points: Number of test points to plot (subsampled if larger)
        seed: Random seed
        figures_dir: Directory to save figures (optional)
    """
    # Subsample test set for visualization
    if len(X_test) > n_plot_points:
        _, X_test_plot, _, y_test_plot = train_test_split(
            X_test,
            y_test,
            test_size=n_plot_points / len(X_test),
            random_state=seed,
            stratify=y_test,
        )
    else:
        X_test_plot, y_test_plot = X_test, y_test

    # Create a grid
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Get predictions on grid
    rng = jax.random.PRNGKey(seed)
    rng, eval_rng = jax.random.split(rng)
    grid_jax = jnp.array(grid_points)

    probs = model.apply(
        params, inputs=grid_jax, rng=eval_rng, training=False, n_samples=n_samples
    )
    probs_class1 = np.array(probs[:, 1]).reshape(xx.shape)

    # Plot: Single plot with probability contour and overlaid decision boundary
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Probability contour
    contour = ax.contourf(xx, yy, probs_class1, levels=20, cmap="coolwarm", alpha=0.8)

    # Overlay decision boundary (where P(Class=1) = 0.5)
    boundary_contour = ax.contour(
        xx, yy, probs_class1, levels=[0.5], colors="black", linewidths=2.5, linestyles="-"
    )

    # Scatter test points
    scatter = ax.scatter(
        X_test_plot[:, 0],
        X_test_plot[:, 1],
        c=y_test_plot,
        cmap="coolwarm",
        edgecolors="black",
        linewidths=0.3,
        s=15,
        alpha=0.5,
        zorder=10,
    )

    # Labels and title
    ax.set_xlabel("Feature 1", fontsize=12)
    ax.set_ylabel("Feature 2", fontsize=12)
    ax.set_title(
        f"{model_name}: Predictive Probability P(Class=1)", fontsize=12, fontweight="bold"
    )

    # Colorbar for probability
    plt.colorbar(contour, ax=ax, label="P(Class=1)")

    # Legend for decision boundary
    boundary_line = Line2D(
        [0],
        [0],
        color="black",
        linewidth=2.5,
        linestyle="-",
        label="Decision Boundary (P=0.5)",
    )
    ax.legend(handles=[boundary_line], loc="upper right", fontsize=11)

    plt.tight_layout()
    if figures_dir:
        model_name_safe = model_name.lower().replace(" ", "_")
        plt.savefig(
            figures_dir / f"{model_name_safe}_predictive_posterior_mc{n_samples}.png",
            dpi=150,
            bbox_inches="tight",
        )
    plt.show()


def plot_uncertainty(
    model: Any,
    params: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    n_samples: int = 100,
    n_plot_points: int = 2000,
    seed: int = 42,
    figures_dir: Path | None = None,
) -> None:
    """
    Plot uncertainty (entropy of predictions) over the input space.

    Args:
        model: Model instance
        params: Model parameters
        X_test: Test features
        y_test: Test labels (not one-hot)
        model_name: Name of the model
        n_samples: Number of MC samples for uncertainty estimation
        n_plot_points: Number of test points to plot (subsampled if larger)
        seed: Random seed
        figures_dir: Directory to save figures (optional)
    """
    # Subsample test set for visualization
    if len(X_test) > n_plot_points:
        _, X_test_plot, _, y_test_plot = train_test_split(
            X_test,
            y_test,
            test_size=n_plot_points / len(X_test),
            random_state=seed,
            stratify=y_test,
        )
    else:
        X_test_plot, y_test_plot = X_test, y_test

    # Create a grid
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Get multiple predictions on grid
    rng = jax.random.PRNGKey(seed)
    grid_jax = jnp.array(grid_points)

    all_probs = []
    for i in range(n_samples):
        rng, eval_rng = jax.random.split(rng)
        probs = model.apply(
            params, inputs=grid_jax, rng=eval_rng, training=False, n_samples=1
        )
        all_probs.append(np.array(probs))

    all_probs = np.array(all_probs)  # (n_samples, n_points, n_classes)
    mean_probs = np.mean(all_probs, axis=0)
    std_probs = np.std(all_probs, axis=0)

    # Use entropy as uncertainty measure: -sum(p * log(p))
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)
    entropy = entropy.reshape(xx.shape)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    contour = ax.contourf(xx, yy, entropy, levels=20, cmap="viridis", alpha=0.8)
    ax.scatter(
        X_test_plot[:, 0],
        X_test_plot[:, 1],
        c=y_test_plot,
        cmap="coolwarm",
        edgecolors="black",
        linewidths=0.3,
        s=15,
        alpha=0.5,
        zorder=10,
    )
    ax.set_xlabel("Feature 1", fontsize=12)
    ax.set_ylabel("Feature 2", fontsize=12)
    ax.set_title(
        f"{model_name}: Predictive Uncertainty (Entropy)", fontsize=14, fontweight="bold"
    )
    plt.colorbar(contour, ax=ax, label="Entropy (bits)")
    plt.tight_layout()
    if figures_dir:
        model_name_safe = model_name.lower().replace(" ", "_")
        plt.savefig(
            figures_dir / f"{model_name_safe}_uncertainty_mc{n_samples}.png",
            dpi=150,
            bbox_inches="tight",
        )
    plt.show()

