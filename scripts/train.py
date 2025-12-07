"""Training script."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import serialization

from bayescal.config import settings
from bayescal.data import loaders, preprocessing
from bayescal.models import bayesian, feedforward
from bayescal.training import optimizers, trainer


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["bayesian", "feedforward"],
        default="bayesian",
        help="Type of model to train",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "fashion_mnist"],
        default="mnist",
        help="Dataset to use",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=settings.num_epochs,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.batch_size,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=settings.learning_rate,
        help="Learning rate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.results_dir,
        help="Output directory for saved models",
    )

    args = parser.parse_args()

    # Set random seed
    rng = jax.random.PRNGKey(settings.seed)

    # Load data
    if args.dataset == "mnist":
        train_loader = loaders.load_mnist(
            split="train",
            batch_size=args.batch_size,
        )
    else:
        train_loader = loaders.load_fashion_mnist(
            split="train",
            batch_size=args.batch_size,
        )

    # Initialize model
    if args.model_type == "bayesian":
        model = bayesian.BayesianMLP(
            hidden_dims=(settings.hidden_dim,) * settings.num_layers,
            num_classes=10,
        )
    else:
        model = feedforward.FFN(
            hidden_dims=(settings.hidden_dim,) * settings.num_layers,
            num_classes=10,
        )

    # Initialize parameters
    rng, init_rng = jax.random.split(rng)
    params = model.init_params(init_rng, input_shape=(784,))

    # Initialize optimizer
    optimizer = optimizers.get_optimizer(
        learning_rate=args.learning_rate,
        optimizer_type="adam",
    )
    opt_state = optimizer.init(params)

    # Training loop
    final_metrics = {}
    for epoch in range(args.epochs):
        rng, epoch_rng = jax.random.split(rng)
        params, opt_state, metrics = trainer.train_epoch(
            model,
            params,
            opt_state,
            train_loader,
            epoch_rng,
            optimizer,
        )
        final_metrics = metrics  # Keep track of final epoch metrics
        _metrics = {k: f"{v:.3f}" for k, v in metrics.items()}
        print(f"Epoch {epoch + 1}/{args.epochs}: {_metrics}")

    # Save model
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model parameters
    model_path = args.output_dir / f"{args.model_type}_{args.dataset}_params.flax"
    model_bytes = serialization.to_bytes(params)
    model_path.write_bytes(model_bytes)
    print(f"Model parameters saved to {model_path}")
    
    # Save model metadata
    metadata = {
        "model_type": args.model_type,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_dims": list((settings.hidden_dim,) * settings.num_layers),
        "num_classes": 10,
        "final_metrics": {k: float(v) for k, v in final_metrics.items()},
    }
    metadata_path = args.output_dir / f"{args.model_type}_{args.dataset}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()

