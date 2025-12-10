"""Model export utilities for saving Flax models using Orbax checkpointing."""

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
)
from orbax.checkpoint import args as ocp_args


def export_model(
    model: Any,
    params: dict[str, Any],
    input_shape: tuple[int, ...],
    output_path: Path | str,
    model_name: str = "model",
    seed: int = 42,
) -> None:
    """
    Export a Flax/JAX model using Orbax checkpointing.

    Args:
        model: Flax model instance
        params: Model parameters dictionary
        input_shape: Input shape without batch dimension (e.g., (2,) for 2D input)
        output_path: Path where the model will be saved
        model_name: Name of the model (used for directory naming)
        seed: Random seed for inference (used to test model)
    """
    output_path = Path(output_path)
    
    # Create model-specific directory structure: {model_name}/
    if output_path.suffix == "":
        # output_path is already a directory
        model_dir = output_path / model_name.lower()
    else:
        # output_path is a file, create directory based on model name in same location
        model_dir = output_path.parent / model_name.lower()
    
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = (model_dir / "checkpoints").resolve()
    metadata_file = model_dir / "model.json"

    # Ensure params are in the correct format for apply()
    if "params" not in params:
        variables = {"params": params}
    else:
        variables = params

    # Test the model to get output shape
    rng_key = jax.random.PRNGKey(seed)
    batch_size = 1
    if len(input_shape) == 1:
        dummy_input = jnp.zeros((batch_size, input_shape[0]), dtype=jnp.float32)
    else:
        raise ValueError(f"Unsupported input shape: {input_shape}")

    # Test that the model works
    test_output = model.apply(
        variables,
        dummy_input,
        rng=rng_key,
        training=False,
        n_samples=1,
        method=model.__call__,
    )
    output_array = np.array(test_output)
    output_shape = output_array.shape[1:]  # Remove batch dimension

    # Save model parameters using Orbax checkpointing
    options = CheckpointManagerOptions(create=True)
    with CheckpointManager(checkpoint_dir, options=options) as checkpoint_manager:
        # Save checkpoint with step 0 using StandardSave
        checkpoint_manager.save(
            0,
            args=ocp_args.StandardSave({"params": params}),
        )

    # Save model metadata
    metadata = {
        "model_name": model_name,
        "input_shape": list(input_shape),
        "output_shape": list(output_shape),
        "framework": "jax/flax",
        "checkpoint_format": "orbax",
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)


def load_model(
    checkpoint_dir: Path | str,
    metadata_path: Path | str | None = None,
    step: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load a Flax model from Orbax checkpoint.

    Args:
        checkpoint_dir: Path to the checkpoint directory
        metadata_path: Optional path to metadata file. If None, looks for model.json
                      in the parent directory of checkpoint_dir
        step: Optional checkpoint step to load. If None, loads the latest checkpoint.

    Returns:
        Tuple of (parameters dictionary, metadata dictionary)
    """
    checkpoint_dir = Path(checkpoint_dir).resolve()
    
    # Load checkpoint using Orbax
    options = CheckpointManagerOptions(create=False)
    with CheckpointManager(checkpoint_dir, options=options) as checkpoint_manager:
        # Load the latest checkpoint if step not specified
        if step is None:
            step = checkpoint_manager.latest_step()
            if step is None:
                raise ValueError(f"No checkpoint found in {checkpoint_dir}")
        
        # Restore checkpoint - CheckpointManager knows the object is saved using
        # standard pytree logic, so we can restore directly
        restored = checkpoint_manager.restore(step)
        params = restored["params"]
    
    print(f"Loaded checkpoint from step {step} in {checkpoint_dir}")

    # Load metadata
    if metadata_path is None:
        # Look for model.json in parent directory
        metadata_path = checkpoint_dir.parent / "model.json"
    else:
        metadata_path = Path(metadata_path)
    
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    return params, metadata
