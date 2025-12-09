"""Data loaders for CIFAR-10, OOD datasets, and synthetic datasets."""

from typing import Any, List, Tuple

import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from bayescal.data import preprocessing


def generate_toy_dataset(
    n_samples: int = 2000,
    overlap: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 2D toy dataset with two overlapping Gaussian classes.
    
    This synthetic dataset is useful for:
    - Testing calibration methods
    - Visualizing decision boundaries
    - Understanding uncertainty estimation
    - Demonstrating Bayesian neural networks
    
    Args:
        n_samples: Total number of samples
        overlap: Controls overlap between classes (0 = no overlap, 1+ = maximum overlap).
                Higher values increase the covariance of class 1, creating more overlap.
        seed: Random seed for reproducibility
    
    Returns:
        X: Features of shape (n_samples, 2)
        y: Labels (0 or 1) of shape (n_samples,)
        y_onehot: One-hot encoded labels of shape (n_samples, 2)
    
    Example:
        >>> X, y, y_onehot = generate_toy_dataset(n_samples=2000, overlap=0.8, seed=42)
        >>> print(X.shape, y.shape, y_onehot.shape)
        (2000, 2) (2000,) (2000, 2)
    """
    np.random.seed(seed)
    n_per_class = n_samples // 2
    
    # Class 0: centered at (-1, -1) with unit covariance
    mean_0 = np.array([-1.0, -1.0])
    cov_0 = np.array([[1.0, 0.0], [0.0, 1.0]])
    X_0 = np.random.multivariate_normal(mean_0, cov_0, n_per_class)
    y_0 = np.zeros(n_per_class, dtype=int)
    
    # Class 1: centered at (1, 1) with controlled overlap
    # Higher overlap increases covariance, creating more spread and overlap
    mean_1 = np.array([1.0, 1.0])
    cov_1 = np.array([[1.0 + overlap, 0.0], [0.0, 1.0 + overlap]])
    X_1 = np.random.multivariate_normal(mean_1, cov_1, n_per_class)
    y_1 = np.ones(n_per_class, dtype=int)
    
    # Combine
    X = np.vstack([X_0, X_1])
    y = np.hstack([y_0, y_1])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # One-hot encode
    y_onehot = np.zeros((len(y), 2))
    y_onehot[np.arange(len(y)), y] = 1
    
    return X, y, y_onehot


def load_cifar10(
    split: str = "train",
    shuffle: bool = True,
    batch_size: int = 128,
    downsample_factor: float = 1.0,
    seed: int = 42,
) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Load CIFAR-10 dataset with preprocessing.

    Args:
        split: Dataset split ('train' or 'test')
        shuffle: Whether to shuffle the data
        batch_size: Batch size for the dataset
        downsample_factor: Fraction of data to use (1.0 = use all, 0.5 = use half).
                          Only applied to 'train' split. Uses deterministic seeding.
        seed: Random seed for deterministic downsampling

    Returns:
        List of batches, each containing (images, labels) as JAX arrays
        Images are normalized, shape (batch, 32, 32, 3)
        Labels are one-hot encoded to (batch, 10)
    """
    ds = tfds.load("cifar10", split=split, as_supervised=True)
    
    # For training split, apply deterministic downsampling if requested
    if split == "train" and downsample_factor < 1.0:
        # Load all data first (convert to list of numpy arrays)
        all_images = []
        all_labels = []
        for img, lbl in tfds.as_numpy(ds):
            # Ensure we have proper array shapes (handle both single samples and batches)
            img_array = np.array(img)
            lbl_array = np.array(lbl)
            
            # If single sample (0D or 1D), add batch dimension
            if img_array.ndim == 3:  # (H, W, C) -> (1, H, W, C)
                img_array = img_array[np.newaxis, ...]
            if lbl_array.ndim == 0:  # scalar -> (1,)
                lbl_array = lbl_array[np.newaxis]
            
            all_images.append(img_array)
            all_labels.append(lbl_array)
        
        # Concatenate all data
        if all_images:
            all_images = np.concatenate(all_images, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            
            # Deterministic downsampling using seeded random permutation
            n_samples = len(all_images)
            n_keep = int(n_samples * downsample_factor)
            
            # Use numpy random with seed for deterministic random selection
            rng = np.random.RandomState(seed)
            indices = rng.permutation(n_samples)[:n_keep]
            # Note: indices are in random order (from permutation), which is fine since we shuffle anyway
            
            # Select downsampled data
            all_images = all_images[indices]
            all_labels = all_labels[indices]
            
            # Convert back to tf.data.Dataset format for batching
            ds = tf.data.Dataset.from_tensor_slices((all_images, all_labels))
        else:
            # Empty dataset - shouldn't happen but handle gracefully
            ds = tf.data.Dataset.from_tensor_slices((np.array([]), np.array([])))
    
    if shuffle:
        ds = ds.shuffle(10000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    # Convert to JAX arrays with preprocessing
    batches = []
    for img_batch, lbl_batch in tfds.as_numpy(ds):
        # Convert to JAX arrays
        images = jnp.array(img_batch, dtype=jnp.float32)
        labels = jnp.array(lbl_batch, dtype=jnp.int32)
        
        # Preprocess: normalize images (keep 4D shape for CNNs)
        images = preprocessing.normalize_images(images)
        
        # One-hot encode labels
        labels = preprocessing.one_hot_encode(labels, num_classes=10)
        
        batches.append((images, labels))
    
    return batches


def load_cifar100(
    split: str = "train",
    shuffle: bool = True,
    batch_size: int = 128,
) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Load CIFAR-100 dataset with preprocessing (used as OOD dataset).

    Args:
        split: Dataset split ('train' or 'test')
        shuffle: Whether to shuffle the data
        batch_size: Batch size for the dataset

    Returns:
        List of batches, each containing (images, labels) as JAX arrays
        Images are normalized, shape (batch, 32, 32, 3)
        Labels are kept as integers (not one-hot) for OOD detection
    """
    ds = tfds.load("cifar100", split=split, as_supervised=True)
    if shuffle:
        ds = ds.shuffle(10000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    # Convert to JAX arrays with preprocessing
    batches = []
    for img_batch, lbl_batch in tfds.as_numpy(ds):
        # Convert to JAX arrays
        images = jnp.array(img_batch, dtype=jnp.float32)
        labels = jnp.array(lbl_batch, dtype=jnp.int32)
        
        # Preprocess: normalize images (keep 4D shape for CNNs)
        images = preprocessing.normalize_images(images)
        
        # For OOD detection, we don't need one-hot encoding
        # Labels are kept as integers
        batches.append((images, labels))
    
    return batches


def load_cifar10_corrupted(
    corruption_type: str = "gaussian_noise",
    severity: int = 1,
    shuffle: bool = False,
    batch_size: int = 128,
) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Load CIFAR-10-C (corrupted CIFAR-10) dataset with preprocessing.
    
    CIFAR-10-C contains 19 corruption types at 5 severity levels.
    This function loads a specific corruption type and severity level.

    Args:
        corruption_type: Type of corruption (e.g., 'gaussian_noise', 'shot_noise', 
                        'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
                        'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast',
                        'elastic_transform', 'pixelate', 'jpeg_compression', 'speckle_noise',
                        'gaussian_blur', 'spatter', 'saturate')
        severity: Severity level (1-5)
        shuffle: Whether to shuffle the data
        batch_size: Batch size for the dataset

    Returns:
        List of batches, each containing (images, labels) as JAX arrays
        Images are normalized, shape (batch, 32, 32, 3)
        Labels are one-hot encoded to (batch, 10)
    """
    print(f"Loading CIFAR-10-C: {corruption_type}, severity {severity}...")
    print("Note: First-time download may take several minutes (~2.7GB). Subsequent runs will be fast.")
    
    try:
        # CIFAR-10-C in tensorflow_datasets uses builder configs
        # Format: "cifar10_corrupted/{corruption_type}_{severity}"
        builder_name = f"cifar10_corrupted/{corruption_type}_{severity}"
        
        # Load the dataset with the specific builder name
        ds = tfds.load(
            builder_name,
            split="test",
            as_supervised=True,
            download=True,
        )
    except Exception as e:
        # If tensorflow_datasets doesn't have it, provide helpful error message
        raise ValueError(
            f"Failed to load CIFAR-10-C with corruption_type='{corruption_type}', "
            f"severity={severity}.\n"
            f"Error: {e}\n\n"
            f"To use CIFAR-10-C, you may need to:\n"
            f"1. Install: pip install tensorflow-datasets[cifar10_corrupted]\n"
            f"2. Or download CIFAR-10-C manually from: "
            f"https://github.com/hendrycks/robustness\n"
            f"3. Check that the dataset builder name is correct in your tfds version."
        )
    
    if shuffle:
        ds = ds.shuffle(10000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    # Convert to JAX arrays with preprocessing
    batches = []
    for img_batch, lbl_batch in tfds.as_numpy(ds):
        # Convert to JAX arrays
        images = jnp.array(img_batch, dtype=jnp.float32)
        labels = jnp.array(lbl_batch, dtype=jnp.int32)
        
        # Preprocess: normalize images (keep 4D shape for CNNs)
        images = preprocessing.normalize_images(images)
        
        # One-hot encode labels (CIFAR-10-C uses same labels as CIFAR-10)
        labels = preprocessing.one_hot_encode(labels, num_classes=10)
        
        batches.append((images, labels))
    
    return batches

