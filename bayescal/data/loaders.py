"""Data loaders for MNIST and Fashion-MNIST datasets."""

from typing import Any, List, Tuple

import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from bayescal.data import preprocessing


def load_mnist(
    split: str = "train",
    shuffle: bool = True,
    batch_size: int = 128,
) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Load MNIST dataset with preprocessing.

    Args:
        split: Dataset split ('train' or 'test')
        shuffle: Whether to shuffle the data
        batch_size: Batch size for the dataset

    Returns:
        List of batches, each containing (images, labels) as JAX arrays
        Images are normalized, flattened to (batch, 784)
        Labels are one-hot encoded to (batch, 10)
    """
    ds = tfds.load("mnist", split=split, as_supervised=True)
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
        
        # Preprocess: normalize and flatten images
        images = preprocessing.normalize_images(images)
        images = preprocessing.flatten_images(images)
        
        # One-hot encode labels
        labels = preprocessing.one_hot_encode(labels, num_classes=10)
        
        batches.append((images, labels))
    
    return batches


def load_fashion_mnist(
    split: str = "train",
    shuffle: bool = True,
    batch_size: int = 128,
) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Load Fashion-MNIST dataset with preprocessing.

    Args:
        split: Dataset split ('train' or 'test')
        shuffle: Whether to shuffle the data
        batch_size: Batch size for the dataset

    Returns:
        List of batches, each containing (images, labels) as JAX arrays
        Images are normalized, flattened to (batch, 784)
        Labels are one-hot encoded to (batch, 10)
    """
    ds = tfds.load("fashion_mnist", split=split, as_supervised=True)
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
        
        # Preprocess: normalize and flatten images
        images = preprocessing.normalize_images(images)
        images = preprocessing.flatten_images(images)
        
        # One-hot encode labels
        labels = preprocessing.one_hot_encode(labels, num_classes=10)
        
        batches.append((images, labels))
    
    return batches

