"""Data loaders for MNIST and Fashion-MNIST datasets."""

from typing import Tuple

import jax.numpy as jnp
import tensorflow_datasets as tfds


def load_mnist(
    split: str = "train",
    shuffle: bool = True,
    batch_size: int = 128,
) -> tfds.core.Dataset:
    """
    Load MNIST dataset.

    Args:
        split: Dataset split ('train' or 'test')
        shuffle: Whether to shuffle the data
        batch_size: Batch size for the dataset

    Returns:
        Batched TensorFlow dataset
    """
    ds = tfds.load("mnist", split=split, as_supervised=True)
    if shuffle:
        ds = ds.shuffle(10000)
    ds = ds.batch(batch_size)
    return ds.prefetch(tfds.AUTOTUNE)


def load_fashion_mnist(
    split: str = "train",
    shuffle: bool = True,
    batch_size: int = 128,
) -> tfds.core.Dataset:
    """
    Load Fashion-MNIST dataset.

    Args:
        split: Dataset split ('train' or 'test')
        shuffle: Whether to shuffle the data
        batch_size: Batch size for the dataset

    Returns:
        Batched TensorFlow dataset
    """
    ds = tfds.load("fashion_mnist", split=split, as_supervised=True)
    if shuffle:
        ds = ds.shuffle(10000)
    ds = ds.batch(batch_size)
    return ds.prefetch(tfds.AUTOTUNE)


def convert_to_jax(
    dataset: tfds.core.Dataset,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert TensorFlow dataset to JAX arrays.

    Args:
        dataset: TensorFlow dataset

    Returns:
        Tuple of (images, labels) as JAX arrays
    """
    images = []
    labels = []
    for img, lbl in tfds.as_numpy(dataset):
        images.append(img)
        labels.append(lbl)
    return jnp.array(images), jnp.array(labels)

