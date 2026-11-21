"""Loads the photos from disk and describes the dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf

from .config import CLASS_NAMES, Config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


class DatasetError(RuntimeError):
    """The data folders are missing or too small to train on."""


@dataclass
class Datasets:
    train: tf.data.Dataset
    valid: tf.data.Dataset
    test: tf.data.Dataset
    train_counts: Dict[str, int]
    test_counts: Dict[str, int]

    def class_weights(self) -> Dict[int, float]:
        """Weights that make a rare class count as much as a common one while training."""
        counts = [self.train_counts[name] for name in CLASS_NAMES]
        total = sum(counts)
        return {index: total / (len(counts) * count) for index, count in enumerate(counts)}


def list_images(folder: Path) -> List[Path]:
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def count_images(split_dir: Path) -> Dict[str, int]:
    """Number of photos per class folder, for example {'ok_front': 2, 'def_front': 6}."""
    return {name: len(list_images(split_dir / name)) for name in CLASS_NAMES}


def check_layout(config: Config) -> None:
    """Fails early, with a message that says what to create, instead of deep inside TensorFlow."""
    for split_dir in (config.train_dir, config.test_dir):
        for name in CLASS_NAMES:
            folder = split_dir / name
            if not folder.is_dir():
                raise DatasetError(f"Missing folder: {folder}\nCreate it and put the photos in it (see Docs/Dataset.md).")
            if not list_images(folder):
                raise DatasetError(f"No photos in {folder}")
    small = [name for name, count in count_images(config.train_dir).items() if count < 2]
    if small:
        raise DatasetError(f"Need at least two training photos in each class, but {', '.join(small)} has fewer.")


def load_gray(path: Path, size) -> np.ndarray:
    """One photo as a float array of shape (height, width) with values 0..255."""
    image = tf.keras.utils.load_img(path, color_mode="grayscale", target_size=size)
    return np.asarray(image, dtype="float32")


def load_datasets(config: Config) -> Datasets:
    check_layout(config)
    common = dict(
        labels="inferred",
        label_mode="binary",
        class_names=list(CLASS_NAMES),
        color_mode="grayscale",
        image_size=config.image_size,
        batch_size=config.batch_size,
    )
    train = tf.keras.utils.image_dataset_from_directory(
        config.train_dir, validation_split=config.validation_split, subset="training", seed=config.seed, **common
    )
    valid = tf.keras.utils.image_dataset_from_directory(
        config.train_dir, validation_split=config.validation_split, subset="validation", seed=config.seed, **common
    )
    test = tf.keras.utils.image_dataset_from_directory(config.test_dir, shuffle=False, **common)
    autotune = tf.data.AUTOTUNE
    return Datasets(
        train=train.cache().prefetch(autotune),
        valid=valid.cache().prefetch(autotune),
        test=test,
        train_counts=count_images(config.train_dir),
        test_counts=count_images(config.test_dir),
    )
