"""The network: two convolution blocks and a small classifier, with augmentation built in."""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import Config


def build_model(config: Config) -> keras.Model:
    """Takes a grayscale photo with values 0..255 and returns the probability that it is defective.

    Scaling and augmentation are part of the model, so a saved model can be used on a raw photo.
    The augmentation layers only act while training.
    """
    height, width = config.image_size
    model = keras.Sequential(
        [
            layers.Input(shape=(height, width, 1)),
            layers.Rescaling(1.0 / 255),
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.1),
            layers.Conv2D(32, 3, strides=2, padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=2, strides=2),
            layers.Conv2D(64, 3, strides=2, padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=2, strides=2),
            layers.Flatten(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="casting_cnn",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_model(path) -> keras.Model:
    return keras.models.load_model(path)


def set_seed(seed: int) -> None:
    """Makes a run repeatable (as far as TensorFlow allows)."""
    keras.utils.set_random_seed(seed)
