"""Trains the model and keeps the best version."""
from __future__ import annotations

from typing import Dict, List

from tensorflow import keras

from .config import Config
from .data import Datasets
from .model import build_model, set_seed


def train(config: Config, datasets: Datasets, verbose: int = 2):
    """Returns the best model (lowest validation loss) and the history of every epoch."""
    set_seed(config.seed)
    model = build_model(config)
    config.model_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(config.model_path), monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.patience, restore_best_weights=True),
    ]
    history = model.fit(
        datasets.train,
        validation_data=datasets.valid,
        epochs=config.epochs,
        class_weight=datasets.class_weights(),
        callbacks=callbacks,
        verbose=verbose,
    )
    return model, history_dict(history)


def history_dict(history) -> Dict[str, List[float]]:
    return {key: [float(value) for value in values] for key, values in history.history.items()}
