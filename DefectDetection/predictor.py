"""Uses a trained model on single photos."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np

from .config import Config, LABELS
from .data import load_gray
from .model import load_model


@dataclass(frozen=True)
class Prediction:
    path: Path
    defect_probability: float
    threshold: float

    @property
    def is_defective(self) -> bool:
        return self.defect_probability >= self.threshold

    @property
    def label(self) -> str:
        return LABELS[int(self.is_defective)]

    @property
    def confidence(self) -> float:
        """How sure the model is about the label it gave."""
        return self.defect_probability if self.is_defective else 1 - self.defect_probability


class DefectPredictor:
    def __init__(self, config: Config, model=None):
        self.config = config
        if model is None:
            if not config.model_path.exists():
                raise FileNotFoundError(f"No trained model at {config.model_path}. Run `python main.py train` first.")
            model = load_model(config.model_path)
        self.model = model

    def predict(self, paths: Iterable[Path]) -> List[Prediction]:
        paths = list(paths)
        if not paths:
            return []
        batch = np.stack([load_gray(path, self.config.image_size) for path in paths])[..., np.newaxis]
        probabilities = self.model.predict(batch, verbose=0).reshape(-1)
        return [Prediction(path, float(p), self.config.threshold) for path, p in zip(paths, probabilities)]
