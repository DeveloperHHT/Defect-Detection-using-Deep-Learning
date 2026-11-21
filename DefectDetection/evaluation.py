"""Scores a model on the test photos."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .config import Config


@dataclass
class Evaluation:
    """Predictions for the test set. A photo counts as defective when its probability reaches the threshold."""

    paths: List[Path]
    y_true: np.ndarray
    y_prob: np.ndarray
    threshold: float = 0.5

    @property
    def y_pred(self) -> np.ndarray:
        return (self.y_prob >= self.threshold).astype(int)

    @property
    def confusion(self) -> np.ndarray:
        """Rows are the real class, columns the predicted one: [[OK->OK, OK->Defective], [Defective->OK, Defective->Defective]]."""
        matrix = np.zeros((2, 2), dtype=int)
        for real, predicted in zip(self.y_true, self.y_pred):
            matrix[int(real), int(predicted)] += 1
        return matrix

    @property
    def accuracy(self) -> float:
        return float(np.mean(self.y_true == self.y_pred)) if len(self.y_true) else 0.0

    def _ratio(self, numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    @property
    def precision(self) -> float:
        """Of the photos called defective, how many really were."""
        return self._ratio(self.confusion[1, 1], self.confusion[:, 1].sum())

    @property
    def recall(self) -> float:
        """Of the really defective photos, how many were found."""
        return self._ratio(self.confusion[1, 1], self.confusion[1].sum())

    @property
    def f1(self) -> float:
        return self._ratio(2 * self.precision * self.recall, self.precision + self.recall)

    @property
    def misclassified(self) -> np.ndarray:
        return np.flatnonzero(self.y_pred != self.y_true)

    def report(self) -> str:
        c = self.confusion
        return (
            f"Test photos: {len(self.y_true)}\n"
            f"Accuracy:  {self.accuracy:.1%}\n"
            f"Precision: {self.precision:.1%}   (of the photos called defective, how many were)\n"
            f"Recall:    {self.recall:.1%}   (of the defective photos, how many were found)\n"
            f"F1 score:  {self.f1:.1%}\n"
            f"Confusion: OK {c[0, 0]} right / {c[0, 1]} wrong,  Defective {c[1, 1]} right / {c[1, 0]} wrong"
        )


def evaluate(model, datasets, config: Config) -> Evaluation:
    """Runs the model over the test set. The set is not shuffled, so file order matches the predictions."""
    test = datasets.test
    probabilities = model.predict(test, verbose=0).reshape(-1)
    labels = np.concatenate([batch.numpy().reshape(-1) for _, batch in test]).astype(int)
    paths = [Path(p) for p in test.file_paths]
    return Evaluation(paths=paths, y_true=labels, y_prob=probabilities, threshold=config.threshold)
