"""Settings and constants shared by every part of the project."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent.parent

# Folder names inside `Data/train` and `Data/test`. The position is the label: 0 = OK, 1 = defective.
CLASS_NAMES: Tuple[str, str] = ("ok_front", "def_front")
LABELS = {0: "OK", 1: "Defective"}

# Colours used by every plot, so the figures look like one set.
COLOR_OK = "#2e9e6b"
COLOR_DEFECT = "#e5484d"
COLOR_ACCENT = "#5b6ee1"


@dataclass(frozen=True)
class Config:
    """Everything that can be tuned. Command line options override these defaults."""

    data_dir: Path = ROOT / "Data"
    model_path: Path = ROOT / "Models" / "casting_cnn.keras"
    figures_dir: Path = ROOT / "Docs" / "Images"
    image_size: Tuple[int, int] = (300, 300)
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 1e-3
    validation_split: float = 0.25
    patience: int = 15
    threshold: float = 0.5
    seed: int = 0

    @property
    def train_dir(self) -> Path:
        return self.data_dir / "train"

    @property
    def test_dir(self) -> Path:
        return self.data_dir / "test"
