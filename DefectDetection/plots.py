"""Figures for the README and for looking at results. Each function returns a matplotlib figure."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .config import CLASS_NAMES, COLOR_ACCENT, COLOR_DEFECT, COLOR_OK, LABELS, Config
from .data import list_images, load_gray
from .evaluation import Evaluation

INK = "#1f2430"
MUTED = "#6b7280"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#d5d9e2",
            "axes.labelcolor": INK,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": "#eceef4",
            "font.size": 11,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def _label_color(label: int) -> str:
    return COLOR_DEFECT if label == 1 else COLOR_OK


def plot_samples(config: Config, per_class: int = 3):
    """A few training photos of each class, side by side."""
    apply_style()
    fig, axes = plt.subplots(2, per_class, figsize=(3.2 * per_class, 6.6))
    for row, name in enumerate(CLASS_NAMES):
        paths = list_images(config.train_dir / name)
        for column in range(per_class):
            ax = axes[row, column]
            ax.axis("off")
            if column < len(paths):
                ax.imshow(load_gray(paths[column], config.image_size), cmap="gray", vmin=0, vmax=255)
            if column == 0:
                ax.set_title(LABELS[row], color=_label_color(row), loc="left")
    fig.suptitle("Training photos", x=0.02, ha="left", fontweight="bold", fontsize=14)
    fig.tight_layout()
    return fig


def plot_distribution(train_counts: Dict[str, int], test_counts: Dict[str, int]):
    """How many photos each class has in the training and the test folder."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 4))
    splits = ("Train", "Test")
    ok = [train_counts[CLASS_NAMES[0]], test_counts[CLASS_NAMES[0]]]
    defect = [train_counts[CLASS_NAMES[1]], test_counts[CLASS_NAMES[1]]]
    x = np.arange(len(splits))
    bars_ok = ax.bar(x - 0.18, ok, 0.36, color=COLOR_OK, label="OK")
    bars_def = ax.bar(x + 0.18, defect, 0.36, color=COLOR_DEFECT, label="Defective")
    for bar in list(bars_ok) + list(bars_def):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, int(bar.get_height()), ha="center", va="bottom", fontweight="bold")
    ax.set_xticks(x, splits)
    ax.set_ylabel("Photos")
    ax.set_title("Class distribution")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    return fig


def plot_history(history: Dict[str, Sequence[float]]):
    """Loss and accuracy per epoch, for the training and the validation photos."""
    apply_style()
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in ((left, "loss", "Loss"), (right, "accuracy", "Accuracy")):
        ax.plot(epochs, history[metric], color=COLOR_ACCENT, linewidth=2, label="Training")
        if f"val_{metric}" in history:
            ax.plot(epochs, history[f"val_{metric}"], color=COLOR_DEFECT, linewidth=2, linestyle="--", label="Validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True)
        ax.legend(frameon=False)
    right.set_ylim(0, 1.05)
    fig.suptitle("Learning curve", x=0.02, ha="left", fontweight="bold", fontsize=14)
    fig.tight_layout()
    return fig


def plot_confusion(evaluation: Evaluation):
    """Where the model was right and wrong on the test photos."""
    apply_style()
    matrix = evaluation.confusion
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), 1))
    for (row, column), value in np.ndenumerate(matrix):
        dark = value > matrix.max() / 2
        ax.text(column, row, int(value), ha="center", va="center", fontsize=20, fontweight="bold", color="white" if dark else INK)
    ax.set_xticks([0, 1], [LABELS[0], LABELS[1]])
    ax.set_yticks([0, 1], [LABELS[0], LABELS[1]])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig


def _prediction_grid(evaluation: Evaluation, indices: Sequence[int], config: Config, title: str, columns: int = 3):
    apply_style()
    count = max(len(indices), 1)
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.4 * columns, 4.3 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, index in zip(axes.flat, indices):
        real = int(evaluation.y_true[index])
        probability = float(evaluation.y_prob[index])
        predicted = int(evaluation.y_pred[index])
        confidence = probability if predicted == 1 else 1 - probability
        ax.imshow(load_gray(evaluation.paths[index], config.image_size), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Actual: {LABELS[real]}", loc="left", fontsize=11)
        ax.text(
            0, -0.04, f"Predicted: {LABELS[predicted]} ({confidence:.0%})", transform=ax.transAxes, va="top",
            color=COLOR_OK if predicted == real else COLOR_DEFECT, fontweight="bold",
        )
    if not len(indices):
        axes.flat[0].text(0.5, 0.5, "Nothing to show", ha="center", va="center", color=MUTED, transform=axes.flat[0].transAxes)
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold", fontsize=14)
    fig.tight_layout(h_pad=2.5)
    return fig


def plot_predictions(evaluation: Evaluation, config: Config, limit: int = 6):
    """The first test photos with what the model said about each."""
    return _prediction_grid(evaluation, list(range(min(limit, len(evaluation.y_true)))), config, "Predictions on test photos")


def plot_misclassified(evaluation: Evaluation, config: Config, limit: int = 6):
    """The test photos the model got wrong. Empty when there were none."""
    wrong = list(evaluation.misclassified[:limit])
    title = "Misclassified test photos" if wrong else "Misclassified test photos: none"
    return _prediction_grid(evaluation, wrong, config, title)


def save(fig, folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    fig.savefig(path)
    plt.close(fig)
    return path
