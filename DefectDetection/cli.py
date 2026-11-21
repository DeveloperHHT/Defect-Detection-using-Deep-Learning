"""Command line: train a model, evaluate it, or check single photos."""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

from .config import ROOT, Config
from .data import IMAGE_SUFFIXES, DatasetError

DEFAULTS = Config()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py", description="Detect defective metal buttons in photos.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, data: bool = True) -> None:
        p.add_argument("--model", type=Path, default=DEFAULTS.model_path, help="where the model is saved or loaded from")
        p.add_argument("--threshold", type=float, default=DEFAULTS.threshold, help="probability from which a photo counts as defective")
        if data:
            p.add_argument("--data", type=Path, default=DEFAULTS.data_dir, help="folder with train/ and test/")
            p.add_argument("--figures", type=Path, default=DEFAULTS.figures_dir, help="where the figures are written")
            p.add_argument("--no-figures", action="store_true", help="do not write figures")
            p.add_argument("--show", action="store_true", help="also open the figures in windows")

    train = sub.add_parser("train", help="train a model, then evaluate it and draw the figures")
    common(train)
    train.add_argument("--epochs", type=int, default=DEFAULTS.epochs)
    train.add_argument("--batch-size", type=int, default=DEFAULTS.batch_size)
    train.add_argument("--seed", type=int, default=DEFAULTS.seed)

    evaluate = sub.add_parser("evaluate", help="score a saved model on the test photos")
    common(evaluate)

    predict = sub.add_parser("predict", help="check one or more photos (or folders of photos)")
    common(predict, data=False)
    predict.add_argument("paths", nargs="+", type=Path)
    return parser


def _config(args: argparse.Namespace) -> Config:
    config = replace(DEFAULTS, model_path=args.model, threshold=args.threshold)
    for field, option in (("data_dir", "data"), ("figures_dir", "figures"), ("epochs", "epochs"), ("batch_size", "batch_size"), ("seed", "seed")):
        if hasattr(args, option):
            config = replace(config, **{field: getattr(args, option)})
    return config


def _write_figures(config: Config, datasets, evaluation, history, show: bool) -> None:
    from . import plots

    figures = {"samples.png": plots.plot_samples(config), "distribution.png": plots.plot_distribution(datasets.train_counts, datasets.test_counts)}
    if history is not None:
        figures["learning_curve.png"] = plots.plot_history(history)
    figures["confusion_matrix.png"] = plots.plot_confusion(evaluation)
    figures["predictions.png"] = plots.plot_predictions(evaluation, config)
    figures["misclassified.png"] = plots.plot_misclassified(evaluation, config)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        fig.savefig(config.figures_dir / name)
        print(f"figure: {config.figures_dir / name}")
    if show:
        plots.plt.show()
    plots.plt.close("all")


def _run_train(args: argparse.Namespace, config: Config) -> int:
    from .data import load_datasets
    from .evaluation import evaluate
    from .training import train

    datasets = load_datasets(config)
    print(f"Training photos {datasets.train_counts}, test photos {datasets.test_counts}")
    model, history = train(config, datasets)
    evaluation = evaluate(model, datasets, config)
    print("\n" + evaluation.report())
    print(f"\nBest model saved to {config.model_path}")
    if not args.no_figures:
        _write_figures(config, datasets, evaluation, history, args.show)
    return 0


def _run_evaluate(args: argparse.Namespace, config: Config) -> int:
    from .data import load_datasets
    from .evaluation import evaluate
    from .predictor import DefectPredictor

    datasets = load_datasets(config)
    model = DefectPredictor(config).model
    evaluation = evaluate(model, datasets, config)
    print(evaluation.report())
    if not args.no_figures:
        _write_figures(config, datasets, evaluation, None, args.show)
    return 0


def _expand(paths: List[Path]) -> List[Path]:
    photos: List[Path] = []
    for path in paths:
        if path.is_dir():
            photos += sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file():
            photos.append(path)
        else:
            raise FileNotFoundError(f"No such file or folder: {path}")
    if not photos:
        raise FileNotFoundError("No photos found in the given paths")
    return photos


def _run_predict(args: argparse.Namespace, config: Config) -> int:
    from .predictor import DefectPredictor

    predictions = DefectPredictor(config).predict(_expand(args.paths))
    width = max(len(p.path.name) for p in predictions)
    for p in predictions:
        print(f"{p.path.name:<{width}}  {p.label:<9}  {p.confidence:.1%}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not getattr(args, "show", False):
        os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("KERAS_HOME", str(ROOT / ".keras"))  # keep Keras files inside the project, not in ~/.keras
    config = _config(args)
    runners = {"train": _run_train, "evaluate": _run_evaluate, "predict": _run_predict}
    try:
        return runners[args.command](args, config)
    except (DatasetError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
