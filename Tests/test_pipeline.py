"""Runs the whole pipeline on the sample photos, with a tiny model so it stays quick."""
from dataclasses import replace

import numpy as np
import pytest

from DefectDetection import plots
from DefectDetection.config import CLASS_NAMES, Config
from DefectDetection.data import DatasetError, check_layout, count_images, load_datasets
from DefectDetection.evaluation import evaluate
from DefectDetection.model import build_model, set_seed
from DefectDetection.predictor import DefectPredictor
from DefectDetection.training import train


@pytest.fixture(scope="module")
def config(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    return replace(Config(), epochs=2, batch_size=4, image_size=(64, 64), patience=2,
                   model_path=out / "model.keras", figures_dir=out / "figures")


@pytest.fixture(scope="module")
def trained(config):
    set_seed(config.seed)
    datasets = load_datasets(config)
    model, history = train(config, datasets, verbose=0)
    return datasets, model, history


def test_sample_data_is_found(config):
    counts = count_images(config.train_dir)
    assert set(counts) == set(CLASS_NAMES)
    assert all(n >= 2 for n in counts.values())


def test_missing_folder_gives_clear_error(tmp_path):
    with pytest.raises(DatasetError):
        check_layout(replace(Config(), data_dir=tmp_path))


def test_model_output_is_one_probability(config):
    model = build_model(config)
    out = model.predict(np.zeros((3, *config.image_size, 1), dtype="float32"), verbose=0)
    assert out.shape == (3, 1)
    assert ((out >= 0) & (out <= 1)).all()


def test_training_saves_model_and_history(config, trained):
    _, _, history = trained
    assert config.model_path.exists()
    assert len(history["loss"]) >= 1


def test_evaluation_numbers_add_up(config, trained):
    datasets, model, _ = trained
    result = evaluate(model, datasets, config)
    assert result.confusion.sum() == len(result.y_true)
    assert 0 <= result.accuracy <= 1
    assert "Accuracy" in result.report()


def test_predictor_labels_every_photo(config, trained):
    datasets, model, _ = trained
    paths = [p for p in (config.test_dir / CLASS_NAMES[0]).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    results = DefectPredictor(config, model).predict(paths)
    assert len(results) == len(paths)
    assert all(r.label in {"OK", "Defective"} and 0.5 <= r.confidence <= 1 for r in results)


def test_figures_are_written(config, trained, tmp_path):
    datasets, model, history = trained
    result = evaluate(model, datasets, config)
    for name, fig in {
        "samples": plots.plot_samples(config),
        "distribution": plots.plot_distribution(datasets.train_counts, datasets.test_counts),
        "curve": plots.plot_history(history),
        "confusion": plots.plot_confusion(result),
        "predictions": plots.plot_predictions(result, config),
        "wrong": plots.plot_misclassified(result, config),
    }.items():
        assert plots.save(fig, tmp_path, f"{name}.png").stat().st_size > 1000


def test_predict_command_finds_photos_in_subfolders():
    from DefectDetection.cli import _expand

    config = Config()
    assert len(_expand([config.test_dir])) == sum(count_images(config.test_dir).values())
