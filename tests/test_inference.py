"""Inference-side tests: bundle loading, output shape, and preprocessing parity.

The backend deliberately does not depend on torch, so predictor.py reimplements the
evaluation transform with PIL and numpy. That reimplementation is the riskiest part
of the deployment path - if it drifts from the torchvision version used at training
time the model silently sees different pixels than it was validated on. The parity
test below is the reason that reimplementation is safe to ship.
"""

import numpy as np
import pytest
from PIL import Image


def test_class_mapping_loads_and_matches_the_training_config(predictor):
    from src import config

    assert predictor.classes == config.BINARY_CLASSES
    assert predictor.positive_class == config.POSITIVE_CLASS
    assert set(predictor.source_classes) == set(config.SOURCE_CLASSES)


def test_predict_returns_the_documented_shape(predictor, sample_leaf_path):
    result = predictor.predict(Image.open(sample_leaf_path))

    assert set(result) == {
        "predicted_label", "confidence", "low_confidence",
        "confidence_threshold", "top_predictions", "message",
    }
    assert result["predicted_label"] in predictor.classes
    assert isinstance(result["message"], str) and result["message"]


def test_confidence_is_a_probability_and_tops_the_ranking(predictor, sample_leaf_path):
    result = predictor.predict(Image.open(sample_leaf_path))

    assert 0.0 <= result["confidence"] <= 1.0
    assert result["confidence"] == max(p["confidence"] for p in result["top_predictions"])
    assert result["top_predictions"][0]["label"] == result["predicted_label"]


def test_probabilities_cover_every_class_and_sum_to_one(predictor, sample_leaf_path):
    result = predictor.predict(Image.open(sample_leaf_path))
    probabilities = [p["confidence"] for p in result["top_predictions"]]

    assert len(probabilities) == len(predictor.classes)
    assert probabilities == sorted(probabilities, reverse=True)
    assert sum(probabilities) == pytest.approx(1.0, abs=1e-3)


def test_low_confidence_flag_follows_the_threshold(predictor, sample_leaf_path):
    result = predictor.predict(Image.open(sample_leaf_path))
    assert result["low_confidence"] == (result["confidence"] < result["confidence_threshold"])


def test_grayscale_and_rgba_uploads_are_accepted(predictor):
    """Real uploads are not always RGB; the predictor converts rather than crashing."""
    for mode, colour in (("L", 128), ("RGBA", (60, 110, 70, 255)), ("P", 4)):
        image = Image.new(mode, (300, 220), colour)
        result = predictor.predict(image)
        assert result["predicted_label"] in predictor.classes


def test_non_square_input_is_handled(predictor):
    result = predictor.predict(Image.new("RGB", (640, 200), (70, 120, 80)))
    assert 0.0 <= result["confidence"] <= 1.0


def test_prediction_is_deterministic(predictor, sample_leaf_path):
    """The deployed API must return the same answer for the same upload."""
    first = predictor.predict(Image.open(sample_leaf_path))
    second = predictor.predict(Image.open(sample_leaf_path))
    assert first == second


def test_preprocessing_matches_the_torchvision_eval_transform(predictor, sample_leaf_path):
    """The claim predictor.py makes about its PIL/numpy reimplementation, verified.

    torchvision is a training-time dependency and is not installed on the deployed
    host, so this runs only where it is available.
    """
    torchvision = pytest.importorskip("torchvision")  # noqa: F841

    from src.preprocessing.transforms import eval_transform

    image = Image.open(sample_leaf_path).convert("RGB")
    reference = eval_transform()(image).numpy()[None, ...]
    ours = predictor.preprocess(Image.open(sample_leaf_path))

    assert ours.shape == reference.shape
    assert np.abs(ours - reference).max() < 2e-2
