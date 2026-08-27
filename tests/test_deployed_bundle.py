"""End-to-end check that the shipped bundle behaves like the reported model.

Everything else tests a component. This tests the artefact that actually gets
deployed - models/export/ - against the same held-out images the report quotes, and
fails if the two disagree.

The failure it exists to catch is a silently inverted class mapping. If `healthy`
and `diseased` were swapped between training and export, every unit test would still
pass, the API would still return well-formed JSON, and every prediction would be
exactly wrong.
"""

import json

import pytest
from PIL import Image

from src import config
from src.data.dataset import leaf_grouped_split, load_manifest


@pytest.fixture(scope="module")
def test_rows():
    if not config.MANIFEST_PATH.exists():
        pytest.skip("dataset not prepared")
    return leaf_grouped_split(load_manifest())[2]


@pytest.fixture(scope="module")
def reported():
    path = config.METRICS_DIR / "mobilenet_leaf_test.json"
    if not path.exists():
        pytest.skip("no test metrics to compare against")
    return json.load(open(path, encoding="utf-8"))


def test_bundle_reproduces_the_reported_accuracy(predictor, test_rows, reported):
    """Run the exported model over the full test split and compare.

    Tolerance is one image: ONNX and torch differ by ~1e-6 in the logits, which can
    flip a sample sitting exactly on the boundary, but nothing more than that.
    """
    correct = 0
    for row in test_rows:
        with Image.open(config.RAW_DIR / row["rel_path"]) as img:
            result = predictor.predict(img)
        if result["predicted_label"] == row["binary_label"]:
            correct += 1

    accuracy = correct / len(test_rows)
    tolerance = 1.5 / len(test_rows)
    assert abs(accuracy - reported["accuracy"]) <= tolerance, (
        f"deployed bundle scores {accuracy:.4f} but the report claims "
        f"{reported['accuracy']:.4f} - the exported model is not the evaluated one"
    )


def test_labels_are_not_inverted(predictor, test_rows):
    """A sanity check that survives even if the metrics file is missing.

    An inverted mapping would put accuracy near zero rather than near one, so a
    coarse threshold catches it without re-asserting the exact score.
    """
    sample = test_rows[:120]
    correct = sum(
        predictor.predict(Image.open(config.RAW_DIR / r["rel_path"]))["predicted_label"]
        == r["binary_label"]
        for r in sample
    )
    assert correct / len(sample) > 0.5, "labels look inverted between training and export"


def test_healthy_images_are_called_healthy(predictor, test_rows):
    """The two classes are checked separately: a model that answered a single label
    for everything would still pass an overall accuracy check on an unbalanced slice."""
    for label in config.BINARY_CLASSES:
        subset = [r for r in test_rows if r["binary_label"] == label][:40]
        assert subset, f"no {label} images in the test split"
        hits = sum(
            predictor.predict(Image.open(config.RAW_DIR / r["rel_path"]))["predicted_label"]
            == label
            for r in subset
        )
        assert hits / len(subset) > 0.8, f"{label} images are mostly not predicted {label}"


def test_exported_config_matches_training_config(predictor):
    """Preprocessing shipped with the model must match what training used."""
    assert predictor.image_size == config.IMAGE_SIZE
    assert predictor.resize == config.RESIZE_BEFORE_CROP
    assert predictor.classes == config.BINARY_CLASSES
