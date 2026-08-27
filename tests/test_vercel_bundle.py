"""The Vercel function directory holds copies. These assert they are still copies.

frontend/api/ cannot import from src/ or models/ because Vercel builds with
frontend/ as its root, so the predictor and the model bundle are vendored in by
src/inference/sync_vercel.py. Copies rot: retrain the model or edit the
preprocessing and the deployed function keeps serving the old one, with nothing
failing anywhere. These tests are what make that impossible.
"""

import hashlib
import re

import pytest

from src import config
from src.inference import sync_vercel


def digest(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_vendored_predictor_matches_the_source():
    if not sync_vercel.VENDORED_PREDICTOR.exists():
        pytest.skip("vercel bundle not synced; run src.inference.sync_vercel")
    assert digest(sync_vercel.VENDORED_PREDICTOR) == digest(sync_vercel.SOURCE_PREDICTOR), (
        "frontend/api/_predictor.py has drifted from src/inference/predictor.py - "
        "re-run python -m src.inference.sync_vercel"
    )


def test_vendored_model_matches_the_export():
    if not sync_vercel.VENDORED_MODEL.exists():
        pytest.skip("vercel bundle not synced; run src.inference.sync_vercel")
    for name in sync_vercel.BUNDLE_FILES:
        vendored = sync_vercel.VENDORED_MODEL / name
        source = sync_vercel.SOURCE_MODEL / name
        if not source.exists():
            pytest.skip("no exported model to compare against")
        assert vendored.exists(), f"{name} missing from the Vercel bundle"
        assert digest(vendored) == digest(source), (
            f"frontend/api/model/{name} has drifted from models/export/{name} - "
            f"re-run python -m src.inference.sync_vercel"
        )


def test_serverless_functions_exist_for_every_route():
    """The rewrites in vercel.json map these three paths onto these three files."""
    api_dir = sync_vercel.API_DIR
    for name in ("predict.py", "health.py", "model-info.py"):
        assert (api_dir / name).exists(), f"frontend/api/{name} is missing"


def test_runtime_requirements_carry_no_training_dependencies():
    """torch in here would blow past the Vercel function size limit."""
    requirements = (config.PROJECT_ROOT / "frontend" / "requirements.txt").read_text(
        encoding="utf-8").lower()
    for forbidden in ("torch", "torchvision", "matplotlib", "scikit-learn", "pandas"):
        assert forbidden not in requirements, (
            f"{forbidden} must not ship to the serverless function")
    for needed in ("onnxruntime", "numpy", "pillow"):
        assert needed in requirements


def test_onnxruntime_is_new_enough_for_the_interpreter_vercel_picks():
    """Three builds died on this, so it is worth a test.

    Vercel selects its own interpreter - CPython 3.14 at the time of writing - and
    ignores .python-version in both the repository root and the configured root
    directory. onnxruntime only began publishing cp314 wheels at 1.24.3, so anything
    older cannot be resolved there no matter where the pin is written.
    """
    runtime = (config.PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    match = re.search(r"onnxruntime==(\d+)\.(\d+)\.(\d+)", runtime)
    assert match, "requirements.txt must pin onnxruntime explicitly"

    version = tuple(int(g) for g in match.groups())
    assert version >= (1, 24, 3), (
        f"onnxruntime {'.'.join(map(str, version))} ships no cp314 wheel; "
        f"the Vercel build will fail to resolve dependencies"
    )


def test_runtime_pins_agree_across_every_file_that_declares_them():
    """A mismatch means local behaviour diverging from deployed behaviour."""
    declarations = {
        "requirements.txt": config.PROJECT_ROOT / "requirements.txt",
        "frontend/requirements.txt": config.PROJECT_ROOT / "frontend" / "requirements.txt",
        "frontend/pyproject.toml": config.PROJECT_ROOT / "frontend" / "pyproject.toml",
        "backend/requirements.txt": config.PROJECT_ROOT / "backend" / "requirements.txt",
    }
    found = {}
    for label, path in declarations.items():
        if not path.exists():
            continue
        match = re.search(r"onnxruntime==(\d+\.\d+\.\d+)", path.read_text(encoding="utf-8"))
        if match:
            found[label] = match.group(1)

    assert len(set(found.values())) == 1, f"onnxruntime pins disagree: {found}"


def test_vercel_config_rewrites_the_documented_paths():
    import json

    config_path = config.PROJECT_ROOT / "frontend" / "vercel.json"
    assert config_path.exists(), "frontend/vercel.json is missing"
    rewrites = json.loads(config_path.read_text(encoding="utf-8"))["rewrites"]
    mapping = {r["source"]: r["destination"] for r in rewrites}
    assert mapping["/predict"] == "/api/predict"
    assert mapping["/health"] == "/api/health"
    assert mapping["/model-info"] == "/api/model-info"
