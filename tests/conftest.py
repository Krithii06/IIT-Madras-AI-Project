import io
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"

# The backend is deployed with backend/ as its working directory, so `app` is a
# top-level package there. Mirror that here instead of rewriting the imports.
for path in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

EXPORT_DIR = REPO_ROOT / "models" / "export"


@pytest.fixture(scope="session")
def export_dir():
    if not (EXPORT_DIR / "model.onnx").exists():
        pytest.skip("no exported model; run src.inference.export first")
    return EXPORT_DIR


@pytest.fixture(scope="session")
def predictor(export_dir):
    from src.inference.predictor import LeafPredictor

    return LeafPredictor(export_dir)


@pytest.fixture(scope="session")
def sample_leaf_path():
    """A real image from the prepared dataset, so tests exercise the true input."""
    raw = REPO_ROOT / "data" / "raw"
    candidates = sorted(raw.glob("*/*.JPG")) if raw.exists() else []
    if not candidates:
        pytest.skip("dataset not prepared; run src.data.prepare first")
    return candidates[0]


def make_image_bytes(size=(256, 256), colour=(60, 110, 70), fmt="JPEG"):
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def client(export_dir):
    from fastapi.testclient import TestClient

    from app.main import app

    # TestClient runs the lifespan handler, which is what loads the model.
    with TestClient(app) as test_client:
        yield test_client
