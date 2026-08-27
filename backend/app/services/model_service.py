"""Holds the single loaded model for the process.

The ONNX session is built once at startup rather than per request: constructing it
costs far more than running it, and on a free host that difference is the whole
response time.
"""

import logging
import os
from pathlib import Path

from src.inference.predictor import LeafPredictor

logger = logging.getLogger(__name__)

# Resolve to <repo>/models/export unless the deployment overrides it.
DEFAULT_DIR = Path(__file__).resolve().parents[3] / "models" / "export"

_predictor = None
_load_error = None


def load_predictor():
    """Called on startup. Failures are recorded, not raised, so /health can report
    a degraded service instead of the container dying in a restart loop."""
    global _predictor, _load_error
    model_dir = os.getenv("MODEL_DIR", str(DEFAULT_DIR))
    try:
        _predictor = LeafPredictor(model_dir)
        _load_error = None
        logger.info("loaded %s model from %s", _predictor.arch, model_dir)
    except Exception as exc:  # noqa: BLE001 - startup must not crash the worker
        _predictor = None
        _load_error = f"{type(exc).__name__}: {exc}"
        logger.error("could not load model from %s: %s", model_dir, _load_error)
    return _predictor


def get_predictor():
    return _predictor


def get_load_error():
    return _load_error
