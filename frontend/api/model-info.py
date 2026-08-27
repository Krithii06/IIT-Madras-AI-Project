"""GET /api/model-info - what the deployed model is, mirroring the FastAPI route.

The front end reads this on load rather than hard-coding the architecture or the
threshold, so the panel cannot drift from the model actually being served.
"""

import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _service import get_predictor, send_json  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        predictor = get_predictor()
        if predictor is None:
            send_json(self, 503, {"detail": "Model is not loaded. Try again shortly."})
            return

        send_json(self, 200, {
            "architecture": predictor.arch,
            "classes": predictor.classes,
            "num_classes": len(predictor.classes),
            "input_size": predictor.image_size,
            "confidence_threshold": predictor.confidence_threshold,
            "source_classes": predictor.source_classes,
            "dataset": "PlantVillage (Apple subset, color)",
        })
