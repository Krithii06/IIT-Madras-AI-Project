"""GET /api/health - liveness probe, mirroring the FastAPI route.

Always answers 200 so the caller can tell "running but the model is missing" apart
from "not running at all".
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
            send_json(self, 200, {"status": "degraded", "model_loaded": False,
                                  "detail": "model unavailable"})
            return
        send_json(self, 200, {"status": "ok", "model_loaded": True, "detail": None})
