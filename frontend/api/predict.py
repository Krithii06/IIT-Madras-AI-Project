"""POST /api/predict - classify an uploaded leaf image.

Mirrors the FastAPI route in backend/app/routes/predict.py, including its status
codes, so the two deployments are interchangeable from the front end's point of view.
"""

import sys
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _service import decode_image, get_predictor, read_upload, send_json  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        predictor = get_predictor()
        if predictor is None:
            send_json(self, 503, {"detail": "Model is not loaded. Try again shortly."})
            return

        payload, error = read_upload(self)
        if error:
            send_json(self, error[0], {"detail": error[1]})
            return

        image, error = decode_image(payload)
        if error:
            send_json(self, error[0], {"detail": error[1]})
            return

        try:
            started = time.perf_counter()
            result = predictor.predict(image)
            elapsed_ms = (time.perf_counter() - started) * 1000
        except Exception:
            # Detail stays in the platform logs; the caller gets something generic.
            self.log_error("inference failed", exc_info=True)
            send_json(self, 500, {"detail": "Prediction failed."})
            return

        result["inference_ms"] = round(elapsed_ms, 1)
        send_json(self, 200, result)

    def do_GET(self):
        send_json(self, 405, {"detail": "Use POST with a multipart 'file' field."})
