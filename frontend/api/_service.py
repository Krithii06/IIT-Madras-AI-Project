"""Shared pieces for the Vercel serverless functions.

The three handlers next to this file mirror the FastAPI routes in backend/app: the
same validation, the same status codes and the same JSON shape, so the front end
does not care which of the two is serving it.

The predictor is built at import time. Vercel reuses a warm function instance across
requests, so the ONNX session is created once per instance rather than per request -
the same reasoning as the backend's startup hook.
"""

import json
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "model"

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_PIXELS = 40_000_000

_predictor = None
_load_error = None


def get_predictor():
    """Build the predictor once per warm instance; record failures rather than raise."""
    global _predictor, _load_error
    if _predictor is None and _load_error is None:
        try:
            from _predictor import LeafPredictor

            _predictor = LeafPredictor(MODEL_DIR)
        except Exception as exc:  # noqa: BLE001 - a dead import must not 500 /health
            _load_error = f"{type(exc).__name__}: {exc}"
    return _predictor


def send_json(handler, status, payload):
    body = json.dumps(payload).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_upload(handler):
    """Pull the file part out of a multipart body.

    Returns (payload, error). The stdlib cgi module that used to do this was removed
    in Python 3.13, so the body is reassembled as a MIME message instead, which is
    what cgi did underneath anyway.
    """
    content_type = handler.headers.get("Content-Type", "")
    length = int(handler.headers.get("Content-Length") or 0)

    if not content_type.startswith("multipart/form-data"):
        return None, (422, "Send the image as multipart/form-data in a 'file' field.")
    if length <= 0:
        return None, (400, "The uploaded file is empty.")
    if length > MAX_UPLOAD_BYTES + 8192:  # a little slack for the multipart envelope
        return None, (413, f"File is larger than the "
                           f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")

    raw = handler.rfile.read(length)
    envelope = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
    message = BytesParser(policy=default).parsebytes(envelope + raw)

    part = None
    for candidate in message.iter_parts():
        disposition = candidate.get("Content-Disposition", "")
        if "filename" in disposition or 'name="file"' in disposition:
            part = candidate
            break
    if part is None:
        return None, (422, "No 'file' field in the request.")

    declared = (part.get_content_type() or "").lower()
    if declared not in ALLOWED_CONTENT_TYPES:
        return None, (415, f"Unsupported file type. Send one of: "
                           f"{', '.join(sorted(ALLOWED_CONTENT_TYPES))}.")

    payload = part.get_payload(decode=True)
    if not payload:
        return None, (400, "The uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        return None, (413, f"File is larger than the "
                           f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
    return payload, None


def decode_image(payload):
    """Decode to a PIL image, guarding against a decompression bomb."""
    import io

    from PIL import Image, UnidentifiedImageError

    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        return None, (400, "The file could not be read as an image.")

    if image.width * image.height > MAX_PIXELS:
        return None, (413, "Image resolution is too large to process.")
    return image, None
