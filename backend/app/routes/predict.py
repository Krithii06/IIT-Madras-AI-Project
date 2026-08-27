"""Prediction, health and model-info endpoints."""

import io
import logging
import time

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.schemas.prediction import HealthResponse, ModelInfoResponse, PredictionResponse
from app.services import model_service

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}

# A leaf photo bigger than this is decoded but downscaled first; it guards against
# a decompression bomb turning one upload into gigabytes of pixels.
MAX_PIXELS = 40_000_000


@router.get("/health", response_model=HealthResponse)
def health():
    """Liveness probe for the host. Always 200 so a restarting platform can tell
    the difference between 'process up, model missing' and 'process down'."""
    predictor = model_service.get_predictor()
    if predictor is None:
        return HealthResponse(status="degraded", model_loaded=False,
                              detail="model unavailable")
    return HealthResponse(status="ok", model_loaded=True)


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    predictor = _require_predictor()
    return ModelInfoResponse(
        architecture=predictor.arch,
        classes=predictor.classes,
        num_classes=len(predictor.classes),
        input_size=predictor.image_size,
        confidence_threshold=predictor.confidence_threshold,
        source_classes=predictor.source_classes,
        dataset="PlantVillage (Apple subset, color)",
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    predictor = _require_predictor()

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Send one of: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}.",
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )

    try:
        image = Image.open(io.BytesIO(payload))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400,
                            detail="The file could not be read as an image.")

    if image.width * image.height > MAX_PIXELS:
        raise HTTPException(status_code=413,
                            detail="Image resolution is too large to process.")

    try:
        started = time.perf_counter()
        result = predictor.predict(image)
        elapsed_ms = (time.perf_counter() - started) * 1000
    except Exception:
        # Log the detail for us, return something generic to the caller.
        logger.exception("inference failed")
        raise HTTPException(status_code=500, detail="Prediction failed.")

    return PredictionResponse(inference_ms=round(elapsed_ms, 1), **result)


def _require_predictor():
    predictor = model_service.get_predictor()
    if predictor is None:
        raise HTTPException(status_code=503,
                            detail="Model is not loaded. Try again shortly.")
    return predictor
