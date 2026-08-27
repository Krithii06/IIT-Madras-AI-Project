"""Response models. Keeping them explicit means the frontend contract is visible
in one place and FastAPI can document it without extra annotation."""

from typing import List, Optional

from pydantic import BaseModel, Field


class ClassScore(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    predicted_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    low_confidence: bool
    confidence_threshold: float
    top_predictions: List[ClassScore]
    message: str
    inference_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    detail: Optional[str] = None


class ModelInfoResponse(BaseModel):
    architecture: str
    classes: List[str]
    num_classes: int
    input_size: int
    confidence_threshold: float
    source_classes: List[str]
    dataset: str
