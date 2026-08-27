"""FastAPI application for the plant disease classifier.

Run locally:
    uvicorn app.main:app --reload --port 8000     (from the backend/ directory)
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.predict import router
from app.services import model_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Comma-separated list of allowed frontend origins. Defaults to local dev only;
# the deployed frontend URL is supplied through the environment, never committed.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if o.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load_predictor()
    yield


app = FastAPI(
    title="Plant Disease Classification API",
    description="Binary healthy/diseased classification for apple leaf images.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "service": "plant-disease-classification",
        "docs": "/docs",
        "endpoints": ["/health", "/model-info", "/predict"],
    }
