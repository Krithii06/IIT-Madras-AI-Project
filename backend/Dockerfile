# Build context is the repository root, not backend/, because the image needs
# src/inference and models/export alongside the API:
#   docker build -f backend/Dockerfile -t plant-disease-api .
FROM python:3.11-slim

# The layout under /app mirrors the repository so the relative paths that locate
# models/export resolve identically in the container and in a local checkout.
WORKDIR /app/backend

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app /app/backend/app
COPY src/__init__.py /app/src/__init__.py
COPY src/inference /app/src/inference
COPY models/export /app/models/export

# Run as a non-root user. Render does not require this, but it is the right
# default and keeps the image portable to hosts that do.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860

# Single worker on purpose: each worker would load its own copy of the ONNX
# session, and Render's free instance has 512 MB of RAM to fit everything in.
#
# The port is read from the environment because hosts disagree about it - Render
# injects PORT, Hugging Face Spaces expects 7860 - and hard-coding either one
# makes the image non-portable.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
