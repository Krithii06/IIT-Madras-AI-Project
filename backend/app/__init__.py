"""FastAPI application package.

The service imports src.inference.predictor from the repository root. In the
container that works because the Dockerfile sets PYTHONPATH=/app; running
`uvicorn app.main:app` from backend/ during development has no such setting, so
the root is added here instead. Without it the documented local run command fails
with ModuleNotFoundError, which is a poor first experience for anyone cloning this.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
