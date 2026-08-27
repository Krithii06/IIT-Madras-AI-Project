"""Copy the inference code and model bundle into the Vercel function directory.

Vercel builds this project with frontend/ as its root, so a serverless function in
frontend/api/ cannot read src/ or models/ above it. The pieces it needs are copied
in rather than reimplemented - a second hand-written copy of the preprocessing is
exactly the divergence tests/test_inference.py exists to prevent.

tests/test_vercel_bundle.py asserts the copies are byte-identical to their sources,
so this cannot silently go stale.

    python -m src.inference.sync_vercel
"""

import shutil

from src import config

API_DIR = config.PROJECT_ROOT / "frontend" / "api"
VENDORED_PREDICTOR = API_DIR / "_predictor.py"
VENDORED_MODEL = API_DIR / "model"

SOURCE_PREDICTOR = config.PROJECT_ROOT / "src" / "inference" / "predictor.py"
SOURCE_MODEL = config.MODELS_DIR / "export"

BUNDLE_FILES = ("model.onnx", "class_mapping.json", "preprocess.json")


def main():
    if not SOURCE_MODEL.exists():
        raise SystemExit("no exported model; run src.inference.export first")

    API_DIR.mkdir(parents=True, exist_ok=True)
    VENDORED_MODEL.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(SOURCE_PREDICTOR, VENDORED_PREDICTOR)
    print(f"  {VENDORED_PREDICTOR.relative_to(config.PROJECT_ROOT)}")

    total = 0
    for name in BUNDLE_FILES:
        source = SOURCE_MODEL / name
        if not source.exists():
            raise SystemExit(f"missing from the export bundle: {name}")
        target = VENDORED_MODEL / name
        shutil.copyfile(source, target)
        size = target.stat().st_size / 1e6
        total += size
        print(f"  {target.relative_to(config.PROJECT_ROOT)}  ({size:.2f} MB)")

    print(f"\nsynced {total:.2f} MB into the Vercel function directory")


if __name__ == "__main__":
    main()
