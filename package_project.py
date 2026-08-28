"""Package the project into a single zip that runs without downloading the dataset.

The full PlantVillage archive is 2.2 GB and is deliberately not committed. That is
right for a repository and wrong for a zip someone is meant to open and try, so this
bundles everything git tracks plus a small stratified sample of real leaf images -
enough to start the app, classify something, and see the pipeline work.

Sample images are drawn from the held-out test split, so nothing in the zip was
trained on.

    python package_project.py
    python package_project.py --per-class 25 --out ../submission.zip
"""

import argparse
import csv
import random
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = 42

SAMPLE_README = """\
# Sample images

A small slice of PlantVillage for trying the app without downloading the full 2.2 GB
archive. {per_class} images per class, drawn from the **held-out test split** with a
fixed seed, so none of them were trained on.

```
{tree}
```

The full dataset and how to rebuild it are described in ../README.md. Everything else
in this package is complete: the trained model, the API, the web app and the tests.

## Quickest way to see it work

```bash
pip install -r requirements-dev.txt
cd backend && uvicorn app.main:app --port 8000     # terminal 1
cd frontend && npm install && npm run dev          # terminal 2
```

Then open http://localhost:5173 and upload one of these images.

Or skip the install entirely and use the deployed app:
https://plant-disease-classification-rosy.vercel.app
"""


def tracked_files():
    result = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit("not a git repository, or git is unavailable")
    return [line for line in result.stdout.splitlines() if line.strip()]


def pick_samples(per_class):
    """Stratified sample from the test split, deterministic for a fixed seed."""
    manifest = ROOT / "data" / "manifest.csv"
    if not manifest.exists():
        print("  no manifest.csv - packaging source only, without sample images")
        return {}

    with open(manifest, encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["official_split"] == "test"]

    by_class = {}
    for row in rows:
        by_class.setdefault(row["source_class"], []).append(row)

    rng = random.Random(SEED)
    chosen = {}
    for source_class, members in sorted(by_class.items()):
        # Sort first so the sample does not depend on manifest row order.
        members.sort(key=lambda r: r["rel_path"])
        chosen[source_class] = rng.sample(members, min(per_class, len(members)))
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=15)
    parser.add_argument("--out", default=None)
    parser.add_argument("--slim", action="store_true",
                        help="drop frontend/api/model/, a second copy of the same 8.7 MB "
                             "ONNX file that only the Vercel build needs, to fit under "
                             "a 25 MB mail attachment limit")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else ROOT.parent / "plant-disease-classification.zip"
    out_path = out_path.resolve()

    files = tracked_files()
    if args.slim:
        before = len(files)
        files = [f for f in files if not f.startswith("frontend/api/model/")]
        print(f"  slim: dropped {before - len(files)} vendored model file(s); "
              f"models/export/ is still included")
    samples = pick_samples(args.per_class)
    total_samples = sum(len(v) for v in samples.values())

    tree_lines = [f"{cls}/  ({len(rows)} images)" for cls, rows in sorted(samples.items())]
    readme = SAMPLE_README.format(per_class=args.per_class, tree="\n".join(tree_lines))

    print(f"packaging {len(files)} tracked files + {total_samples} sample images")

    if out_path.exists():
        out_path.unlink()

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in files:
            source = ROOT / rel
            if source.exists():
                zf.write(source, f"plant-disease-classification/{rel}")

        for source_class, rows in samples.items():
            for row in rows:
                source = ROOT / "data" / "raw" / row["rel_path"]
                if source.exists():
                    name = Path(row["rel_path"]).name
                    zf.write(source,
                             f"plant-disease-classification/data/sample/{source_class}/{name}")

        if samples:
            zf.writestr("plant-disease-classification/data/sample/README.md", readme)

        if args.slim:
            zf.writestr(
                "plant-disease-classification/frontend/api/model/README.md",
                "The exported model is not in this archive copy, to keep it under a mail\n"
                "attachment limit. It is a byte-identical copy of models/export/, which is\n"
                "included. Restore it with:\n\n"
                "    python -m src.inference.sync_vercel\n\n"
                "It is only needed to deploy to Vercel; running the app locally uses\n"
                "models/export/ directly. The full tree is on GitHub.\n")

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nwrote {out_path}")
    print(f"  {size_mb:.1f} MB")
    for source_class, rows in sorted(samples.items()):
        print(f"  sample: {source_class:<32} {len(rows)}")


if __name__ == "__main__":
    main()
