"""Measure the deployment cost of each trained architecture.

The report says the architecture was chosen on deployment cost rather than accuracy,
because every candidate saturates validation. This produces the numbers that claim
rests on: exported size and single-image CPU latency under the same one-thread
setting the API uses.

Exports go to a scratch directory so the live models/export bundle is untouched.

    python -m src.report.arch_costs
"""

import json
import shutil
import tempfile
from pathlib import Path

from src import config
from src.inference.export import export


def main():
    runs = []
    for run_dir in sorted(p for p in config.MODELS_DIR.iterdir() if p.is_dir()):
        cfg_path = run_dir / "train_config.json"
        if cfg_path.exists():
            runs.append((run_dir.name, json.load(open(cfg_path, encoding="utf-8"))))

    results = {}
    scratch = Path(tempfile.mkdtemp(prefix="arch_costs_"))
    try:
        for name, cfg in runs:
            # One run per architecture is enough; the two MobileNet runs differ only
            # in how their data was split, which cannot change the graph.
            arch = cfg["arch"]
            if arch in results:
                continue
            _, info = export(name, out_dir=scratch / name)
            params = _count_parameters(arch)
            results[arch] = {
                "run": name,
                "params_millions": round(params / 1e6, 2),
                "onnx_mb": info["onnx_size_mb"],
                "cpu_ms": info["single_image_cpu_ms"],
                "train_minutes": round(cfg["total_train_seconds"] / 60, 1),
            }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    out = config.METRICS_DIR / "architecture_costs.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"\n{'architecture':<18}{'params':>10}{'ONNX MB':>10}{'CPU ms':>9}{'train min':>11}")
    for arch, r in results.items():
        print(f"{arch:<18}{r['params_millions']:>9.2f}M{r['onnx_mb']:>10.1f}"
              f"{r['cpu_ms']:>9.1f}{r['train_minutes']:>11.1f}")
    print(f"\nwrote {out}")


def _count_parameters(arch):
    from src.training.models import create_model

    model = create_model(arch, len(config.BINARY_CLASSES), pretrained=False)
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    main()
