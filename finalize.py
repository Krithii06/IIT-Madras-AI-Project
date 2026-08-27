"""Run everything that happens after training: evaluate, export, figures, report.

Kept as one script because these steps have to happen in this order and because
re-running them by hand after a retrain is how numbers in a write-up go stale.

    python finalize.py
    python finalize.py --best mobilenet_leaf --app-url https://... --api-url https://...
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(args, label):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}", flush=True)
    result = subprocess.run([sys.executable, "-u", "-m"] + args, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f"step failed: {label}")


def trained_runs():
    models = ROOT / "models"
    if not models.exists():
        return []
    return sorted(d.name for d in models.iterdir()
                  if d.is_dir() and (d / "train_config.json").exists())


def split_strategy_of(run_name):
    cfg = json.load(open(ROOT / "models" / run_name / "train_config.json", encoding="utf-8"))
    return cfg.get("split", "leaf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best", default="mobilenet_leaf",
                        help="run whose weights get exported for serving")
    parser.add_argument("--app-url", default=None)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--skip-screenshots", action="store_true",
                        help="screenshots need the backend and frontend running locally")
    args = parser.parse_args()

    runs = trained_runs()
    if not runs:
        sys.exit("no completed training runs found; run run_experiments.py first")
    print(f"completed runs: {', '.join(runs)}")

    for name in runs:
        strategy = split_strategy_of(name)
        for split in ("val", "test"):
            run(["src.evaluation.evaluate", "--run-name", name,
                 "--split", split, "--split-strategy", strategy],
                f"evaluate {name} ({split})")

    run(["src.evaluation.leakage"], "leakage audit")
    run(["src.evaluation.error_analysis", "--run-name", args.best, "--split", "test"],
        f"error analysis ({args.best})")

    run(["src.evaluation.figures", "--dataset"], "dataset figures")
    run(["src.evaluation.figures", "--training", "--runs"] + runs, "training curves")

    run(["src.inference.export", "--run-name", args.best], f"export {args.best} to ONNX")

    if not args.skip_screenshots:
        try:
            run(["src.report.screenshots"], "application screenshots")
        except SystemExit:
            print("screenshots skipped - start the backend and frontend first")

    report_args = ["src.report.build_report"]
    if args.app_url:
        report_args += ["--app-url", args.app_url]
    if args.api_url:
        report_args += ["--api-url", args.api_url]
    run(report_args, "build PDF report")

    run(["src.report.latex_assets"], "copy figures into the LaTeX project")
    # Fails loudly if a retrain moved a number the report still quotes.
    run(["src.report.check_latex"], "validate the LaTeX report")

    run(["src.evaluation.summarize"], "summary")
    print("\nfinalize complete")


if __name__ == "__main__":
    main()
