"""Run the four training runs this project reports on, one after another.

Sequential on purpose: the machine has four physical cores and running two jobs
at once made both slower in testing.

    python run_experiments.py            # all four
    python run_experiments.py --only mobilenet_leaf
"""

import argparse
import subprocess
import sys
import time

# name -> (architecture, split strategy). mobilenet_leaf and mobilenet_random are
# the two halves of the leakage comparison; the other two are the architecture
# candidates the brief names.
RUNS = {
    "mobilenet_leaf": ("mobilenet_v2", "leaf"),
    "mobilenet_random": ("mobilenet_v2", "random"),
    "efficientnet_leaf": ("efficientnet_b0", "leaf"),
    "resnet18_leaf": ("resnet18", "leaf"),
}

EPOCHS = 6
WARMUP = 2
PATIENCE = 3

# Two loader workers, not four. Benchmarked on this box the loader needs ~13s per
# epoch at 160px while the model step needs ~127s, so two workers keep the queue
# full and leave the physical cores to the forward/backward pass instead of
# fighting it for them.
WORKERS = 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=list(RUNS))
    args = parser.parse_args()

    selected = args.only or list(RUNS)
    started = time.time()

    for name in selected:
        arch, split = RUNS[name]
        print(f"\n{'=' * 70}\n{name}  ({arch}, {split} split)\n{'=' * 70}", flush=True)
        cmd = [
            sys.executable, "-u", "-m", "src.training.train",
            "--arch", arch,
            "--run-name", name,
            "--split", split,
            "--epochs", str(EPOCHS),
            "--warmup-epochs", str(WARMUP),
            "--patience", str(PATIENCE),
            "--workers", str(WORKERS),
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(f"run {name} failed with exit code {result.returncode}")

    print(f"\nall runs finished in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
