"""Collect every run's metrics into one comparison table.

Numbers in the README and the report are pasted from this script's output rather
than typed by hand, so there is no route by which a figure in the write-up can
drift away from what was actually measured.

    python -m src.evaluation.summarize
    python -m src.evaluation.summarize --markdown
"""

import argparse
import json

from src import config


def load_metrics(split="test"):
    if not config.METRICS_DIR.exists():
        return []
    found = []
    for path in sorted(config.METRICS_DIR.glob(f"*_{split}.json")):
        found.append(json.load(open(path, encoding="utf-8")))
    return found


def load_run_config(run_name):
    path = config.MODELS_DIR / run_name / "train_config.json"
    if not path.exists():
        return {}
    return json.load(open(path, encoding="utf-8"))


def rows_for(metrics, split):
    rows = []
    for m in metrics:
        cfg = load_run_config(m["run_name"])
        rows.append({
            "run": m["run_name"],
            "arch": m["arch"],
            "split_strategy": m["split_strategy"],
            "images": m["images"],
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "precision_diseased": m["precision_positive"],
            "recall_diseased": m["recall_positive"],
            "f1_diseased": m["f1_positive"],
            "best_epoch": cfg.get("best_epoch"),
            "train_minutes": (round(cfg["total_train_seconds"] / 60, 1)
                              if cfg.get("total_train_seconds") else None),
        })
    return rows


def print_table(rows, split):
    if not rows:
        print(f"no {split} metrics yet")
        return
    header = (f"{'run':<20}{'arch':<17}{'split':<8}{'acc':>8}{'macroF1':>9}"
              f"{'prec_d':>8}{'rec_d':>8}{'epoch':>7}{'min':>7}")
    print(f"\n=== {split} split ===")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['run']:<20}{r['arch']:<17}{r['split_strategy']:<8}"
              f"{r['accuracy']:>8.4f}{r['macro_f1']:>9.4f}"
              f"{r['precision_diseased']:>8.4f}{r['recall_diseased']:>8.4f}"
              f"{str(r['best_epoch']):>7}{str(r['train_minutes']):>7}")


def print_markdown(rows, split):
    if not rows:
        return
    print(f"\n**{split} split**\n")
    print("| Run | Architecture | Split | Accuracy | Macro F1 | Precision (diseased) "
          "| Recall (diseased) | Best epoch | Train (min) |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| `{r['run']}` | {r['arch']} | {r['split_strategy']} | {r['accuracy']:.4f} "
              f"| {r['macro_f1']:.4f} | {r['precision_diseased']:.4f} "
              f"| {r['recall_diseased']:.4f} | {r['best_epoch']} | {r['train_minutes']} |")


def print_per_source(metrics):
    """Binary model broken down by the original disease, where confusions live."""
    for m in metrics:
        per_source = m.get("per_source_class")
        if not per_source:
            continue
        print(f"\n--- {m['run_name']} ({m['split']}): per source class ---")
        print(f"{'source class':<32}{'n':>6}{'correct':>9}{'recall':>9}{'mean conf':>11}")
        for name, stats in sorted(per_source.items()):
            print(f"{name:<32}{stats['images']:>6}{stats['correct']:>9}"
                  f"{stats['recall']:>9.4f}{stats['mean_confidence']:>11.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--splits", nargs="*", default=["val", "test"])
    args = parser.parse_args()

    for split in args.splits:
        metrics = load_metrics(split)
        rows = rows_for(metrics, split)
        if args.markdown:
            print_markdown(rows, split)
        else:
            print_table(rows, split)
            if split == "test":
                print_per_source(metrics)

    leakage_path = config.METRICS_DIR / "leakage_analysis.json"
    if leakage_path.exists() and not args.markdown:
        leak = json.load(open(leakage_path, encoding="utf-8"))
        grouped = leak["leaf_grouped_split"]["val_vs_train"]["leaked_fraction"]
        naive = leak["random_image_split"]["val_vs_train"]["leaked_fraction"]
        print(f"\n=== split integrity ===")
        print(f"validation images sharing a leaf with train:")
        print(f"  leaf-grouped split : {grouped * 100:.1f}%")
        print(f"  naive random split : {naive * 100:.1f}%")


if __name__ == "__main__":
    main()
