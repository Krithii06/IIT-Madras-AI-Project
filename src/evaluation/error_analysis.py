"""Error analysis for a trained run.

Two things this answers that the headline metrics do not:

1. Errors on this dataset are not independent. Because several photographs share a
   physical leaf, one badly-handled leaf produces several "separate" mistakes. The
   leaf-level score below is the honest denominator.
2. Whether the confidence threshold would have caught the mistakes. If the model is
   confidently wrong, exposing confidence in the UI does not protect the user.

    python -m src.evaluation.error_analysis --run-name mobilenet_leaf
"""

import argparse
import collections
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src import config
from src.data.dataset import leaf_grouped_split, load_manifest
from src.evaluation.evaluate import collect_predictions, load_run


def analyse(run_name, split_name="test", max_panels=8):
    model, train_cfg, mapping, _ = load_run(run_name)
    class_names = mapping["classes"]

    rows = load_manifest()
    train_rows, val_rows, test_rows = leaf_grouped_split(rows, seed=train_cfg["seed"])
    subset = {"train": train_rows, "val": val_rows, "test": test_rows}[split_name]

    probs, targets = collect_predictions(model, subset)
    preds = probs.argmax(1)
    confidences = probs.max(1)

    wrong = [i for i in range(len(subset)) if preds[i] != targets[i]]

    # Leaf-level view: a leaf counts as missed if any of its photographs is wrong.
    by_leaf = collections.defaultdict(list)
    for i, row in enumerate(subset):
        by_leaf[row["leaf_id"]].append(i)
    missed_leaves = {lid for lid, idxs in by_leaf.items()
                     if any(preds[i] != targets[i] for i in idxs)}

    threshold = 0.70
    caught = [i for i in wrong if confidences[i] < threshold]

    error_leaves = collections.Counter(subset[i]["leaf_id"] for i in wrong)
    error_sources = collections.Counter(subset[i]["source_class"] for i in wrong)

    report = {
        "run_name": run_name,
        "split": split_name,
        "images": len(subset),
        "leaves": len(by_leaf),
        "image_errors": len(wrong),
        "image_accuracy": round(1 - len(wrong) / len(subset), 4),
        "leaves_with_at_least_one_error": len(missed_leaves),
        "leaf_level_accuracy": round(1 - len(missed_leaves) / len(by_leaf), 4),
        "distinct_leaves_among_errors": len(error_leaves),
        "errors_per_error_leaf": dict(error_leaves),
        "errors_by_source_class": dict(error_sources),
        "confidence_threshold": threshold,
        "errors_below_threshold": len(caught),
        "errors_above_threshold": len(wrong) - len(caught),
        "mean_confidence_when_wrong": (round(float(confidences[wrong].mean()), 4)
                                       if wrong else None),
        "max_confidence_when_wrong": (round(float(confidences[wrong].max()), 4)
                                      if wrong else None),
    }

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = config.METRICS_DIR / f"{run_name}_{split_name}_error_analysis.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    if wrong:
        _plot_errors(subset, wrong, preds, confidences, class_names,
                     config.FIGURES_DIR / f"{run_name}_{split_name}_errors.png",
                     max_panels)

    print(f"{report['image_errors']} wrong of {report['images']} images "
          f"(accuracy {report['image_accuracy']:.4f})")
    print(f"{report['leaves_with_at_least_one_error']} affected of {report['leaves']} "
          f"leaves (leaf-level accuracy {report['leaf_level_accuracy']:.4f})")
    print(f"errors come from {report['distinct_leaves_among_errors']} distinct leaf/leaves: "
          f"{report['errors_per_error_leaf']}")
    print(f"by source class: {report['errors_by_source_class']}")
    print(f"confidence when wrong: mean {report['mean_confidence_when_wrong']}, "
          f"max {report['max_confidence_when_wrong']}")
    print(f"a {threshold} threshold would have flagged "
          f"{report['errors_below_threshold']} of {report['image_errors']}")
    return report


def _plot_errors(subset, wrong, preds, confidences, class_names, out_path, max_panels):
    picks = sorted(wrong, key=lambda i: -confidences[i])[:max_panels]
    cols = min(4, len(picks))
    rowsn = (len(picks) + cols - 1) // cols

    fig, axes = plt.subplots(rowsn, cols, figsize=(cols * 2.3, rowsn * 2.7), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for panel, i in enumerate(picks):
        ax = axes[panel // cols][panel % cols]
        row = subset[i]
        with Image.open(config.RAW_DIR / row["rel_path"]) as img:
            ax.imshow(img.convert("RGB"))
        source = row["source_class"].replace("Apple___", "").replace("_", " ")
        leaf = row["leaf_id"].split(":::")[-1]
        ax.set_title(f"{source}\npredicted {class_names[preds[i]]} "
                     f"({confidences[i]:.2f})\nleaf {leaf}", fontsize=7)

    fig.suptitle("Misclassified test images, most confident first", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()
    analyse(args.run_name, args.split)
