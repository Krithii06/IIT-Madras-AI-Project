"""Evaluate a trained run and write metrics, a confusion matrix and an error list.

    python -m src.evaluation.evaluate --run-name mobilenet_leaf --split test

The test split is the official PlantVillage held-out set and is only ever touched
by this script, after model selection has already finished on validation.
"""

import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from torch.utils.data import DataLoader

from src import config
from src.data.dataset import (PlantLeafDataset, leaf_grouped_split, load_manifest,
                              random_image_split)
from src.preprocessing.transforms import eval_transform
from src.training.models import create_model


def load_run(run_name):
    run_dir = config.MODELS_DIR / run_name
    train_cfg = json.load(open(run_dir / "train_config.json", encoding="utf-8"))
    mapping = json.load(open(run_dir / "class_mapping.json", encoding="utf-8"))

    model = create_model(train_cfg["arch"], len(mapping["classes"]), pretrained=False)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location="cpu"))
    model.eval()
    return model, train_cfg, mapping, run_dir


def collect_predictions(model, rows, batch_size=64, workers=0):
    dataset = PlantLeafDataset(rows, eval_transform())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    probabilities = []
    with torch.no_grad():
        for images, _ in loader:
            probabilities.append(torch.softmax(model(images), dim=1))
    probs = torch.cat(probabilities).numpy()
    targets = np.array([dataset.class_to_idx[r["binary_label"]] for r in rows])
    return probs, targets


def plot_confusion(matrix, class_names, title, out_path):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(range(len(class_names)), class_names)
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title, fontsize=10)

    threshold = matrix.max() / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def per_source_class_breakdown(rows, probs, targets, class_names):
    """How the binary model performs on each original PlantVillage class.

    The deployed model only says healthy or diseased, but the brief also asks about
    the source classes, and this is where the confusions actually show up.
    """
    preds = probs.argmax(1)
    out = {}
    for source in sorted({r["source_class"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["source_class"] == source]
        if not idx:
            continue
        correct = int((preds[idx] == targets[idx]).sum())
        confidences = probs[idx].max(1)
        out[source] = {
            "images": len(idx),
            "correct": correct,
            "recall": round(correct / len(idx), 4),
            "mean_confidence": round(float(confidences.mean()), 4),
            "expected_binary_label": config.to_binary_label(source),
        }
    return out


def evaluate(run_name, split_name, split_strategy):
    model, train_cfg, mapping, run_dir = load_run(run_name)
    class_names = mapping["classes"]

    rows = load_manifest()
    splitter = leaf_grouped_split if split_strategy == "leaf" else random_image_split
    train_rows, val_rows, test_rows = splitter(rows, seed=train_cfg["seed"])
    subset = {"train": train_rows, "val": val_rows, "test": test_rows}[split_name]

    probs, targets = collect_predictions(model, subset)
    preds = probs.argmax(1)

    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, labels=range(len(class_names)), zero_division=0
    )
    positive_idx = class_names.index(mapping["positive_class"])
    matrix = confusion_matrix(targets, preds, labels=range(len(class_names)))

    confidences = probs.max(1)
    correct_mask = preds == targets

    metrics = {
        "run_name": run_name,
        "arch": train_cfg["arch"],
        "split": split_name,
        "split_strategy": split_strategy,
        "images": len(subset),
        "leaves": len({r["leaf_id"] for r in subset}),
        "accuracy": round(float(accuracy_score(targets, preds)), 4),
        "macro_f1": round(float(f1_score(targets, preds, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(targets, preds, average="weighted", zero_division=0)), 4),
        "positive_class": mapping["positive_class"],
        "precision_positive": round(float(precision[positive_idx]), 4),
        "recall_positive": round(float(recall[positive_idx]), 4),
        "f1_positive": round(float(f1[positive_idx]), 4),
        "per_class": {
            name: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": class_names,
        "confidence": {
            "mean_when_correct": round(float(confidences[correct_mask].mean()), 4),
            "mean_when_wrong": (round(float(confidences[~correct_mask].mean()), 4)
                                if (~correct_mask).any() else None),
            "share_below_0.70": round(float((confidences < 0.70).mean()), 4),
        },
        "per_source_class": per_source_class_breakdown(subset, probs, targets, class_names),
    }

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    stem = f"{run_name}_{split_name}"
    with open(config.METRICS_DIR / f"{stem}.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    plot_confusion(matrix, class_names,
                   f"{train_cfg['arch']} - {split_name} split (n={len(subset)})",
                   config.FIGURES_DIR / f"{stem}_confusion_matrix.png")

    # Persist every mistake so the error analysis works from real examples.
    errors = []
    for i, row in enumerate(subset):
        if preds[i] != targets[i]:
            errors.append({
                "rel_path": row["rel_path"],
                "source_class": row["source_class"],
                "true_label": class_names[targets[i]],
                "predicted_label": class_names[preds[i]],
                "confidence": round(float(confidences[i]), 4),
                "leaf_id": row["leaf_id"],
            })
    errors.sort(key=lambda e: -e["confidence"])
    if errors:
        with open(config.METRICS_DIR / f"{stem}_errors.csv", "w", newline="",
                  encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(errors[0].keys()))
            writer.writeheader()
            writer.writerows(errors)

    print(classification_report(targets, preds, target_names=class_names, digits=4,
                                zero_division=0))
    print(f"accuracy {metrics['accuracy']:.4f} | macro-F1 {metrics['macro_f1']:.4f} | "
          f"weighted-F1 {metrics['weighted_f1']:.4f}")
    print(f"confusion matrix (rows=actual {class_names}):\n{matrix}")
    print(f"{len(errors)} misclassified of {len(subset)}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--split-strategy", default="leaf", choices=["leaf", "random"])
    args = parser.parse_args()
    evaluate(args.run_name, args.split, args.split_strategy)
