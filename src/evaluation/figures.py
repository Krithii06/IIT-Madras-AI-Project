"""Figures for the report.

Each function answers one question, so the report does not end up padded with
charts nobody reads. Image resolution is intentionally not plotted: every file in
the subset is 256x256 RGB, which is a sentence, not a histogram.

    python -m src.evaluation.figures --dataset
    python -m src.evaluation.figures --training
"""

import argparse
import json
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src import config
from src.data.dataset import leaf_grouped_split, load_manifest
from src.preprocessing.transforms import train_transform

SHORT_NAMES = {
    "Apple___Apple_scab": "Apple scab",
    "Apple___Black_rot": "Black rot",
    "Apple___Cedar_apple_rust": "Cedar apple rust",
    "Apple___healthy": "Healthy",
}


def _short(name):
    return SHORT_NAMES.get(name, name.replace("Apple___", "").replace("_", " "))


def class_distribution(rows, out_path):
    """Shows both the source classes and the binary target they collapse into."""
    counts = Counter(r["source_class"] for r in rows)
    binary = Counter(r["binary_label"] for r in rows)

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8),
                                      gridspec_kw={"width_ratios": [2.2, 1]})

    names = [_short(c) for c in config.SOURCE_CLASSES]
    values = [counts[c] for c in config.SOURCE_CLASSES]
    colours = ["#a8442a" if c != "Apple___healthy" else "#2f6f43"
               for c in config.SOURCE_CLASSES]
    bars = left.bar(names, values, color=colours)
    left.set_title("Source classes", fontsize=10)
    left.set_ylabel("images")
    left.tick_params(axis="x", labelsize=8, rotation=12)
    for bar, value in zip(bars, values):
        left.text(bar.get_x() + bar.get_width() / 2, value + 20, str(value),
                  ha="center", fontsize=8)

    bnames = config.BINARY_CLASSES
    bvalues = [binary[n] for n in bnames]
    bars = right.bar(bnames, bvalues, color=["#2f6f43", "#a8442a"])
    right.set_title("Binary target", fontsize=10)
    right.tick_params(axis="x", labelsize=9)
    for bar, value in zip(bars, bvalues):
        right.text(bar.get_x() + bar.get_width() / 2, value + 20, str(value),
                   ha="center", fontsize=8)

    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Apple subset of PlantVillage (3,171 images)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def sample_grid(rows, out_path, per_class=4, seed=config.SEED):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(config.SOURCE_CLASSES), per_class,
                             figsize=(per_class * 1.7, len(config.SOURCE_CLASSES) * 1.85))

    for row_idx, source in enumerate(config.SOURCE_CLASSES):
        pool = [r for r in rows if r["source_class"] == source]
        picks = rng.choice(len(pool), size=min(per_class, len(pool)), replace=False)
        for col_idx in range(per_class):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if col_idx < len(picks):
                with Image.open(config.RAW_DIR / pool[picks[col_idx]]["rel_path"]) as img:
                    ax.imshow(img.convert("RGB"))
            if col_idx == 0:
                ax.set_title(_short(source), fontsize=8, loc="left")

    fig.suptitle("Representative images per class", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def leaf_group_figure(rows, out_path, n_leaves=3, per_leaf=5):
    """The reason the split is grouped: these are all one physical leaf.

    Without this grouping a random split scatters these across train and test and
    the reported accuracy stops meaning anything.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[row["leaf_id"]].append(row)
    biggest = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:n_leaves]

    fig, axes = plt.subplots(n_leaves, per_leaf, figsize=(per_leaf * 1.7, n_leaves * 1.9))
    for row_idx, (leaf_id, members) in enumerate(biggest):
        for col_idx in range(per_leaf):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if col_idx < len(members):
                with Image.open(config.RAW_DIR / members[col_idx]["rel_path"]) as img:
                    ax.imshow(img.convert("RGB"))
            if col_idx == 0:
                label = leaf_id.split(":::")[0].replace("Apple___", "")
                ax.set_title(f"leaf {leaf_id.split(':::')[-1]} ({label}, "
                             f"{len(members)} photos)", fontsize=7.5, loc="left")

    fig.suptitle("Each row is one physical leaf photographed repeatedly", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def augmentation_figure(rows, out_path, n=6, seed=config.SEED):
    rng = np.random.default_rng(seed)
    row = rows[rng.integers(len(rows))]
    with Image.open(config.RAW_DIR / row["rel_path"]) as img:
        original = img.convert("RGB")

    transform = train_transform()
    mean = np.array(config.IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(config.IMAGENET_STD).reshape(3, 1, 1)

    fig, axes = plt.subplots(1, n + 1, figsize=((n + 1) * 1.6, 2.1))
    axes[0].imshow(original)
    axes[0].set_title("original", fontsize=8)
    axes[0].axis("off")

    import torch

    torch.manual_seed(seed)
    for i in range(n):
        tensor = transform(original).numpy()
        # Undo normalisation so the panel shows what the crop actually looks like.
        restored = np.clip(tensor * std + mean, 0, 1).transpose(1, 2, 0)
        axes[i + 1].imshow(restored)
        axes[i + 1].set_title(f"aug {i + 1}", fontsize=8)
        axes[i + 1].axis("off")

    fig.suptitle(f"Training augmentations ({_short(row['source_class'])})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def training_curves(run_names, out_path):
    fig, (loss_ax, f1_ax) = plt.subplots(1, 2, figsize=(9.5, 3.6))

    for run in run_names:
        path = config.MODELS_DIR / run / "history.json"
        if not path.exists():
            continue
        history = json.load(open(path, encoding="utf-8"))
        epochs = [h["epoch"] for h in history]
        loss_ax.plot(epochs, [h["train_loss"] for h in history], label=f"{run} train", lw=1.4)
        loss_ax.plot(epochs, [h["val_loss"] for h in history], "--", label=f"{run} val", lw=1.4)
        f1_ax.plot(epochs, [h["val_macro_f1"] for h in history], label=run, lw=1.4, marker="o",
                   ms=3)

    loss_ax.set_xlabel("epoch")
    loss_ax.set_ylabel("cross-entropy loss")
    loss_ax.set_title("Loss", fontsize=10)
    loss_ax.legend(fontsize=7)

    f1_ax.set_xlabel("epoch")
    f1_ax.set_ylabel("validation macro F1")
    f1_ax.set_title("Validation macro F1", fontsize=10)
    f1_ax.legend(fontsize=7)

    for ax in (loss_ax, f1_ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, lw=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--runs", nargs="*",
                        default=["mobilenet_leaf", "efficientnet_leaf", "resnet18_leaf"])
    args = parser.parse_args()

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_manifest()

    if args.dataset or not (args.dataset or args.training):
        class_distribution(rows, config.FIGURES_DIR / "class_distribution.png")
        sample_grid(rows, config.FIGURES_DIR / "class_samples.png")
        leaf_group_figure(rows, config.FIGURES_DIR / "leaf_groups.png")
        augmentation_figure(rows, config.FIGURES_DIR / "augmentations.png")
        print("wrote dataset figures")

    if args.training:
        training_curves(args.runs, config.FIGURES_DIR / "training_curves.png")
        print("wrote training curves")


if __name__ == "__main__":
    main()
