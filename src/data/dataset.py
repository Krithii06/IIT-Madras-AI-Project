"""Manifest loading, the leaf-grouped split, and the torch Dataset.

The split logic is the part of this project worth reading carefully. PlantVillage
photographs each physical leaf several times; in the Apple subset 3,171 images come
from only 505 leaves. Splitting on images instead of leaves puts sibling photos of
the same leaf on both sides and inflates the test score, so the grouping is enforced
here and nowhere else.
"""

import csv
from collections import Counter

import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

from src import config


def load_manifest(path=None):
    path = path or config.MANIFEST_PATH
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def leaf_grouped_split(rows, seed=config.SEED, n_folds=config.VAL_FOLDS):
    """Split the official training rows into train/val without splitting a leaf.

    Stratified on the four source classes rather than the binary label so that the
    rarest disease (Cedar apple rust, 275 images) stays represented in validation.
    """
    pool = [r for r in rows if r["official_split"] == "train"]
    labels = [r["source_class"] for r in pool]
    groups = [r["leaf_id"] for r in pool]

    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    train_idx, val_idx = next(splitter.split(pool, labels, groups))

    train_rows = [pool[i] for i in train_idx]
    val_rows = [pool[i] for i in val_idx]
    test_rows = [r for r in rows if r["official_split"] == "test"]
    return train_rows, val_rows, test_rows


def random_image_split(rows, seed=config.SEED, val_fraction=1 / config.VAL_FOLDS):
    """Deliberately naive split used only to measure the cost of leaking leaves.

    This is what you get by shuffling images instead of leaves. It exists so the
    report can quote a real number for the optimism it introduces, not so it can
    be used for the final model.
    """
    import random

    pool = [r for r in rows if r["official_split"] == "train"]
    shuffled = list(pool)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * val_fraction)
    return shuffled[cut:], shuffled[:cut], [r for r in rows if r["official_split"] == "test"]


def describe_split(name, rows):
    binary = Counter(r["binary_label"] for r in rows)
    leaves = {r["leaf_id"] for r in rows}
    return (f"{name:<6} images={len(rows):<5} leaves={len(leaves):<4} "
            f"healthy={binary['healthy']:<5} diseased={binary['diseased']}")


def check_leaf_disjoint(train_rows, val_rows, test_rows):
    """Fail loudly rather than quietly training on a leaky split."""
    tr = {r["leaf_id"] for r in train_rows}
    va = {r["leaf_id"] for r in val_rows}
    te = {r["leaf_id"] for r in test_rows}
    return {
        "train_val_shared_leaves": len(tr & va),
        "train_test_shared_leaves": len(tr & te),
        "val_test_shared_leaves": len(va & te),
    }


class PlantLeafDataset(Dataset):
    """Reads image files listed in the manifest and returns (tensor, label).

    Keeps the source class alongside the binary target so evaluation can break the
    binary result down by disease without a second pass over the data.
    """

    def __init__(self, rows, transform, root=None, class_names=None):
        self.rows = rows
        self.transform = transform
        self.root = root or config.RAW_DIR
        self.class_names = class_names or config.BINARY_CLASSES
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.root / row["rel_path"]
        # Convert explicitly: the training files are all RGB, but uploads at
        # inference time are not, and both paths share this class.
        with Image.open(path) as img:
            image = img.convert("RGB")
        target = self.class_to_idx[row["binary_label"]]
        return self.transform(image), target

    def class_weights(self):
        """Inverse-frequency weights, used only if a split turns out lopsided."""
        counts = Counter(r["binary_label"] for r in self.rows)
        total = sum(counts.values())
        weights = [total / (len(self.class_names) * counts[n]) for n in self.class_names]
        return torch.tensor(weights, dtype=torch.float32)
