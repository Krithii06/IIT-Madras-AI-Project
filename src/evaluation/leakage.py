"""Quantify what a naive random split actually leaks, without training anything.

The accuracy comparison between the leaf-grouped and random runs is the obvious way
to show leakage, but it only works if the benchmark is hard enough for the optimism
to be visible. On this subset it is not - the binary apple task saturates - so the
accuracy gap understates the problem.

This measures the leak structurally instead: how many validation images have a
photograph of the same physical leaf sitting in the training set. That number does
not depend on the model being weak enough to expose it.

    python -m src.evaluation.leakage
"""

import collections
import json

from src import config
from src.data.dataset import leaf_grouped_split, load_manifest, random_image_split


def sibling_leakage(train_rows, held_out_rows):
    """Share of held-out images whose leaf also appears in training."""
    train_leaves = collections.Counter(r["leaf_id"] for r in train_rows)
    affected = [r for r in held_out_rows if r["leaf_id"] in train_leaves]
    siblings = [train_leaves[r["leaf_id"]] for r in affected]
    return {
        "held_out_images": len(held_out_rows),
        "images_with_a_sibling_in_train": len(affected),
        "leaked_fraction": round(len(affected) / len(held_out_rows), 4),
        "shared_leaves": len({r["leaf_id"] for r in affected}),
        "mean_siblings_per_leaked_image": (round(sum(siblings) / len(siblings), 2)
                                           if siblings else 0.0),
    }


def duplicate_leakage(train_rows, held_out_rows, key):
    """Exact (md5) or near (dhash) duplicates that cross the split boundary."""
    train_keys = {r[key] for r in train_rows if r.get(key)}
    hits = [r for r in held_out_rows if r.get(key) in train_keys]
    return {
        "held_out_images": len(held_out_rows),
        f"{key}_matches_in_train": len(hits),
        "fraction": round(len(hits) / len(held_out_rows), 4),
    }


def main():
    rows = load_manifest()

    leaf_train, leaf_val, test_rows = leaf_grouped_split(rows)
    rand_train, rand_val, _ = random_image_split(rows)

    report = {
        "subset": {
            "images": len(rows),
            "leaves": len({r["leaf_id"] for r in rows}),
            "images_per_leaf": round(len(rows) / len({r["leaf_id"] for r in rows}), 2),
        },
        "leaf_grouped_split": {
            "train_images": len(leaf_train),
            "val_images": len(leaf_val),
            "val_vs_train": sibling_leakage(leaf_train, leaf_val),
            "test_vs_train": sibling_leakage(leaf_train, test_rows),
            "exact_duplicates_val_vs_train": duplicate_leakage(leaf_train, leaf_val, "md5"),
            "near_duplicates_val_vs_train": duplicate_leakage(leaf_train, leaf_val, "dhash"),
        },
        "random_image_split": {
            "train_images": len(rand_train),
            "val_images": len(rand_val),
            "val_vs_train": sibling_leakage(rand_train, rand_val),
            "exact_duplicates_val_vs_train": duplicate_leakage(rand_train, rand_val, "md5"),
            "near_duplicates_val_vs_train": duplicate_leakage(rand_train, rand_val, "dhash"),
        },
    }

    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.METRICS_DIR / "leakage_analysis.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    subset = report["subset"]
    print(f"{subset['images']} images from {subset['leaves']} leaves "
          f"({subset['images_per_leaf']} per leaf)\n")

    for name in ("leaf_grouped_split", "random_image_split"):
        block = report[name]
        v = block["val_vs_train"]
        print(f"{name}:")
        print(f"  validation images                 {v['held_out_images']}")
        print(f"  ... with same leaf in train       {v['images_with_a_sibling_in_train']} "
              f"({v['leaked_fraction'] * 100:.1f}%)")
        print(f"  ... shared leaves                 {v['shared_leaves']}")
        print(f"  exact duplicates crossing split   "
              f"{block['exact_duplicates_val_vs_train']['md5_matches_in_train']}")
        print(f"  near duplicates crossing split    "
              f"{block['near_duplicates_val_vs_train']['dhash_matches_in_train']}")
        print()

    held = report["leaf_grouped_split"]["test_vs_train"]
    print(f"official test set vs leaf-grouped train: "
          f"{held['images_with_a_sibling_in_train']} of {held['held_out_images']} "
          f"images share a leaf ({held['leaked_fraction'] * 100:.1f}%)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
