"""Training entry point.

Two-stage transfer learning: train the new head with the backbone frozen, then
unfreeze and fine-tune the whole network at a lower learning rate. Everything the
later stages need - weights, class mapping, preprocessing settings, history - is
written into a single run directory under models/.

    python -m src.training.train --arch mobilenet_v2 --run-name mobilenet_leaf
    python -m src.training.train --arch mobilenet_v2 --split random --run-name mobilenet_random
"""

import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src import config
from src.data.dataset import (PlantLeafDataset, check_leaf_disjoint, describe_split,
                              leaf_grouped_split, load_manifest, random_image_split)
from src.preprocessing.transforms import eval_transform, train_transform
from src.training.models import create_model, head_parameters, set_backbone_trainable


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(train_rows, val_rows, batch_size, workers):
    train_ds = PlantLeafDataset(train_rows, train_transform())
    val_ds = PlantLeafDataset(val_rows, eval_transform())

    # persistent_workers avoids re-spawning processes every epoch, which is the
    # expensive part on Windows where workers are spawned rather than forked.
    extra = {"persistent_workers": True} if workers else {}
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, **extra)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=workers, **extra)
    return train_ds, train_loader, val_loader


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss, seen = 0.0, 0
    all_preds, all_targets = [], []

    with torch.set_grad_enabled(training):
        for images, targets in loader:
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            seen += images.size(0)
            all_preds.append(logits.argmax(1))
            all_targets.append(targets)

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    return {
        "loss": total_loss / seen,
        "accuracy": float((preds == targets).mean()),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
    }


def train(args):
    set_seed(args.seed)
    torch.set_num_threads(args.threads)

    rows = load_manifest()
    splitter = leaf_grouped_split if args.split == "leaf" else random_image_split
    train_rows, val_rows, test_rows = splitter(rows, seed=args.seed)

    print(f"split strategy: {args.split}")
    for name, subset in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        print("  " + describe_split(name, subset))
    leakage = check_leaf_disjoint(train_rows, val_rows, test_rows)
    print(f"  leaf overlap: {leakage}")

    train_ds, train_loader, val_loader = make_loaders(
        train_rows, val_rows, args.batch_size, args.workers
    )

    model = create_model(args.arch, len(config.BINARY_CLASSES))

    # The Apple binary target is close to balanced (53/47), so weighting is off by
    # default; the flag stays available because other class selections are not.
    weights = train_ds.class_weights() if args.class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weights)

    run_dir = config.MODELS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_f1, best_epoch, epochs_without_gain = -1.0, -1, 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        if epoch == 1 and args.warmup_epochs > 0:
            set_backbone_trainable(model, args.arch, False)
            params = head_parameters(model, args.arch)
            optimizer = torch.optim.AdamW(params, lr=args.head_lr, weight_decay=1e-4)
            scheduler = None
            stage = "head"
        elif epoch == args.warmup_epochs + 1:
            set_backbone_trainable(model, args.arch, True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr,
                                          weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - args.warmup_epochs)
            )
            stage = "finetune"

        t0 = time.time()
        epoch_lr = optimizer.param_groups[0]["lr"]
        train_metrics = run_epoch(model, train_loader, criterion, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion)
        if scheduler is not None:
            scheduler.step()

        record = {
            "epoch": epoch,
            "stage": stage,
            "lr": epoch_lr,
            "seconds": round(time.time() - t0, 1),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)
        print(f"epoch {epoch:>2}/{args.epochs} [{stage:<8}] "
              f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} | "
              f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f} "
              f"macroF1 {val_metrics['macro_f1']:.4f} | {record['seconds']}s")

        # Select on macro F1 rather than accuracy so neither class can be ignored.
        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch, epochs_without_gain = val_metrics["macro_f1"], epoch, 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        else:
            epochs_without_gain += 1
            if args.patience and epochs_without_gain >= args.patience:
                print(f"early stop: no val macro-F1 gain for {args.patience} epochs")
                break

    elapsed = time.time() - started

    with open(run_dir / "class_mapping.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "classes": config.BINARY_CLASSES,
                "class_to_idx": {n: i for i, n in enumerate(config.BINARY_CLASSES)},
                "positive_class": config.POSITIVE_CLASS,
                "source_classes": config.SOURCE_CLASSES,
            },
            fh,
            indent=2,
        )

    with open(run_dir / "train_config.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                **vars(args),
                "image_size": config.IMAGE_SIZE,
                "resize_before_crop": config.RESIZE_BEFORE_CROP,
                "normalize_mean": config.IMAGENET_MEAN,
                "normalize_std": config.IMAGENET_STD,
                "train_images": len(train_rows),
                "val_images": len(val_rows),
                "test_images": len(test_rows),
                "leaf_overlap": leakage,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_f1,
                "total_train_seconds": round(elapsed, 1),
                "torch_version": torch.__version__,
            },
            fh,
            indent=2,
        )

    with open(run_dir / "history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    print(f"\nbest val macro-F1 {best_f1:.4f} at epoch {best_epoch}; "
          f"{elapsed/60:.1f} min total -> {run_dir}")
    return run_dir


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", default="mobilenet_v2",
                   choices=["mobilenet_v2", "efficientnet_b0", "resnet18"])
    p.add_argument("--run-name", required=True)
    p.add_argument("--split", default="leaf", choices=["leaf", "random"],
                   help="'random' is only for the leakage comparison")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--warmup-epochs", type=int, default=2,
                   help="epochs with the backbone frozen before fine-tuning")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=4, help="0 disables early stopping")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--class-weights", action="store_true")
    p.add_argument("--seed", type=int, default=config.SEED)
    return p


if __name__ == "__main__":
    # Windows spawns dataloader workers instead of forking, so the entry point
    # must be guarded or every worker re-executes this module.
    train(build_parser().parse_args())
