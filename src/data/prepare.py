"""Build the working dataset: pull the selected classes out of the PlantVillage
archive, attach leaf-group ids, and run integrity checks.

Produces data/manifest.csv, which every later stage reads. Run once:

    python -m src.data.prepare --zip D:\\pv_cache\\raw\\data.zip
    python -m src.data.prepare --download          # fetch the 2.2 GB archive first
"""

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from collections import Counter

import numpy as np
from PIL import Image

from src import config

# Only these files are needed to reproduce the official split and the leaf groups.
META_FILES = [
    "leaf_grouping/leaf-map.json",
    f"splits/{config.IMAGE_VARIANT}_train.txt",
    f"splits/{config.IMAGE_VARIANT}_test.txt",
]


def fetch_metadata():
    """Download the small split/leaf-map files (about 5 MB total)."""
    from huggingface_hub import hf_hub_download

    config.META_DIR.mkdir(parents=True, exist_ok=True)
    for name in META_FILES:
        hf_hub_download(
            config.HF_DATASET_REPO,
            name,
            repo_type="dataset",
            local_dir=str(config.META_DIR),
        )
    return config.META_DIR


def fetch_archive():
    from huggingface_hub import hf_hub_download

    return hf_hub_download(config.HF_DATASET_REPO, "data.zip", repo_type="dataset")


def _leaf_key(file_name):
    """Normalise a filename to the key used by the dataset's leaf map.

    Mirrors the logic in the upstream plant_village.py loader; if we invented our
    own normalisation the ids would not line up with the published leaf groups.
    """
    ident = file_name.replace("_final_masked", "")
    if "___" in ident:
        ident = ident.split("___")[-1]
    ident = ident.split("copy")[0]
    for ext in (".jpg", ".JPG", ".png", ".PNG"):
        ident = ident.replace(ext, "")
    return ident.strip().lower()


def resolve_leaf_id(leaf_map, source_class, file_name):
    """Return (leaf_id, resolved_from_map).

    A leaf id groups every photograph taken of one physical leaf. When the map has
    no entry we fall back to a per-image id, which is the conservative choice: the
    image then forms its own group and can never be split across train and test.
    """
    key = _leaf_key(file_name)
    hits = leaf_map.get(key)
    if not hits:
        return "fallback_" + key, False
    if len(hits) == 1:
        return hits[0], True
    for hit in hits:
        if source_class in hit:
            return hit, True
    return "fallback_" + key, False


def dhash(image, size=8):
    """64-bit difference hash, used to spot near-duplicate photographs.

    Cheap enough for a few thousand images and avoids pulling in imagehash just
    for one function.
    """
    small = image.convert("L").resize((size + 1, size), Image.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    return "".join("1" if b else "0" for b in bits.flatten())


def read_split(meta_dir, split):
    path = meta_dir / "splits" / f"{config.IMAGE_VARIANT}_{split}.txt"
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def extract_and_index(zip_path, meta_dir):
    """Extract the selected classes and return one manifest row per image."""
    leaf_map = json.load(open(meta_dir / "leaf_grouping" / "leaf-map.json", encoding="utf-8"))
    wanted = set(config.SOURCE_CLASSES)

    entries = []
    for split in ("train", "test"):
        for rel in read_split(meta_dir, split):
            parts = rel.split("/")
            if len(parts) < 4:
                continue
            source_class, file_name = parts[2], parts[3]
            if source_class in wanted:
                entries.append((split, source_class, file_name, rel))

    print(f"selected {len(entries)} images across {len(wanted)} source classes")

    rows = []
    corrupt = []
    with zipfile.ZipFile(zip_path) as zf:
        available = set(zf.namelist())
        for split, source_class, file_name, rel in entries:
            if rel not in available:
                corrupt.append((rel, "missing from archive"))
                continue

            out_path = config.RAW_DIR / source_class / file_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = zf.read(rel)
            out_path.write_bytes(payload)

            # Decode every image once now rather than discovering a bad file
            # halfway through the first training run.
            try:
                with Image.open(out_path) as img:
                    img.verify()
                with Image.open(out_path) as img:
                    width, height = img.size
                    mode = img.mode
                    digest = dhash(img)
            except Exception as exc:  # noqa: BLE001 - we want the filename, not the class
                corrupt.append((rel, f"{type(exc).__name__}: {exc}"))
                out_path.unlink(missing_ok=True)
                continue

            leaf_id, from_map = resolve_leaf_id(leaf_map, source_class, file_name)
            rows.append(
                {
                    "rel_path": f"{source_class}/{file_name}",
                    "source_class": source_class,
                    "binary_label": config.to_binary_label(source_class),
                    "leaf_id": leaf_id,
                    "leaf_id_from_map": int(from_map),
                    "official_split": split,
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "bytes": len(payload),
                    "md5": hashlib.md5(payload).hexdigest(),
                    "dhash": digest,
                }
            )

    return rows, corrupt


def report(rows, corrupt):
    """Print the dataset facts we care about before training anything."""
    print("\n--- integrity ---")
    print(f"usable images   : {len(rows)}")
    print(f"unreadable      : {len(corrupt)}")
    for rel, why in corrupt[:10]:
        print(f"   {rel}: {why}")

    sizes = Counter((r["width"], r["height"]) for r in rows)
    modes = Counter(r["mode"] for r in rows)
    print(f"resolutions     : {dict(sizes)}")
    print(f"colour modes    : {dict(modes)}")

    md5s = Counter(r["md5"] for r in rows)
    exact = {k: v for k, v in md5s.items() if v > 1}
    print(f"byte-identical duplicate groups : {len(exact)} "
          f"({sum(exact.values()) - len(exact)} redundant files)")

    hashes = Counter(r["dhash"] for r in rows)
    near = {k: v for k, v in hashes.items() if v > 1}
    print(f"identical dHash groups          : {len(near)} "
          f"({sum(near.values()) - len(near)} redundant files)")

    print("\n--- class distribution ---")
    src = Counter(r["source_class"] for r in rows)
    for name in config.SOURCE_CLASSES:
        leaves = len({r["leaf_id"] for r in rows if r["source_class"] == name})
        print(f"   {name:<32}{src[name]:>6} images  {leaves:>4} leaves")
    binary = Counter(r["binary_label"] for r in rows)
    print(f"   binary target: {dict(binary)}")

    from_map = sum(r["leaf_id_from_map"] for r in rows)
    print(f"\nleaf ids resolved from the published map: {from_map}/{len(rows)} "
          f"({100 * from_map / len(rows):.1f}%)")

    tr = {r["leaf_id"] for r in rows if r["official_split"] == "train"}
    te = {r["leaf_id"] for r in rows if r["official_split"] == "test"}
    print(f"leaf ids shared by official train and test: {len(tr & te)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", help="path to an already downloaded data.zip")
    parser.add_argument("--download", action="store_true",
                        help="download data.zip from the Hugging Face hub first")
    args = parser.parse_args()

    meta_dir = fetch_metadata()

    if args.download:
        zip_path = fetch_archive()
    elif args.zip:
        zip_path = args.zip
    else:
        parser.error("pass --zip <path> or --download")

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows, corrupt = extract_and_index(zip_path, meta_dir)
    if not rows:
        sys.exit("no usable images were extracted")

    report(rows, corrupt)

    config.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.MANIFEST_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {config.MANIFEST_PATH} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
