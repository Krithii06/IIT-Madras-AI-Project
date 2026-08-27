# Data

The images are not committed. The full PlantVillage archive is about 2.2 GB and the
task brief asks for the dataset source rather than the dataset itself.

## Source

PlantVillage, 54,305 images across 38 classes and 14 crops.

- Dataset: <https://huggingface.co/datasets/mohanty/PlantVillage>
- Upstream repository: <https://github.com/spMohanty/PlantVillage-Dataset>
- Paper: Mohanty, Hughes & Salathé (2016), *Using deep learning for image-based
  plant disease detection*, Frontiers in Plant Science.
- Licence: CC BY-SA 3.0

## Rebuilding this directory

```bash
python -m src.data.prepare --download          # fetches the 2.2 GB archive
python -m src.data.prepare --zip path/to/data.zip   # or reuse a local copy
```

That extracts only the four Apple classes from the `color` variant and writes:

```
data/
├── raw/                     3,171 JPEGs in four class folders
├── meta/                    official split files + leaf-map.json
└── manifest.csv             one row per image, the input to every later stage
```

The download is cached, so re-running the script is cheap.

## What the subset contains

Four of the 38 PlantVillage classes, all apple, three diseased and one healthy:

| Source class | Images | Binary label |
|---|---:|---|
| `Apple___Apple_scab` | 630 | diseased |
| `Apple___Black_rot` | 621 | diseased |
| `Apple___Cedar_apple_rust` | 275 | diseased |
| `Apple___healthy` | 1,645 | healthy |

All 3,171 images are 256×256 RGB JPEG. None are corrupt.

## manifest.csv

One row per image. The columns that matter later:

| Column | Why it exists |
|---|---|
| `rel_path` | location under `data/raw/` |
| `source_class` | original PlantVillage class |
| `binary_label` | `healthy` or `diseased`, derived from `source_class` |
| `leaf_id` | which physical leaf the photo is of — the basis of the split |
| `leaf_id_from_map` | 1 if resolved from the official leaf map, 0 if a fallback |
| `official_split` | PlantVillage's own 80/20 assignment |
| `md5`, `dhash` | duplicate and near-duplicate detection |

## Why `leaf_id` matters

PlantVillage photographs each physical leaf several times from different angles.
In this subset **3,171 images come from only 505 distinct leaves** — a mean of 6.3
images per leaf. Splitting on images rather than leaves puts near-identical photos
of the same leaf on both sides of the split and reports a test score that the model
has not earned.

Every Apple image resolves to a real entry in the official leaf map (0 fallbacks),
which is why this crop was chosen: the grouping can be verified rather than assumed.
`tests/test_data.py` asserts the resulting split shares no leaf between train, val
and test.
