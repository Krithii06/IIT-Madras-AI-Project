"""Central configuration: paths, class selection, split and training defaults.

Everything that another module might want to agree on lives here so the training
scripts, the evaluation scripts and the dataset builder cannot drift apart.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
META_DIR = DATA_DIR / "meta"
MANIFEST_PATH = DATA_DIR / "manifest.csv"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

HF_DATASET_REPO = "mohanty/PlantVillage"
IMAGE_VARIANT = "color"

# PlantVillage ships 38 classes across 14 crops. We keep the four Apple classes.
# The brief asks for 3-5 source classes including at least one healthy, and Apple
# is the only crop whose leaf-grouping metadata covers 100% of its images, which
# is what makes the leak-free split below actually verifiable.
SOURCE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
]

# The deployed model is binary. Index order is pinned here and copied into
# class_mapping.json at train time so inference never has to re-derive it.
BINARY_CLASSES = ["healthy", "diseased"]

# "diseased" is the positive class for binary precision/recall: the useful
# question is how well the model flags a problem, not how well it confirms health.
POSITIVE_CLASS = "diseased"

SEED = 42

# StratifiedGroupKFold over leaf ids; we take fold 0 as validation (~1/6 of leaves).
VAL_FOLDS = 6

# 160 rather than the usual 224. Measured on the 4-core CPU this project trains and
# deploys on, a MobileNetV2 fine-tune step costs 2.95s at 224 against 1.90s at 160,
# and single-image inference drops in step. The source photos are 256x256 close-ups
# of a single leaf filling the frame, so the extra resolution buys little here, and
# the free hosting tier is CPU-only with a request timeout worth staying well under.
IMAGE_SIZE = 160

# Keep the same ~1.14 resize-to-crop ratio that the 256/224 ImageNet recipe uses,
# so the centre crop trims the border rather than cutting into the leaf.
RESIZE_BEFORE_CROP = 182

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_binary_label(source_class: str) -> str:
    """PlantVillage encodes health in the class name after the triple underscore."""
    return "healthy" if source_class.endswith("___healthy") else "diseased"
