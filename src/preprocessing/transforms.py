"""Image transforms for training and for deterministic evaluation/inference.

Source images are 256x256 RGB. The eval transform is shared with the backend so
that what the model sees in production matches what it saw during validation.
"""

from torchvision import transforms

from src import config


def train_transform(image_size=config.IMAGE_SIZE):
    """Augmentation tuned for detached leaves photographed on a plain background.

    Both flips are enabled: a picked leaf on a uniform background has no canonical
    orientation, so mirroring it is a realistic variation rather than a distortion.

    Colour jitter is deliberately mild and hue is nearly frozen (0.02). Leaf colour
    carries the diagnosis here - chlorotic yellowing, rust orange, necrotic brown -
    so a wide hue shift would erase the very signal the model has to learn.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def eval_transform(image_size=config.IMAGE_SIZE, resize=config.RESIZE_BEFORE_CROP):
    """Deterministic path: resize to 256, centre crop, normalise.

    No randomness at all, so validation and test numbers are repeatable and the
    deployed API returns the same answer for the same upload every time.
    """
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
