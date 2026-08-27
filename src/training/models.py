"""Model factory for the three architectures named in the brief.

All three are ImageNet-pretrained and small enough to run on a free CPU host.
Only the classifier head is replaced; everything else stays as torchvision built it.
"""

from torchvision import models

# Each entry: constructor, pretrained weights enum, attribute path of the classifier.
ARCHITECTURES = {
    "mobilenet_v2": (
        models.mobilenet_v2,
        models.MobileNet_V2_Weights.IMAGENET1K_V1,
        "classifier",
    ),
    "efficientnet_b0": (
        models.efficientnet_b0,
        models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "classifier",
    ),
    "resnet18": (
        models.resnet18,
        models.ResNet18_Weights.IMAGENET1K_V1,
        "fc",
    ),
}


def create_model(arch, num_classes, pretrained=True):
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {arch!r}; expected one of {list(ARCHITECTURES)}")

    build, weights, head_attr = ARCHITECTURES[arch]
    model = build(weights=weights if pretrained else None)

    import torch.nn as nn

    head = getattr(model, head_attr)
    if isinstance(head, nn.Sequential):
        # mobilenet_v2 and efficientnet_b0 wrap Dropout + Linear; keep the dropout.
        in_features = head[-1].in_features
        head[-1] = nn.Linear(in_features, num_classes)
    else:
        setattr(model, head_attr, nn.Linear(head.in_features, num_classes))

    return model


def head_parameters(model, arch):
    head_attr = ARCHITECTURES[arch][2]
    return list(getattr(model, head_attr).parameters())


def set_backbone_trainable(model, arch, trainable):
    """Freeze or unfreeze everything except the classifier head.

    Stage one trains the head alone, which is cheap and stops the randomly
    initialised head from pushing large gradients into the pretrained features.
    """
    head_ids = {id(p) for p in head_parameters(model, arch)}
    for param in model.parameters():
        if id(param) not in head_ids:
            param.requires_grad = trainable
