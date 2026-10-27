"""M1: MobileNetV2 and M2: ResNet18 with ImageNet weights and a new classifier head."""

from __future__ import annotations

from torch import nn
from torchvision import models


def mobilenet_v2(num_classes: int, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """MobileNetV2 whose classifier is replaced by Dropout + Linear(1280, num_classes)."""
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(dropout), nn.Linear(model.last_channel, num_classes)
    )
    return model


def resnet18(num_classes: int, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """ResNet18 whose fc layer is replaced by Dropout + Linear(512, num_classes)."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    return model
