"""Build models by name and control which parts are trainable in each training phase."""

from __future__ import annotations

from typing import Any

from torch import nn

from waste.models.custom_cnn import CustomCNN
from waste.models.mobilenet import mobilenet_v2, resnet18

MODEL_NAMES = ("custom_cnn", "mobilenet_v2", "resnet18")


def build_model(name: str, num_classes: int, **kwargs: Any) -> nn.Module:
    """Create a model from its config name and keyword arguments."""
    if name == "custom_cnn":
        return CustomCNN(
            num_classes,
            channels=tuple(kwargs.get("channels", (32, 64, 128, 256))),
            dropout=kwargs.get("dropout", 0.4),
        )
    if name == "mobilenet_v2":
        return mobilenet_v2(num_classes, kwargs.get("pretrained", True), kwargs.get("dropout", 0.3))
    if name == "resnet18":
        return resnet18(num_classes, kwargs.get("pretrained", True), kwargs.get("dropout", 0.3))
    raise ValueError(f"Unknown model '{name}'. Choose one of {MODEL_NAMES}.")


def model_kwargs(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Keyword arguments for ``build_model`` from the 'model' section of a training config."""
    return {k: v for k, v in model_cfg.items() if k != "name"}


def head(model: nn.Module, name: str) -> nn.Module:
    """The classifier head of a model."""
    return model.fc if name == "resnet18" else model.classifier


def backbone_blocks(model: nn.Module, name: str) -> list[nn.Module]:
    """Feature-extractor blocks in input-to-output order (used for partial unfreezing)."""
    if name == "resnet18":
        return [model.layer1, model.layer2, model.layer3, model.layer4]
    return list(model.features)


def set_trainable(model: nn.Module, name: str, mode: str, unfreeze_blocks: int = 0) -> int:
    """Freeze or unfreeze parameters for a phase and return the trainable parameter count.

    mode: 'all' (everything), 'head' (classifier only) or 'last_blocks' (head plus the last
    ``unfreeze_blocks`` backbone blocks).
    """
    if mode not in {"all", "head", "last_blocks"}:
        raise ValueError(f"Unknown trainable mode '{mode}'")
    for p in model.parameters():
        p.requires_grad = mode == "all"
    if mode != "all":
        for p in head(model, name).parameters():
            p.requires_grad = True
    if mode == "last_blocks" and unfreeze_blocks > 0:
        for block in backbone_blocks(model, name)[-unfreeze_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_mode(model: nn.Module) -> None:
    """Switch to train mode but keep BatchNorm layers of frozen blocks in eval mode."""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            params = list(m.parameters())
            if params and not any(p.requires_grad for p in params):
                m.eval()


def gradcam_target_layer(model: nn.Module, name: str) -> nn.Module:
    """The last convolutional stage, used as the Grad-CAM target."""
    if name == "resnet18":
        return model.layer4[-1]
    if name == "custom_cnn":
        return model.features[-1][0]  # last Conv2d, before pooling shrinks the map
    return model.features[-1]
