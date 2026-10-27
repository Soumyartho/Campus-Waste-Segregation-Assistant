"""M0: custom CNN trained from scratch (PLAN.md Section 7.2).

Four blocks of Conv3x3 -> BatchNorm -> ReLU -> MaxPool, then global average pooling,
dropout and a linear classifier.
"""

from __future__ import annotations

import torch
from torch import nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv3x3 + BatchNorm + ReLU + 2x2 MaxPool."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CustomCNN(nn.Module):
    """Baseline CNN; ``channels`` gives the filter count of each block."""

    def __init__(
        self,
        num_classes: int,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        blocks, in_ch = [], 3
        for out_ch in channels:
            blocks.append(conv_block(in_ch, out_ch))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_ch, num_classes))
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x)).flatten(1)
        return self.classifier(x)
