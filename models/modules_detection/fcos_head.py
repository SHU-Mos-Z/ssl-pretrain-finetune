"""FCOS-style dense point classification, regression and centerness head."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _tower(channels: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for _ in range(depth):
        layers.extend((nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True)))
    return nn.Sequential(*layers)


class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.scale


class FCOSHead(nn.Module):
    def __init__(
        self,
        channels: int,
        num_classes: int,
        num_levels: int,
        depth: int = 4,
        prior_probability: float = 0.01,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cls_tower = _tower(channels, depth)
        self.box_tower = _tower(channels, depth)
        self.cls_logits = nn.Conv2d(channels, num_classes, 3, padding=1)
        self.bbox_regression = nn.Conv2d(channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(channels, 1, 3, padding=1)
        self.scales = nn.ModuleList(Scale() for _ in range(num_levels))
        self._initialize(prior_probability)

    def _initialize(self, prior_probability: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        bias = -math.log((1.0 - prior_probability) / prior_probability)
        nn.init.constant_(self.cls_logits.bias, bias)

    def forward(self, features: list[torch.Tensor]) -> dict[str, list[torch.Tensor]]:
        all_logits: list[torch.Tensor] = []
        all_regression: list[torch.Tensor] = []
        all_centerness: list[torch.Tensor] = []
        for feature, scale in zip(features, self.scales):
            b, _, h, w = feature.shape
            cls_feature = self.cls_tower(feature)
            box_feature = self.box_tower(feature)
            logits = self.cls_logits(cls_feature).permute(0, 2, 3, 1).reshape(b, -1, self.num_classes)
            regression = F.relu(scale(self.bbox_regression(box_feature)))
            regression = regression.permute(0, 2, 3, 1).reshape(b, -1, 4)
            centerness = self.centerness(box_feature).permute(0, 2, 3, 1).reshape(b, -1)
            all_logits.append(logits)
            all_regression.append(regression)
            all_centerness.append(centerness)
        return {
            "cls_logits": all_logits,
            "bbox_regression": all_regression,
            "centerness_logits": all_centerness,
        }
