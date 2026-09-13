"""RetinaNet-style dense anchor classification and regression head."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _group_count(channels: int, requested: int) -> int:
    for groups in range(min(channels, requested), 0, -1):
        if channels % groups == 0 and channels // groups >= 2:
            return groups
    return 1


def _tower(channels: int, depth: int, norm: str, norm_groups: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for _ in range(depth):
        convolution = nn.Conv2d(
            channels, channels, 3, padding=1, bias=norm == "none"
        )
        layers.append(convolution)
        if norm == "group_norm":
            layers.append(nn.GroupNorm(_group_count(channels, norm_groups), channels))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class RetinaNetHead(nn.Module):
    def __init__(
        self,
        channels: int,
        num_classes: int,
        anchors_per_location: int,
        depth: int = 4,
        prior_probability: float = 0.01,
        norm: str = "none",
        norm_groups: int = 32,
        quality_mode: str = "legacy",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.anchors_per_location = anchors_per_location
        self.quality_mode = quality_mode
        self.cls_tower = _tower(channels, depth, norm, norm_groups)
        self.box_tower = _tower(channels, depth, norm, norm_groups)
        self.cls_logits = nn.Conv2d(channels, anchors_per_location * num_classes, 3, padding=1)
        self.bbox_deltas = nn.Conv2d(channels, anchors_per_location * 4, 3, padding=1)
        self.quality = (
            nn.Conv2d(channels, anchors_per_location, 3, padding=1)
            if quality_mode == "iou"
            else None
        )
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
        all_deltas: list[torch.Tensor] = []
        all_quality: list[torch.Tensor] = []
        for feature in features:
            b, _, h, w = feature.shape
            logits = self.cls_logits(self.cls_tower(feature))
            logits = logits.view(
                b, self.anchors_per_location, self.num_classes, h, w
            ).permute(0, 3, 4, 1, 2).reshape(b, -1, self.num_classes)
            box_feature = self.box_tower(feature)
            deltas = self.bbox_deltas(box_feature)
            deltas = deltas.view(b, self.anchors_per_location, 4, h, w)
            deltas = deltas.permute(0, 3, 4, 1, 2).reshape(b, -1, 4)
            all_logits.append(logits)
            all_deltas.append(deltas)
            if self.quality is not None:
                quality = self.quality(box_feature).permute(0, 2, 3, 1).reshape(b, -1)
                all_quality.append(quality)
        output = {"cls_logits": all_logits, "bbox_deltas": all_deltas}
        if all_quality:
            output["quality_logits"] = all_quality
        return output
