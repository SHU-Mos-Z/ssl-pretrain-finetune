"""Dynamic FCOS reference-point generation."""

from __future__ import annotations

from collections import OrderedDict

import torch


class PointGenerator:
    def __init__(self, offset: float = 0.5):
        self.offset = float(offset)
        if not 0 <= self.offset <= 1:
            raise ValueError("point offset must be in [0,1]")

    def __call__(
        self,
        features: OrderedDict[str, torch.Tensor],
        strides: OrderedDict[str, int],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        if tuple(features) != tuple(strides):
            raise ValueError("feature and stride levels do not agree")
        points, point_strides, level_ids = [], [], []
        for level, (name, feature) in enumerate(features.items()):
            _, _, height, width = feature.shape
            stride = int(strides[name])
            xs = (torch.arange(width, device=feature.device, dtype=torch.float32) + self.offset) * stride
            ys = (torch.arange(height, device=feature.device, dtype=torch.float32) + self.offset) * stride
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            level_points = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
            points.append(level_points)
            point_strides.append(torch.full((len(level_points),), float(stride), device=feature.device))
            level_ids.append(torch.full((len(level_points),), level, dtype=torch.long, device=feature.device))
        return points, point_strides, level_ids
