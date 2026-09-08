"""Dynamic absolute-coordinate anchor generation."""

from __future__ import annotations

import math
from collections import OrderedDict

import torch


class AnchorGenerator:
    def __init__(
        self,
        sizes: tuple[float, ...],
        scales: tuple[float, ...],
        aspect_ratios: tuple[float, ...],
        offset: float = 0.5,
    ):
        self.sizes = tuple(float(value) for value in sizes)
        self.scales = tuple(float(value) for value in scales)
        self.aspect_ratios = tuple(float(value) for value in aspect_ratios)
        self.offset = float(offset)
        if not 0 <= self.offset <= 1:
            raise ValueError("anchor offset must be in [0,1]")

    @property
    def num_anchors_per_location(self) -> int:
        return len(self.scales) * len(self.aspect_ratios)

    def _templates(self, size: float, device, dtype) -> torch.Tensor:
        widths, heights = [], []
        for scale in self.scales:
            for ratio in self.aspect_ratios:
                widths.append(size * scale * math.sqrt(ratio))
                heights.append(size * scale / math.sqrt(ratio))
        widths = torch.tensor(widths, device=device, dtype=dtype)
        heights = torch.tensor(heights, device=device, dtype=dtype)
        return torch.stack((-widths / 2, -heights / 2, widths / 2, heights / 2), dim=1)

    def __call__(
        self,
        features: OrderedDict[str, torch.Tensor],
        strides: OrderedDict[str, int],
    ) -> list[torch.Tensor]:
        if len(features) != len(self.sizes) or tuple(features) != tuple(strides):
            raise ValueError("feature/stride/anchor-size levels do not agree")
        anchors = []
        for (name, feature), size in zip(features.items(), self.sizes):
            _, _, height, width = feature.shape
            stride = float(strides[name])
            shifts_x = (torch.arange(width, device=feature.device, dtype=torch.float32) + self.offset) * stride
            shifts_y = (torch.arange(height, device=feature.device, dtype=torch.float32) + self.offset) * stride
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=-1).reshape(-1, 4)
            templates = self._templates(size, feature.device, torch.float32)
            anchors.append((shifts[:, None, :] + templates[None, :, :]).reshape(-1, 4))
        return anchors
