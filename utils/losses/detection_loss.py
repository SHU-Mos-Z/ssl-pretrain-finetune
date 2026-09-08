"""Criterion factory exposing one stable training call for both modes."""

from __future__ import annotations

import torch.nn as nn

from models.detection_contracts import DetectionConfig
from .anchor_detection_loss import AnchorDetectionLoss
from .fcos_detection_loss import FCOSDetectionLoss


class DetectionCriterion(nn.Module):
    def __init__(self, config: DetectionConfig):
        super().__init__()
        self.config = config
        self.criterion = (
            AnchorDetectionLoss(config)
            if config.detection_mode == "anchor_based"
            else FCOSDetectionLoss(config)
        )

    def forward(self, output: dict, targets: list[dict]):
        return self.criterion(output, targets)
