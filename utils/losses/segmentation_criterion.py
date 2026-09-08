"""Configurable segmentation losses while preserving the historical default."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.losses.soft_dice_ce_loss import SoftDiceCrossEntropyLoss


SEGMENTATION_LOSS_TYPES = (
    "ce_dice",
    "weighted_ce_dice",
    "focal_dice",
    "weighted_ce_dice_boundary",
)


def primary_segmentation_logits(
    prediction: torch.Tensor | dict[str, object],
) -> torch.Tensor:
    if isinstance(prediction, torch.Tensor):
        return prediction
    logits = prediction.get("logits")
    if not isinstance(logits, torch.Tensor):
        raise TypeError("segmentation prediction dictionary must contain tensor 'logits'")
    return logits


class ConfigurableSegmentationLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        *,
        loss_type: str,
        class_weights: torch.Tensor | None = None,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_gamma: float = 2.0,
        boundary_weight: float = 0.0,
        auxiliary_weight: float = 0.0,
        ignore_index: int = -1,
    ):
        super().__init__()
        if loss_type not in SEGMENTATION_LOSS_TYPES:
            raise ValueError(f"loss_type must be one of {SEGMENTATION_LOSS_TYPES}")
        self.num_classes = int(num_classes)
        self.loss_type = loss_type
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.focal_gamma = float(focal_gamma)
        self.boundary_weight = float(boundary_weight)
        self.auxiliary_weight = float(auxiliary_weight)
        self.ignore_index = int(ignore_index)
        weights = (
            torch.ones(self.num_classes, dtype=torch.float32)
            if class_weights is None
            else torch.as_tensor(class_weights, dtype=torch.float32)
        )
        if weights.shape != (self.num_classes,) or torch.any(weights <= 0):
            raise ValueError("class_weights must contain one positive value per class")
        self.register_buffer("class_weights", weights)

    def _soft_dice_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target.ne(self.ignore_index)
        safe = target.masked_fill(~valid, 0)
        one_hot = F.one_hot(safe, self.num_classes).permute(0, 3, 1, 2).to(logits.dtype)
        mask = valid[:, None]
        probabilities = logits.softmax(dim=1)
        intersection = (probabilities * one_hot * mask).sum((0, 2, 3))
        denominator = ((probabilities + one_hot) * mask).sum((0, 2, 3))
        class_dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        if self.loss_type.startswith("weighted_"):
            normalized = self.class_weights / self.class_weights.sum()
            return 1.0 - (class_dice * normalized).sum()
        return 1.0 - class_dice.mean()

    def _classification_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = self.class_weights if self.loss_type.startswith("weighted_") else None
        if self.loss_type == "focal_dice":
            ce = F.cross_entropy(
                logits, target, reduction="none", ignore_index=self.ignore_index
            )
            valid = target.ne(self.ignore_index)
            if not bool(valid.any()):
                return logits.sum() * 0.0
            probability_true = torch.exp(-ce[valid])
            return (((1.0 - probability_true) ** self.focal_gamma) * ce[valid]).mean()
        return F.cross_entropy(
            logits, target, weight=weight, ignore_index=self.ignore_index
        )

    def _boundary_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target.ne(self.ignore_index)
        safe = target.masked_fill(~valid, 0)
        one_hot = F.one_hot(safe, self.num_classes).permute(0, 3, 1, 2).to(logits.dtype)
        probabilities = logits.softmax(dim=1)

        def boundary_map(value: torch.Tensor) -> torch.Tensor:
            dilation = F.max_pool2d(value, 3, stride=1, padding=1)
            erosion = -F.max_pool2d(-value, 3, stride=1, padding=1)
            return (dilation - erosion).clamp(0.0, 1.0)

        # Foreground-class boundaries are the clinically relevant structures;
        # background boundaries would duplicate the same interfaces.
        pred_boundary = boundary_map(probabilities[:, 1:])
        target_boundary = boundary_map(one_hot[:, 1:])
        boundary_valid = valid[:, None]
        intersection = (pred_boundary * target_boundary * boundary_valid).sum((0, 2, 3))
        denominator = ((pred_boundary + target_boundary) * boundary_valid).sum((0, 2, 3))
        dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        return 1.0 - dice.mean()

    def _single(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        classification = self._classification_loss(logits, target)
        dice = self._soft_dice_loss(logits, target)
        boundary = (
            self._boundary_loss(logits, target)
            if self.boundary_weight > 0
            else logits.new_zeros(())
        )
        total = (
            self.ce_weight * classification
            + self.dice_weight * dice
            + self.boundary_weight * boundary
        )
        return total, {
            "loss_classification": classification.detach(),
            "loss_dice": dice.detach(),
            "loss_boundary": boundary.detach(),
        }

    def forward(self, prediction, target):
        logits = primary_segmentation_logits(prediction)
        loss, components = self._single(logits, target)
        auxiliary_loss = logits.new_zeros(())
        if isinstance(prediction, dict) and self.auxiliary_weight > 0:
            auxiliary = prediction.get("aux_logits", [])
            if not isinstance(auxiliary, (list, tuple)):
                raise TypeError("aux_logits must be a list or tuple")
            if auxiliary:
                auxiliary_loss = torch.stack(
                    [self._single(item, target)[0] for item in auxiliary]
                ).mean()
                loss = loss + self.auxiliary_weight * auxiliary_loss
        logs = {
            "loss_seg": float(loss.detach()),
            "loss_ce_or_focal": float(components["loss_classification"]),
            "loss_dice": float(components["loss_dice"]),
            "loss_boundary": float(components["loss_boundary"]),
            "loss_aux": float(auxiliary_loss.detach()),
        }
        return loss, logs


def build_segmentation_criterion(
    num_classes: int,
    *,
    loss_type: str = "ce_dice",
    class_weights: torch.Tensor | None = None,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
    focal_gamma: float = 2.0,
    boundary_weight: float = 0.0,
    auxiliary_weight: float = 0.0,
) -> nn.Module:
    if (
        loss_type == "ce_dice"
        and class_weights is None
        and ce_weight == 1.0
        and dice_weight == 1.0
        and boundary_weight == 0.0
        and auxiliary_weight == 0.0
    ):
        # Preserve the exact historical implementation for baseline runs.
        return SoftDiceCrossEntropyLoss(num_classes)
    return ConfigurableSegmentationLoss(
        num_classes,
        loss_type=loss_type,
        class_weights=class_weights,
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        focal_gamma=focal_gamma,
        boundary_weight=boundary_weight,
        auxiliary_weight=auxiliary_weight,
    )


__all__ = [
    "SEGMENTATION_LOSS_TYPES",
    "ConfigurableSegmentationLoss",
    "build_segmentation_criterion",
    "primary_segmentation_logits",
]
