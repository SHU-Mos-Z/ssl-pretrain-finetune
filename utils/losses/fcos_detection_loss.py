"""FCOS point assignment, focal, GIoU and centerness losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.detection_contracts import DetectionConfig
from models.modules_detection import FCOSPointAssigner
from models.modules_detection.box_ops import aligned_box_iou, generalized_box_iou, sigmoid_focal_loss
from .detection_common import distributed_normalizer, target_to_device


def _decode_distances(
    points: torch.Tensor, distances: torch.Tensor, strides: torch.Tensor, normalized: bool
) -> torch.Tensor:
    values = distances * strides[:, None] if normalized else distances
    return torch.stack(
        (
            points[:, 0] - values[:, 0],
            points[:, 1] - values[:, 1],
            points[:, 0] + values[:, 2],
            points[:, 1] + values[:, 3],
        ),
        dim=1,
    )


class FCOSDetectionLoss(nn.Module):
    def __init__(self, config: DetectionConfig):
        super().__init__()
        self.config = config
        self.assigner = FCOSPointAssigner(
            config.fcos_ranges_for_features(),
            config.fcos_center_sampling_radius,
            config.fcos_normalize_reg_targets_by_stride,
        )

    def forward(self, output: dict, targets: list[dict]) -> dict[str, torch.Tensor]:
        if output.get("detection_mode") != self.config.detection_mode:
            raise ValueError("model output and FCOS criterion detection modes differ")
        logits = torch.cat(output["cls_logits"], dim=1).float()
        regression = torch.cat(output["bbox_regression"], dim=1).float()
        centerness_logits = (
            torch.cat(output["centerness_logits"], dim=1).float()
            if self.config.quality_mode == "legacy"
            else None
        )
        quality_logits = (
            torch.cat(output["quality_logits"], dim=1).float()
            if self.config.quality_mode == "iou"
            else None
        )
        points = torch.cat(output["points"])
        strides = torch.cat(output["point_strides"])
        if logits.shape[:2] != (len(targets), len(points)):
            raise ValueError("FCOS predictions do not match target batch/point count")
        device = logits.device
        assignments = []
        total_positive = total_negative = total_ignored = 0
        for target in targets:
            target = target_to_device(target, device)
            assigned = self.assigner(output["points"], output["point_strides"], target)
            assignments.append((target, assigned))
            total_positive += int((assigned["states"] == FCOSPointAssigner.POSITIVE).sum())
            total_negative += int((assigned["states"] == FCOSPointAssigner.NEGATIVE).sum())
            total_ignored += int((assigned["states"] == FCOSPointAssigner.IGNORE).sum())
        normalizer = distributed_normalizer(total_positive, device)

        cls_loss = logits.sum() * 0.0
        box_loss = regression.sum() * 0.0
        auxiliary = centerness_logits if centerness_logits is not None else quality_logits
        assert auxiliary is not None
        centerness_loss = auxiliary.sum() * 0.0
        quality_loss = auxiliary.sum() * 0.0
        for batch_index, (target, assigned) in enumerate(assignments):
            states = assigned["states"]
            matched = assigned["matched_gt_indices"]
            valid = states != FCOSPointAssigner.IGNORE
            positive = states == FCOSPointAssigner.POSITIVE
            cls_target = torch.zeros_like(logits[batch_index])
            if positive.any():
                labels = target["labels"][matched[positive]]
                if torch.any((labels < 0) | (labels >= self.config.num_classes)):
                    raise ValueError("target label is outside configured class range")
                cls_target[positive, labels] = 1.0
            cls_loss = cls_loss + sigmoid_focal_loss(
                logits[batch_index, valid],
                cls_target[valid],
                alpha=self.config.focal_alpha,
                gamma=self.config.focal_gamma,
                reduction="sum",
            )
            if positive.any():
                pred_boxes = _decode_distances(
                    points[positive],
                    regression[batch_index, positive],
                    strides[positive],
                    self.config.fcos_normalize_reg_targets_by_stride,
                )
                gt_boxes = target["boxes"][matched[positive]]
                box_loss = box_loss + (1.0 - generalized_box_iou(pred_boxes, gt_boxes)).sum()
                if centerness_logits is not None:
                    centerness_loss = centerness_loss + F.binary_cross_entropy_with_logits(
                        centerness_logits[batch_index, positive],
                        assigned["centerness_targets"][positive],
                        reduction="sum",
                    )
                if quality_logits is not None:
                    quality_target = aligned_box_iou(pred_boxes.detach(), gt_boxes).clamp(0, 1)
                    quality_loss = quality_loss + F.binary_cross_entropy_with_logits(
                        quality_logits[batch_index, positive],
                        quality_target,
                        reduction="sum",
                    )

        cls_loss = cls_loss / normalizer
        box_loss = box_loss / normalizer
        centerness_loss = centerness_loss / normalizer
        quality_loss = quality_loss / normalizer
        total = (
            cls_loss
            + self.config.box_loss_weight * box_loss
            + self.config.centerness_loss_weight * centerness_loss
            + self.config.quality_loss_weight * quality_loss
        )
        return {
            "loss_total": total,
            "loss_cls": cls_loss,
            "loss_box": box_loss,
            "loss_centerness": centerness_loss,
            "loss_quality": quality_loss,
            "num_positive": torch.tensor(float(total_positive), device=device),
            "num_negative": torch.tensor(float(total_negative), device=device),
            "num_ignored": torch.tensor(float(total_ignored), device=device),
            "num_candidates": torch.tensor(float(len(points) * len(targets)), device=device),
        }
