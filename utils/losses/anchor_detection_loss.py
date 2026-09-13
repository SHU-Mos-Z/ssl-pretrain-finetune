"""RetinaNet-style anchor assignment and focal/regression losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.detection_contracts import DetectionConfig
from models.modules_detection import AnchorMatcher, BoxCoder
from models.modules_detection.box_ops import aligned_box_iou, generalized_box_iou, sigmoid_focal_loss
from .detection_common import distributed_normalizer, target_to_device


class AnchorDetectionLoss(nn.Module):
    def __init__(self, config: DetectionConfig):
        super().__init__()
        self.config = config
        self.matcher = AnchorMatcher(
            config.positive_iou_threshold,
            config.negative_iou_threshold,
            config.ignore_iou_threshold,
            config.matcher_chunk_size,
        )
        self.box_coder = BoxCoder(config.box_coder_weights)

    def forward(self, output: dict, targets: list[dict]) -> dict[str, torch.Tensor]:
        if output.get("detection_mode") != self.config.detection_mode:
            raise ValueError("model output and anchor criterion detection modes differ")
        anchors = torch.cat(output["anchors"], dim=0)
        logits = torch.cat(output["cls_logits"], dim=1).float()
        deltas = torch.cat(output["bbox_deltas"], dim=1).float()
        quality_logits = (
            torch.cat(output["quality_logits"], dim=1).float()
            if self.config.quality_mode == "iou"
            else None
        )
        if logits.shape[:2] != (len(targets), len(anchors)):
            raise ValueError("anchor predictions do not match target batch/anchor count")
        device = logits.device
        assignments = []
        total_positive = total_negative = total_ignored = 0
        for target in targets:
            target = target_to_device(target, device)
            states, matched, _ = self.matcher(anchors, target)
            assignments.append((target, states, matched))
            total_positive += int((states == AnchorMatcher.POSITIVE).sum())
            total_negative += int((states == AnchorMatcher.NEGATIVE).sum())
            total_ignored += int((states == AnchorMatcher.IGNORE).sum())
        normalizer = distributed_normalizer(total_positive, device)

        cls_loss = logits.sum() * 0.0
        box_loss = deltas.sum() * 0.0
        quality_loss = deltas.sum() * 0.0
        for batch_index, (target, states, matched) in enumerate(assignments):
            valid = states != AnchorMatcher.IGNORE
            positive = states == AnchorMatcher.POSITIVE
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
                positive_anchors = anchors[positive]
                gt_boxes = target["boxes"][matched[positive]]
                if self.config.box_loss == "smooth_l1":
                    regression_target = self.box_coder.encode(positive_anchors, gt_boxes)
                    box_loss = box_loss + F.smooth_l1_loss(
                        deltas[batch_index, positive],
                        regression_target,
                        beta=self.config.smooth_l1_beta,
                        reduction="sum",
                    )
                else:
                    decoded = self.box_coder.decode(
                        positive_anchors, deltas[batch_index, positive]
                    )
                    box_loss = box_loss + (1.0 - generalized_box_iou(decoded, gt_boxes)).sum()
                if quality_logits is not None:
                    if self.config.box_loss == "smooth_l1":
                        decoded = self.box_coder.decode(
                            positive_anchors, deltas[batch_index, positive]
                        )
                    quality_target = aligned_box_iou(decoded.detach(), gt_boxes).clamp(0, 1)
                    quality_loss = quality_loss + F.binary_cross_entropy_with_logits(
                        quality_logits[batch_index, positive],
                        quality_target,
                        reduction="sum",
                    )

        cls_loss = cls_loss / normalizer
        box_loss = box_loss / normalizer
        quality_loss = quality_loss / normalizer
        total = (
            cls_loss
            + self.config.box_loss_weight * box_loss
            + self.config.quality_loss_weight * quality_loss
        )
        return {
            "loss_total": total,
            "loss_cls": cls_loss,
            "loss_box": box_loss,
            "loss_quality": quality_loss,
            "num_positive": torch.tensor(float(total_positive), device=device),
            "num_negative": torch.tensor(float(total_negative), device=device),
            "num_ignored": torch.tensor(float(total_ignored), device=device),
            "num_candidates": torch.tensor(float(len(anchors) * len(targets)), device=device),
        }
