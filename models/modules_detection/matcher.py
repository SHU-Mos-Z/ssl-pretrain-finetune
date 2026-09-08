"""Anchor-to-GT assignment with explicit custom-ignore handling."""

from __future__ import annotations

import torch

from .box_ops import box_iou, box_ioa, sample_mask_at_points


class AnchorMatcher:
    NEGATIVE = 0
    POSITIVE = 1
    IGNORE = -1

    def __init__(
        self,
        positive_iou_threshold: float,
        negative_iou_threshold: float,
        ignore_iou_threshold: float,
        chunk_size: int = 65536,
    ):
        self.positive_threshold = float(positive_iou_threshold)
        self.negative_threshold = float(negative_iou_threshold)
        self.ignore_threshold = float(ignore_iou_threshold)
        self.chunk_size = int(chunk_size)

    def _match_iou(self, anchors: torch.Tensor, boxes: torch.Tensor):
        max_iou = torch.zeros((len(anchors),), device=anchors.device)
        matched = torch.zeros((len(anchors),), dtype=torch.long, device=anchors.device)
        if not len(boxes):
            return max_iou, matched
        for start in range(0, len(anchors), self.chunk_size):
            values = box_iou(anchors[start : start + self.chunk_size], boxes)
            chunk_iou, chunk_matched = values.max(dim=1)
            max_iou[start : start + len(values)] = chunk_iou
            matched[start : start + len(values)] = chunk_matched
        return max_iou, matched

    def _ignore_overlap(self, anchors: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        output = torch.zeros((len(anchors),), dtype=torch.bool, device=anchors.device)
        if not len(boxes):
            return output
        for start in range(0, len(anchors), self.chunk_size):
            values = box_ioa(anchors[start : start + self.chunk_size], boxes)
            output[start : start + len(values)] = values.max(dim=1).values >= self.ignore_threshold
        return output

    def __call__(self, anchors: torch.Tensor, target: dict[str, torch.Tensor]):
        gt_boxes = target["boxes"]
        max_iou, matched = self._match_iou(anchors, gt_boxes)
        states = torch.full((len(anchors),), self.IGNORE, dtype=torch.int8, device=anchors.device)
        states[max_iou < self.negative_threshold] = self.NEGATIVE
        states[max_iou >= self.positive_threshold] = self.POSITIVE
        if len(gt_boxes):
            # Guarantee that every ordinary GT owns at least one anchor.
            best_anchor_for_gt = torch.empty((len(gt_boxes),), dtype=torch.long, device=anchors.device)
            claimed = torch.zeros((len(anchors),), dtype=torch.bool, device=anchors.device)
            for gt_index in range(len(gt_boxes)):
                best_value, best_index = -1.0, 0
                for start in range(0, len(anchors), self.chunk_size):
                    values = box_iou(anchors[start : start + self.chunk_size], gt_boxes[gt_index : gt_index + 1]).squeeze(1)
                    values = values.masked_fill(claimed[start : start + len(values)], -1.0)
                    value, local_index = values.max(dim=0)
                    if float(value) > best_value:
                        best_value = float(value)
                        best_index = start + int(local_index)
                best_anchor_for_gt[gt_index] = best_index
                claimed[best_index] = True
            states[best_anchor_for_gt] = self.POSITIVE
            matched[best_anchor_for_gt] = torch.arange(len(gt_boxes), device=anchors.device)

        centers = 0.5 * (anchors[:, :2] + anchors[:, 2:])
        custom_ignore = sample_mask_at_points(target["ignore_mask"], centers)
        custom_ignore |= self._ignore_overlap(anchors, target["ignore_boxes"])
        custom_ignore |= self._ignore_overlap(anchors, target["crowd_boxes"])
        # Reliable ordinary positives take precedence over uncertain regions.
        states[custom_ignore & (states != self.POSITIVE)] = self.IGNORE
        return states, matched, max_iou
