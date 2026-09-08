"""FCOS point assignment with center sampling, ranges and exact ignore masks."""

from __future__ import annotations

import torch

from .box_ops import points_inside_boxes, sample_mask_at_points


class FCOSPointAssigner:
    NEGATIVE = 0
    POSITIVE = 1
    IGNORE = -1

    def __init__(
        self,
        regression_ranges: tuple[tuple[float, float], ...],
        center_sampling_radius: float = 1.5,
        normalize_targets_by_stride: bool = True,
    ):
        self.regression_ranges = tuple(tuple(float(v) for v in pair) for pair in regression_ranges)
        self.center_sampling_radius = float(center_sampling_radius)
        self.normalize_targets_by_stride = bool(normalize_targets_by_stride)

    def __call__(
        self,
        points_by_level: list[torch.Tensor],
        strides_by_level: list[torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if len(points_by_level) != len(self.regression_ranges):
            raise ValueError("point levels and regression ranges differ")
        points = torch.cat(points_by_level)
        strides = torch.cat(strides_by_level)
        ranges = torch.cat(
            [
                points.new_tensor(value).view(1, 2).expand(len(level), 2)
                for level, value in zip(points_by_level, self.regression_ranges)
            ],
            dim=0,
        )
        n_points = len(points)
        gt_boxes = target["boxes"]
        states = torch.zeros((n_points,), dtype=torch.int8, device=points.device)
        matched = torch.zeros((n_points,), dtype=torch.long, device=points.device)
        regression = points.new_zeros((n_points, 4))
        centerness = points.new_zeros((n_points,))
        if len(gt_boxes):
            x, y = points[:, 0:1], points[:, 1:2]
            left = x - gt_boxes[None, :, 0]
            top = y - gt_boxes[None, :, 1]
            right = gt_boxes[None, :, 2] - x
            bottom = gt_boxes[None, :, 3] - y
            distances = torch.stack((left, top, right, bottom), dim=2)
            inside = distances.min(dim=2).values >= 0
            max_distance = distances.max(dim=2).values
            in_range = (max_distance >= ranges[:, 0:1]) & (max_distance < ranges[:, 1:2])
            centers = 0.5 * (gt_boxes[:, :2] + gt_boxes[:, 2:])
            radius = self.center_sampling_radius * strides[:, None]
            center_x1 = torch.maximum(gt_boxes[None, :, 0], centers[None, :, 0] - radius)
            center_y1 = torch.maximum(gt_boxes[None, :, 1], centers[None, :, 1] - radius)
            center_x2 = torch.minimum(gt_boxes[None, :, 2], centers[None, :, 0] + radius)
            center_y2 = torch.minimum(gt_boxes[None, :, 3], centers[None, :, 1] + radius)
            in_center = (
                (x >= center_x1) & (y >= center_y1) & (x < center_x2) & (y < center_y2)
            )
            candidates = inside & in_range & in_center
            areas = ((gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]))
            candidate_areas = areas[None, :].expand(n_points, -1).clone()
            candidate_areas[~candidates] = torch.inf
            minimum_area, matched = candidate_areas.min(dim=1)
            positive = torch.isfinite(minimum_area)
            states[positive] = self.POSITIVE
            if positive.any():
                selected = distances[positive, matched[positive]]
                regression[positive] = selected
                lr = selected[:, [0, 2]]
                tb = selected[:, [1, 3]]
                centerness[positive] = torch.sqrt(
                    (lr.min(1).values / lr.max(1).values.clamp(min=1e-7))
                    * (tb.min(1).values / tb.max(1).values.clamp(min=1e-7))
                )
                if self.normalize_targets_by_stride:
                    regression[positive] /= strides[positive, None]

        uncertain = sample_mask_at_points(target["ignore_mask"], points)
        height, width = target["ignore_mask"].shape
        uncertain |= (
            (points[:, 0] < 0)
            | (points[:, 0] >= width)
            | (points[:, 1] < 0)
            | (points[:, 1] >= height)
        )
        uncertain |= points_inside_boxes(points, target["ignore_boxes"]).any(dim=1)
        uncertain |= points_inside_boxes(points, target["crowd_boxes"]).any(dim=1)
        states[uncertain & (states != self.POSITIVE)] = self.IGNORE
        return {
            "states": states,
            "matched_gt_indices": matched,
            "regression_targets": regression,
            "centerness_targets": centerness,
            "points": points,
            "strides": strides,
        }
