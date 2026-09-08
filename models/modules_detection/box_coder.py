"""Anchor delta encoding and decoding."""

from __future__ import annotations

import math

import torch


class BoxCoder:
    def __init__(
        self,
        weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        bbox_xform_clip: float = math.log(1000.0 / 16.0),
    ):
        self.weights = tuple(float(value) for value in weights)
        self.bbox_xform_clip = float(bbox_xform_clip)

    def encode(self, anchors: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        if anchors.shape != gt_boxes.shape:
            raise ValueError("anchors and gt_boxes must have matching shapes")
        anchor_wh = (anchors[:, 2:] - anchors[:, :2]).clamp(min=1e-7)
        anchor_ctr = anchors[:, :2] + 0.5 * anchor_wh
        gt_wh = (gt_boxes[:, 2:] - gt_boxes[:, :2]).clamp(min=1e-7)
        gt_ctr = gt_boxes[:, :2] + 0.5 * gt_wh
        wx, wy, ww, wh = self.weights
        dx = wx * (gt_ctr[:, 0] - anchor_ctr[:, 0]) / anchor_wh[:, 0]
        dy = wy * (gt_ctr[:, 1] - anchor_ctr[:, 1]) / anchor_wh[:, 1]
        dw = ww * torch.log(gt_wh[:, 0] / anchor_wh[:, 0])
        dh = wh * torch.log(gt_wh[:, 1] / anchor_wh[:, 1])
        return torch.stack((dx, dy, dw, dh), dim=1)

    def decode(self, anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        if anchors.shape != deltas.shape:
            raise ValueError("anchors and deltas must have matching shapes")
        anchor_wh = (anchors[:, 2:] - anchors[:, :2]).clamp(min=1e-7)
        anchor_ctr = anchors[:, :2] + 0.5 * anchor_wh
        wx, wy, ww, wh = self.weights
        dx = deltas[:, 0] / wx
        dy = deltas[:, 1] / wy
        dw = (deltas[:, 2] / ww).clamp(max=self.bbox_xform_clip)
        dh = (deltas[:, 3] / wh).clamp(max=self.bbox_xform_clip)
        pred_ctr_x = dx * anchor_wh[:, 0] + anchor_ctr[:, 0]
        pred_ctr_y = dy * anchor_wh[:, 1] + anchor_ctr[:, 1]
        pred_w = dw.exp() * anchor_wh[:, 0]
        pred_h = dh.exp() * anchor_wh[:, 1]
        return torch.stack(
            (
                pred_ctr_x - 0.5 * pred_w,
                pred_ctr_y - 0.5 * pred_h,
                pred_ctr_x + 0.5 * pred_w,
                pred_ctr_y + 0.5 * pred_h,
            ),
            dim=1,
        )
