"""Pure tensor box geometry used by both detection modes."""

from __future__ import annotations

import torch
from torchvision.ops import batched_nms as _torchvision_batched_nms


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    return wh[:, 0] * wh[:, 1]


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    left_top = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (right_bottom - left_top).clamp(min=0).prod(dim=2)
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp(min=1e-7)


def box_ioa(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Intersection divided by area of ``boxes1``."""

    left_top = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (right_bottom - left_top).clamp(min=0).prod(dim=2)
    return intersection / box_area(boxes1)[:, None].clamp(min=1e-7)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.shape != boxes2.shape or boxes1.ndim != 2 or boxes1.shape[1] != 4:
        raise ValueError("paired GIoU requires matching (N,4) tensors")
    left_top = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    right_bottom = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    intersection = (right_bottom - left_top).clamp(min=0).prod(dim=1)
    area1, area2 = box_area(boxes1), box_area(boxes2)
    union = area1 + area2 - intersection
    iou = intersection / union.clamp(min=1e-7)
    enclosing_lt = torch.minimum(boxes1[:, :2], boxes2[:, :2])
    enclosing_rb = torch.maximum(boxes1[:, 2:], boxes2[:, 2:])
    enclosing = (enclosing_rb - enclosing_lt).clamp(min=0).prod(dim=1)
    return iou - (enclosing - union) / enclosing.clamp(min=1e-7)


def clip_boxes_to_image(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    height, width = image_size
    output = boxes.clone()
    output[..., 0::2] = output[..., 0::2].clamp(min=0, max=width)
    output[..., 1::2] = output[..., 1::2].clamp(min=0, max=height)
    return output


def remove_small_boxes(boxes: torch.Tensor, min_size: float) -> torch.Tensor:
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    return torch.where((widths >= min_size) & (heights >= min_size))[0]


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    if not len(boxes):
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    return _torchvision_batched_nms(boxes.float(), scores.float(), labels, iou_threshold)


def points_inside_boxes(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Return an ``(N_points,N_boxes)`` strict-inside matrix."""

    if not len(boxes):
        return torch.zeros((len(points), 0), dtype=torch.bool, device=points.device)
    x, y = points[:, 0:1], points[:, 1:2]
    return (
        (x >= boxes[None, :, 0])
        & (y >= boxes[None, :, 1])
        & (x < boxes[None, :, 2])
        & (y < boxes[None, :, 3])
    )


def sample_mask_at_points(mask: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Nearest-pixel lookup for input-coordinate reference points."""

    if mask.ndim != 2:
        raise ValueError("ignore mask must have shape (H,W)")
    height, width = mask.shape
    xs = points[:, 0].floor().long().clamp(0, max(width - 1, 0))
    ys = points[:, 1].floor().long().clamp(0, max(height - 1, 0))
    return mask[ys, xs].bool()


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "sum",
) -> torch.Tensor:
    probabilities = logits.sigmoid()
    ce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    loss = ce * (1.0 - p_t).pow(gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    if reduction == "none":
        return loss
    raise ValueError(f"invalid reduction={reduction}")
