"""Geometry for runtime detection windows and source-coordinate evaluation.

The preprocessing output remains an immutable source-scene dataset.  This module
contains the reversible geometry used to turn one source scene into model views.
It deliberately has no torch/model dependency so training, evaluation and small
analysis scripts can share exactly the same supervision rules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Any, Iterable, Sequence

import numpy as np


Size2D = tuple[int, int]
Box4 = tuple[float, float, float, float]


def _size2(value: Sequence[int] | int | None, name: str) -> Size2D | None:
    if value is None:
        return None
    if isinstance(value, int):
        output = (int(value), int(value))
    else:
        if len(value) != 2:
            raise ValueError(f"{name} must contain height,width")
        output = (int(value[0]), int(value[1]))
    if min(output) < 1:
        raise ValueError(f"{name} values must be positive")
    return output


@dataclass(frozen=True)
class DetectionViewConfig:
    """Serializable runtime-view contract shared by training and evaluation."""

    view_mode: str = "direct"
    source_crop_size: Size2D | None = None
    model_input_size: Size2D | None = None
    train_views_per_source: int = 1
    positive_guided_fraction: float = 0.5
    eval_stride: Size2D | None = None
    visible_ratio_threshold: float = 0.70
    min_visible_side: float = 0.0
    enable_crop_truncated_positive: bool = True
    ownership_filter: bool = True
    global_nms_threshold: float = 0.5
    seed: int = 42

    def validate(self) -> None:
        if self.view_mode not in {"direct", "runtime_window"}:
            raise ValueError("view_mode must be direct or runtime_window")
        crop = _size2(self.source_crop_size, "source_crop_size")
        output = _size2(self.model_input_size, "model_input_size")
        stride = _size2(self.eval_stride, "eval_stride")
        if self.view_mode == "runtime_window":
            if crop is None or output is None or stride is None:
                raise ValueError(
                    "runtime_window requires source_crop_size, model_input_size and eval_stride"
                )
            if stride[0] > crop[0] or stride[1] > crop[1]:
                raise ValueError("eval_stride cannot exceed source_crop_size (coverage gap)")
        if self.train_views_per_source < 1:
            raise ValueError("train_views_per_source must be positive")
        if not 0.0 <= self.positive_guided_fraction <= 1.0:
            raise ValueError("positive_guided_fraction must be in [0,1]")
        if not 0.0 <= self.visible_ratio_threshold <= 1.0:
            raise ValueError("visible_ratio_threshold must be in [0,1]")
        if self.min_visible_side < 0:
            raise ValueError("min_visible_side must be non-negative")
        if not 0.0 <= self.global_nms_threshold <= 1.0:
            raise ValueError("global_nms_threshold must be in [0,1]")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("source_crop_size", "model_input_size", "eval_stride"):
            if payload[key] is not None:
                payload[key] = list(payload[key])
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DetectionViewConfig":
        values = dict(payload)
        for key in ("source_crop_size", "model_input_size", "eval_stride"):
            if values.get(key) is not None:
                values[key] = _size2(values[key], key)
        config = cls(**values)
        config.validate()
        return config


@dataclass(frozen=True)
class DetectionView:
    view_id: int
    source_image_id: int
    source_stem: str
    crop_xyxy: tuple[int, int, int, int]
    output_size: Size2D
    ownership_xyxy: Box4

    @property
    def crop_size(self) -> Size2D:
        x1, y1, x2, y2 = self.crop_xyxy
        return y2 - y1, x2 - x1

    @property
    def scale_xy(self) -> tuple[float, float]:
        crop_h, crop_w = self.crop_size
        return self.output_size[1] / crop_w, self.output_size[0] / crop_h

    def to_dict(self) -> dict[str, Any]:
        scale_x, scale_y = self.scale_xy
        return {
            "view_id": self.view_id,
            "source_image_id": self.source_image_id,
            "source_stem": self.source_stem,
            "crop_xyxy": list(self.crop_xyxy),
            "output_size": list(self.output_size),
            "scale_x": scale_x,
            "scale_y": scale_y,
            "ownership_xyxy": list(self.ownership_xyxy),
        }


def xywh_to_xyxy(bbox: Sequence[float]) -> Box4:
    x, y, width, height = (float(value) for value in bbox)
    return x, y, x + width, y + height


def box_intersection(box: Sequence[float], crop: Sequence[float]) -> Box4 | None:
    x1 = max(float(box[0]), float(crop[0]))
    y1 = max(float(box[1]), float(crop[1]))
    x2 = min(float(box[2]), float(crop[2]))
    y2 = min(float(box[3]), float(crop[3]))
    return None if x2 <= x1 or y2 <= y1 else (x1, y1, x2, y2)


def box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def project_source_box(box: Sequence[float], view: DetectionView) -> Box4:
    crop_x1, crop_y1, _, _ = view.crop_xyxy
    scale_x, scale_y = view.scale_xy
    return (
        (float(box[0]) - crop_x1) * scale_x,
        (float(box[1]) - crop_y1) * scale_y,
        (float(box[2]) - crop_x1) * scale_x,
        (float(box[3]) - crop_y1) * scale_y,
    )


def inverse_project_boxes(boxes: np.ndarray, view: DetectionView) -> np.ndarray:
    output = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    if not len(output):
        return output
    crop_x1, crop_y1, _, _ = view.crop_xyxy
    scale_x, scale_y = view.scale_xy
    output[:, 0::2] = output[:, 0::2] / scale_x + crop_x1
    output[:, 1::2] = output[:, 1::2] / scale_y + crop_y1
    return output


def _axis_origins(length: int, window: int, stride: int) -> list[int]:
    if window > length:
        raise ValueError(f"window={window} exceeds source axis={length}")
    if window == length:
        return [0]
    origins = list(range(0, length - window + 1, stride))
    final = length - window
    if origins[-1] != final:
        origins.append(final)
    return origins


def _axis_ownership(origins: Sequence[int], window: int, length: int) -> list[tuple[float, float]]:
    centers = [origin + 0.5 * window for origin in origins]
    boundaries = [0.0]
    boundaries.extend(0.5 * (left + right) for left, right in zip(centers[:-1], centers[1:]))
    boundaries.append(float(length))
    return list(zip(boundaries[:-1], boundaries[1:]))


def build_evaluation_views(
    *,
    source_image_id: int,
    source_stem: str,
    source_size: Size2D,
    config: DetectionViewConfig,
    first_view_id: int = 0,
) -> list[DetectionView]:
    config.validate()
    height, width = source_size
    if config.view_mode == "direct":
        return [
            DetectionView(
                first_view_id,
                source_image_id,
                source_stem,
                (0, 0, width, height),
                (height, width),
                (0.0, 0.0, float(width), float(height)),
            )
        ]
    assert config.source_crop_size is not None
    assert config.model_input_size is not None
    assert config.eval_stride is not None
    crop_h, crop_w = config.source_crop_size
    stride_h, stride_w = config.eval_stride
    ys = _axis_origins(height, crop_h, stride_h)
    xs = _axis_origins(width, crop_w, stride_w)
    y_ownership = _axis_ownership(ys, crop_h, height)
    x_ownership = _axis_ownership(xs, crop_w, width)
    views: list[DetectionView] = []
    view_id = first_view_id
    for y_index, y in enumerate(ys):
        for x_index, x in enumerate(xs):
            own_x1, own_x2 = x_ownership[x_index]
            own_y1, own_y2 = y_ownership[y_index]
            views.append(
                DetectionView(
                    view_id,
                    source_image_id,
                    source_stem,
                    (x, y, x + crop_w, y + crop_h),
                    config.model_input_size,
                    (own_x1, own_y1, own_x2, own_y2),
                )
            )
            view_id += 1
    return views


def sample_training_view(
    *,
    source_image_id: int,
    source_stem: str,
    source_size: Size2D,
    annotations: Sequence[dict[str, Any]],
    config: DetectionViewConfig,
    epoch: int,
    source_index: int,
    view_slot: int,
) -> DetectionView:
    config.validate()
    height, width = source_size
    if config.view_mode == "direct":
        return build_evaluation_views(
            source_image_id=source_image_id,
            source_stem=source_stem,
            source_size=source_size,
            config=config,
            first_view_id=source_index,
        )[0]
    assert config.source_crop_size is not None
    assert config.model_input_size is not None
    crop_h, crop_w = config.source_crop_size
    if crop_h > height or crop_w > width:
        raise ValueError(
            f"runtime crop {config.source_crop_size} exceeds {source_stem} size {source_size}"
        )
    generator = random.Random(
        config.seed
        + 1_000_003 * int(epoch)
        + 10_007 * int(source_index)
        + 101 * int(view_slot)
    )
    ordinary = [
        item
        for item in annotations
        if not bool(item.get("ignore", 0)) and not bool(item.get("iscrowd", 0))
    ]
    guided = bool(ordinary) and generator.random() < config.positive_guided_fraction
    if guided:
        chosen = generator.choice(ordinary)
        x1, y1, x2, y2 = xywh_to_xyxy(chosen["bbox"])

        def choose_origin(low: float, high: float, axis_length: int, window: int, center: float) -> int:
            minimum = max(0, int(math.ceil(low)))
            maximum = min(axis_length - window, int(math.floor(high)))
            if minimum <= maximum:
                return generator.randint(minimum, maximum)
            centered = int(round(center - 0.5 * window))
            return min(max(centered, 0), axis_length - window)

        x = choose_origin(x2 - crop_w, x1, width, crop_w, 0.5 * (x1 + x2))
        y = choose_origin(y2 - crop_h, y1, height, crop_h, 0.5 * (y1 + y2))
    else:
        x = generator.randint(0, width - crop_w)
        y = generator.randint(0, height - crop_h)
    view_id = source_index * config.train_views_per_source + view_slot
    return DetectionView(
        view_id,
        source_image_id,
        source_stem,
        (x, y, x + crop_w, y + crop_h),
        config.model_input_size,
        (float(x), float(y), float(x + crop_w), float(y + crop_h)),
    )


def project_annotations_to_view(
    annotations: Iterable[dict[str, Any]],
    view: DetectionView,
    config: DetectionViewConfig,
) -> dict[str, list[dict[str, Any]]]:
    """Apply the agreed positive/source-truncated/crop-truncated policy."""

    positive: list[dict[str, Any]] = []
    crowd: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    crop = view.crop_xyxy
    scale_x, scale_y = view.scale_xy
    for annotation in annotations:
        bbox = annotation.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"invalid bbox for annotation_id={annotation.get('id')}")
        numeric_bbox = np.asarray(bbox, dtype=np.float64)
        if (
            not np.isfinite(numeric_bbox).all()
            or float(numeric_bbox[2]) <= 0
            or float(numeric_bbox[3]) <= 0
        ):
            raise ValueError(f"invalid bbox for annotation_id={annotation.get('id')}")
        source_box = xywh_to_xyxy(bbox)
        intersection = box_intersection(source_box, crop)
        if intersection is None:
            continue
        item = dict(annotation)
        item["source_box_xyxy"] = source_box
        item["visible_source_box_xyxy"] = intersection
        item["box_xyxy"] = project_source_box(intersection, view)
        if bool(annotation.get("ignore", 0)):
            item["runtime_decision"] = "source_ignore"
            ignored.append(item)
            continue
        if bool(annotation.get("iscrowd", 0)):
            item["runtime_decision"] = "source_crowd"
            crowd.append(item)
            continue
        original_area = max(box_area(source_box), 1e-12)
        visible_ratio = box_area(intersection) / original_area
        fully_visible = visible_ratio >= 1.0 - 1e-7
        source_truncated = bool(annotation.get("truncated", 0))
        item["visible_ratio"] = visible_ratio
        item["source_truncated"] = source_truncated
        item["crop_truncated"] = not fully_visible
        if source_truncated:
            if fully_visible:
                item["runtime_decision"] = "positive_source_truncated"
                positive.append(item)
            else:
                item["runtime_decision"] = "ignore_source_and_crop_truncated"
                ignored.append(item)
            continue
        if fully_visible:
            item["runtime_decision"] = "positive_complete"
            positive.append(item)
            continue
        center_x = 0.5 * (source_box[0] + source_box[2])
        center_y = 0.5 * (source_box[1] + source_box[3])
        center_inside = (
            crop[0] <= center_x < crop[2] and crop[1] <= center_y < crop[3]
        )
        visible_width = (intersection[2] - intersection[0]) * scale_x
        visible_height = (intersection[3] - intersection[1]) * scale_y
        trainable_partial = (
            config.enable_crop_truncated_positive
            and visible_ratio >= config.visible_ratio_threshold
            and center_inside
            and visible_width >= config.min_visible_side
            and visible_height >= config.min_visible_side
        )
        if trainable_partial:
            item["runtime_decision"] = "positive_crop_truncated"
            positive.append(item)
        else:
            item["runtime_decision"] = "ignore_crop_truncated"
            ignored.append(item)
    return {"positive": positive, "crowd": crowd, "ignored": ignored}


def centers_inside_ownership(boxes: np.ndarray, ownership: Sequence[float]) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if not len(boxes):
        return np.empty((0,), dtype=bool)
    centers = 0.5 * (boxes[:, :2] + boxes[:, 2:])
    x1, y1, x2, y2 = (float(value) for value in ownership)
    # Half-open cells partition the source; the source's right/bottom boundary is
    # still covered because valid detection centers are strictly inside it.
    return (
        (centers[:, 0] >= x1)
        & (centers[:, 0] < x2)
        & (centers[:, 1] >= y1)
        & (centers[:, 1] < y2)
    )
