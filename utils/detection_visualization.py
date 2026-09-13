"""Pseudo-RGB visualization of detection GT, predictions and ignore regions."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils.preprocessing.offline_nmf import load_intensity_cube


def _pseudo_rgb(cube: np.ndarray) -> np.ndarray:
    bands = np.linspace(0, cube.shape[0] - 1, 3).round().astype(int)
    rgb = np.stack((cube[bands[2]], cube[bands[1]], cube[bands[0]]), axis=2)
    output = np.empty_like(rgb, dtype=np.uint8)
    for channel in range(3):
        values = rgb[..., channel]
        lower, upper = np.percentile(values, (1, 99))
        scaled = np.clip((values - lower) / max(float(upper - lower), 1e-7), 0, 1)
        output[..., channel] = (scaled * 255).astype(np.uint8)
    return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)


def _xywh_to_xyxy(box) -> np.ndarray:
    x, y, width, height = (float(value) for value in box)
    return np.asarray([x, y, x + width, y + height], dtype=np.float32)


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.empty((0,), dtype=np.float32)
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(bottom_right - top_left, 0).prod(axis=1)
    a = np.maximum(box[2:] - box[:2], 0).prod()
    b = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
    return intersection / np.maximum(a + b - intersection, 1e-7)


def _draw_box(image, box, color, text, thickness=2):
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if text:
        cv2.putText(
            image,
            text,
            (x1, max(12, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )


def render_detection_examples(
    data_root: str | Path,
    coco_payload: dict[str, Any],
    prediction_records: list[dict[str, Any]],
    output_dir: str | Path,
    max_samples: int = 12,
    prediction_score_threshold: float = 0.0,
    max_predictions_per_image: int | None = None,
) -> None:
    if max_samples <= 0:
        return
    root, output = Path(data_root), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    gt_by_image: dict[int, list[dict]] = defaultdict(list)
    pred_by_image: dict[int, list[dict]] = defaultdict(list)
    for item in coco_payload.get("annotations", []):
        gt_by_image[int(item["image_id"])].append(item)
    for item in prediction_records:
        pred_by_image[int(item["image_id"])].append(item)
    images = sorted(
        coco_payload.get("images", []),
        key=lambda item: len(gt_by_image[int(item["id"])]),
        reverse=True,
    )[:max_samples]
    for info in images:
        image_id = int(info["id"])
        canvas = _pseudo_rgb(load_intensity_cube(root / info["file_name"]))
        ignore_name = info.get("ignore_mask_file_name")
        if ignore_name and (root / ignore_name).is_file():
            mask = np.load(root / ignore_name, allow_pickle=False).astype(bool)
            overlay = canvas.copy()
            overlay[mask] = (180, 60, 180)
            canvas = cv2.addWeighted(canvas, 0.75, overlay, 0.25, 0)
        ordinary = [
            item for item in gt_by_image[image_id] if not item.get("ignore", 0) and not item.get("iscrowd", 0)
        ]
        uncertain = [
            item for item in gt_by_image[image_id] if item.get("ignore", 0) or item.get("iscrowd", 0)
        ]
        ordinary_boxes = np.asarray([_xywh_to_xyxy(item["bbox"]) for item in ordinary], dtype=np.float32).reshape(-1, 4)
        matched: set[int] = set()
        predictions = sorted(
            (
                item
                for item in pred_by_image[image_id]
                if float(item["score"]) >= float(prediction_score_threshold)
            ),
            key=lambda item: item["score"],
            reverse=True,
        )
        if max_predictions_per_image is not None:
            predictions = predictions[: int(max_predictions_per_image)]
        for prediction in predictions:
            box = _xywh_to_xyxy(prediction["bbox"])
            ious = _iou(box, ordinary_boxes)
            candidate = int(ious.argmax()) if len(ious) else -1
            is_tp = (
                candidate >= 0
                and float(ious[candidate]) >= 0.5
                and candidate not in matched
                and int(ordinary[candidate]["category_id"]) == int(prediction["category_id"])
            )
            if is_tp:
                matched.add(candidate)
            color = (255, 180, 0) if is_tp else (0, 0, 255)
            _draw_box(canvas, box, color, f"{'TP' if is_tp else 'FP'} {prediction['score']:.2f}")
        for index, (annotation, box) in enumerate(zip(ordinary, ordinary_boxes)):
            color = (0, 200, 0) if index in matched else (0, 220, 255)
            _draw_box(canvas, box, color, "GT" if index in matched else "FN", 2)
        for annotation in uncertain:
            _draw_box(canvas, _xywh_to_xyxy(annotation["bbox"]), (180, 60, 180), "IGNORE", 1)
        stem = Path(info["file_name"]).stem
        cv2.imwrite(str(output / f"{image_id:06d}_{stem}.png"), canvas)
