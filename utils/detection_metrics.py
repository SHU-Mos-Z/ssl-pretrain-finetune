"""COCO bbox evaluation with the project's custom-ignore convention."""

from __future__ import annotations

import copy
import contextlib
import csv
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _xywh_to_xyxy(box: list[float]) -> np.ndarray:
    x, y, width, height = (float(value) for value in box)
    return np.asarray([x, y, x + width, y + height], dtype=np.float32)


def _iou_one(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.empty((0,), dtype=np.float32)
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(bottom_right - top_left, 0).prod(axis=1)
    area1 = np.maximum(box[2:] - box[:2], 0).prod()
    area2 = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
    return intersection / np.maximum(area1 + area2 - intersection, 1e-7)


def _ioa_one(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.empty((0,), dtype=np.float32)
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(bottom_right - top_left, 0).prod(axis=1)
    area = np.maximum(box[2:] - box[:2], 0).prod()
    return intersection / max(float(area), 1e-7)


def per_image_detection_metrics(
    coco_payload: dict[str, Any], records: list[dict[str, Any]], iou_threshold: float = 0.5
) -> list[dict[str, Any]]:
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    predictions_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in coco_payload.get("annotations", []):
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    for prediction in records:
        predictions_by_image.setdefault(int(prediction["image_id"]), []).append(prediction)
    rows = []
    for image in coco_payload.get("images", []):
        image_id = int(image["id"])
        annotations = annotations_by_image.get(image_id, [])
        ordinary = [
            item for item in annotations if not item.get("ignore", 0) and not item.get("iscrowd", 0)
        ]
        uncertain = [
            item for item in annotations if item.get("ignore", 0) or item.get("iscrowd", 0)
        ]
        matched: set[int] = set()
        tp = fp = ignored_predictions = 0
        predictions = sorted(
            predictions_by_image.get(image_id, []), key=lambda item: float(item["score"]), reverse=True
        )
        for prediction in predictions:
            box = _xywh_to_xyxy(prediction["bbox"])
            eligible_indices = [
                index
                for index, target in enumerate(ordinary)
                if index not in matched and int(target["category_id"]) == int(prediction["category_id"])
            ]
            eligible_boxes = np.asarray(
                [_xywh_to_xyxy(ordinary[index]["bbox"]) for index in eligible_indices],
                dtype=np.float32,
            ).reshape(-1, 4)
            ious = _iou_one(box, eligible_boxes)
            if len(ious) and float(ious.max()) >= iou_threshold:
                matched.add(eligible_indices[int(ious.argmax())])
                tp += 1
                continue
            uncertain_boxes = np.asarray(
                [_xywh_to_xyxy(item["bbox"]) for item in uncertain], dtype=np.float32
            ).reshape(-1, 4)
            if len(uncertain_boxes) and float(_ioa_one(box, uncertain_boxes).max()) >= iou_threshold:
                ignored_predictions += 1
            else:
                fp += 1
        fn = len(ordinary) - len(matched)
        rows.append(
            {
                "image_id": image_id,
                "file_name": image.get("file_name", ""),
                "num_gt": len(ordinary),
                "num_predictions": len(predictions),
                "tp_iou50": tp,
                "fp_iou50": fp,
                "fn_iou50": fn,
                "ignored_predictions": ignored_predictions,
                "precision_iou50": tp / max(tp + fp, 1),
                "recall_iou50": tp / max(tp + fn, 1),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


COCO_STAT_NAMES = (
    "AP50_95",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
)


def _ground_truth_for_eval(coco_payload: dict[str, Any]) -> dict[str, Any]:
    """Convert custom ``ignore=1`` boxes to standard COCO crowd-ignore boxes."""

    payload = copy.deepcopy(coco_payload)
    for annotation in payload.get("annotations", []):
        if bool(annotation.get("ignore", 0)):
            annotation["ignore"] = 1
            annotation["iscrowd"] = 1
    return payload


def _make_coco(payload: dict[str, Any]) -> COCO:
    payload.setdefault("info", {})
    payload.setdefault("licenses", [])
    coco = COCO()
    coco.dataset = payload
    coco.createIndex()
    return coco


def detections_to_coco(
    detections: list[dict[str, Any]], label_to_category_id: dict[int, int]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for detection in detections:
        image_id = int(detection["image_id"])
        boxes = np.asarray(detection["boxes"], dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(detection["scores"], dtype=np.float32).reshape(-1)
        labels = np.asarray(detection["labels"], dtype=np.int64).reshape(-1)
        if not (len(boxes) == len(scores) == len(labels)):
            raise ValueError(f"inconsistent prediction lengths for image_id={image_id}")
        for box, score, label in zip(boxes, scores, labels):
            if int(label) not in label_to_category_id:
                raise ValueError(f"unknown internal prediction label={int(label)}")
            x1, y1, x2, y2 = (float(value) for value in box)
            records.append(
                {
                    "image_id": image_id,
                    "category_id": int(label_to_category_id[int(label)]),
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": float(score),
                }
            )
    return records


def _run_coco_eval(
    coco_gt: COCO,
    prediction_records: list[dict[str, Any]],
    image_ids: list[int],
    category_ids: list[int] | None = None,
    summarize: bool = True,
) -> COCOeval | None:
    if not prediction_records:
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(prediction_records)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.imgIds = list(image_ids)
    if category_ids is not None:
        evaluator.params.catIds = list(category_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    if summarize:
        evaluator.summarize()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            evaluator.summarize()
    return evaluator


def save_coco_precision_recall_curve(
    evaluator: COCOeval | None,
    output_path: str | Path,
) -> None:
    """Save macro COCO PR curves at IoU=.50, .75 and averaged .50:.95."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if evaluator is None:
        recall = np.linspace(0.0, 1.0, 101, dtype=np.float64)
        curves = {
            "precision_iou50": np.zeros_like(recall),
            "precision_iou75": np.zeros_like(recall),
            "precision_iou50_95": np.zeros_like(recall),
        }
    else:
        recall = np.asarray(evaluator.params.recThrs, dtype=np.float64)
        precision = np.asarray(evaluator.eval["precision"], dtype=np.float64)
        # COCO precision dimensions: [IoU, recall, category, area, maxDet].
        all_area_max100 = precision[:, :, :, 0, -1]

        def macro_curve(iou_indices: np.ndarray) -> np.ndarray:
            selected = all_area_max100[iou_indices]
            valid = selected >= 0
            numerator = np.where(valid, selected, 0.0).sum(axis=(0, 2))
            denominator = valid.sum(axis=(0, 2))
            return np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator, dtype=np.float64),
                where=denominator > 0,
            )

        iou_thresholds = np.asarray(evaluator.params.iouThrs)
        index50 = np.flatnonzero(np.isclose(iou_thresholds, 0.50))
        index75 = np.flatnonzero(np.isclose(iou_thresholds, 0.75))
        curves = {
            "precision_iou50": macro_curve(index50),
            "precision_iou75": macro_curve(index75),
            "precision_iou50_95": macro_curve(np.arange(len(iou_thresholds))),
        }

    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["recall", *curves])
        for index, recall_value in enumerate(recall):
            writer.writerow(
                [float(recall_value), *[float(values[index]) for values in curves.values()]]
            )

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    styles = {
        "precision_iou50": ("IoU=0.50", "-"),
        "precision_iou75": ("IoU=0.75", "--"),
        "precision_iou50_95": ("IoU=0.50:0.95 mean", "-."),
    }
    for key, values in curves.items():
        label, linestyle = styles[key]
        ax.plot(
            recall,
            values,
            linestyle=linestyle,
            linewidth=2.0,
            label=f"{label} (mean precision={float(np.mean(values)):.3f})",
        )
    ax.set(
        xlabel="Recall",
        ylabel="Interpolated precision",
        title="Validation COCO Bounding-box Precision–Recall",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.02),
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def evaluate_coco_detections(
    coco_payload: dict[str, Any],
    detections: list[dict[str, Any]],
    label_to_category_id: dict[int, int],
    *,
    output_json: str | Path | None = None,
    pr_curve_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate all images, including images with no predictions."""

    image_ids = sorted(int(image["id"]) for image in coco_payload.get("images", []))
    predicted_ids = [int(item["image_id"]) for item in detections]
    if len(predicted_ids) != len(set(predicted_ids)):
        raise ValueError("detection payload contains duplicate image IDs")
    missing = set(image_ids).difference(predicted_ids)
    if missing:
        raise ValueError(f"predictions missing {len(missing)} evaluation images")
    records = detections_to_coco(detections, label_to_category_id)
    if output_json is not None:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    coco_gt = _make_coco(_ground_truth_for_eval(coco_payload))
    evaluator = _run_coco_eval(coco_gt, records, image_ids)
    if pr_curve_path is not None:
        save_coco_precision_recall_curve(evaluator, pr_curve_path)
    if evaluator is None:
        metrics = {name: 0.0 for name in COCO_STAT_NAMES}
    else:
        metrics = {name: float(value) for name, value in zip(COCO_STAT_NAMES, evaluator.stats)}
    metrics["num_images"] = len(image_ids)
    metrics["num_detections"] = len(records)
    ordinary_counts: dict[int, int] = {image_id: 0 for image_id in image_ids}
    num_ignore_gt = 0
    for annotation in coco_payload.get("annotations", []):
        if annotation.get("ignore", 0) or annotation.get("iscrowd", 0):
            num_ignore_gt += 1
        else:
            ordinary_counts[int(annotation["image_id"])] += 1
    metrics["num_gt"] = sum(ordinary_counts.values())
    metrics["num_ignore_gt"] = num_ignore_gt
    metrics["num_empty_gt_images"] = sum(value == 0 for value in ordinary_counts.values())

    per_class: dict[str, dict[str, float]] = {}
    categories = {int(item["id"]): str(item["name"]) for item in coco_payload["categories"]}
    for category_id in sorted(categories):
        class_records = [item for item in records if item["category_id"] == category_id]
        class_eval = _run_coco_eval(
            coco_gt, class_records, image_ids, [category_id], summarize=False
        )
        values = (
            {name: 0.0 for name in COCO_STAT_NAMES}
            if class_eval is None
            else {name: float(value) for name, value in zip(COCO_STAT_NAMES, class_eval.stats)}
        )
        per_class[categories[category_id]] = values
    metrics["per_class"] = per_class
    if output_json is not None:
        output_dir = Path(output_json).parent
        class_rows = [
            {"category": category, **values} for category, values in per_class.items()
        ]
        _write_csv(output_dir / "per_class_metrics.csv", class_rows)
        _write_csv(
            output_dir / "per_image_metrics.csv",
            per_image_detection_metrics(coco_payload, records),
        )
    return metrics
