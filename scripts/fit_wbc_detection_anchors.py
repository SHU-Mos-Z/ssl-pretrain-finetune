#!/usr/bin/env python3
"""Fit RetinaNet anchor shapes to trainable WBC runtime-window boxes.

Only training annotations are read.  The script simulates deterministic runtime
views with the same geometry/supervision contract used by fine-tuning and writes
a JSON that can be passed through ``--anchor-config-json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.datasets.conditioned_detection_dataset import read_coco_annotation_source
from utils.datasets.detection_view_geometry import (
    DetectionViewConfig,
    project_annotations_to_view,
    sample_training_view,
)


def size_pair(text: str) -> tuple[int, int]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if len(values) == 1:
        values = values * 2
    if len(values) != 2 or min(values) < 1:
        raise argparse.ArgumentTypeError("size must be N or height,width")
    return values


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--annotation", default="annotations")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-crop-size", type=size_pair, default=(640, 640))
    parser.add_argument("--model-input-size", type=size_pair, default=(512, 512))
    parser.add_argument("--views-per-source", type=int, default=4)
    parser.add_argument("--simulation-epochs", type=int, default=10)
    parser.add_argument("--positive-guided-fraction", type=float, default=0.5)
    parser.add_argument("--visible-ratio", type=float, default=0.7)
    parser.add_argument("--min-visible-side", type=float, default=32.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def kmeans_1d_log(values: np.ndarray, clusters: int = 4) -> np.ndarray:
    values = np.log(np.maximum(values.astype(np.float64), 1e-6))
    centers = np.quantile(values, np.linspace(0.1, 0.9, clusters))
    for _ in range(100):
        assignment = np.abs(values[:, None] - centers[None, :]).argmin(axis=1)
        updated = np.asarray(
            [values[assignment == index].mean() if np.any(assignment == index) else centers[index]
             for index in range(clusters)]
        )
        if np.allclose(updated, centers, atol=1e-7):
            break
        centers = updated
    return np.sort(np.exp(centers))


def centered_shape_iou(widths: np.ndarray, heights: np.ndarray, templates: np.ndarray) -> np.ndarray:
    intersection = np.minimum(widths[:, None], templates[None, :, 0]) * np.minimum(
        heights[:, None], templates[None, :, 1]
    )
    union = widths[:, None] * heights[:, None] + templates[None, :, 0] * templates[None, :, 1] - intersection
    return intersection / np.maximum(union, 1e-12)


def main() -> None:
    args = get_args()
    if args.views_per_source < 1 or args.simulation_epochs < 1:
        raise ValueError("views-per-source and simulation-epochs must be positive")
    root = Path(args.data_root).expanduser().resolve()
    annotation_path = Path(args.annotation).expanduser()
    if not annotation_path.is_absolute():
        annotation_path = root / annotation_path
    coco = read_coco_annotation_source(annotation_path.resolve())
    by_image: dict[int, list[dict]] = {}
    for annotation in coco.get("annotations", []):
        by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    config = DetectionViewConfig(
        view_mode="runtime_window",
        source_crop_size=args.source_crop_size,
        model_input_size=args.model_input_size,
        train_views_per_source=args.views_per_source,
        positive_guided_fraction=args.positive_guided_fraction,
        eval_stride=args.source_crop_size,
        visible_ratio_threshold=args.visible_ratio,
        min_visible_side=args.min_visible_side,
        seed=args.seed,
    )
    widths: list[float] = []
    heights: list[float] = []
    decision_counts: dict[str, int] = {}
    images = sorted(coco.get("images", []), key=lambda item: int(item["id"]))
    for epoch in range(1, args.simulation_epochs + 1):
        for source_index, image in enumerate(images):
            source_id = int(image["id"])
            annotations = by_image.get(source_id, [])
            source_size = (int(image["height"]), int(image["width"]))
            stem = str(image.get("source_stem", Path(str(image["file_name"])).stem))
            for view_slot in range(args.views_per_source):
                view = sample_training_view(
                    source_image_id=source_id,
                    source_stem=stem,
                    source_size=source_size,
                    annotations=annotations,
                    config=config,
                    epoch=epoch,
                    source_index=source_index,
                    view_slot=view_slot,
                )
                projected = project_annotations_to_view(annotations, view, config)
                for group in projected.values():
                    for item in group:
                        decision = str(item["runtime_decision"])
                        decision_counts[decision] = decision_counts.get(decision, 0) + 1
                for item in projected["positive"]:
                    x1, y1, x2, y2 = item["box_xyxy"]
                    widths.append(x2 - x1)
                    heights.append(y2 - y1)
    if not widths:
        raise RuntimeError("simulation produced no trainable boxes")
    widths_array = np.asarray(widths, dtype=np.float64)
    heights_array = np.asarray(heights, dtype=np.float64)
    sizes = kmeans_1d_log(np.sqrt(widths_array * heights_array), 4)
    ratios_observed = widths_array / np.maximum(heights_array, 1e-9)
    ratios = np.quantile(ratios_observed, (0.15, 0.5, 0.85))
    scales = np.asarray((0.8, 1.0, 1.25), dtype=np.float64)
    template_shapes = []
    for size in sizes:
        for scale in scales:
            for ratio in ratios:
                template_shapes.append(
                    (size * scale * np.sqrt(ratio), size * scale / np.sqrt(ratio))
                )
    best_shape_iou = centered_shape_iou(
        widths_array, heights_array, np.asarray(template_shapes)
    ).max(axis=1)
    quantiles = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    payload = {
        "data_root": str(root),
        "annotation": str(annotation_path.resolve()),
        "view_config": config.to_dict(),
        "simulation_epochs": args.simulation_epochs,
        "num_trainable_box_observations": len(widths),
        "runtime_decision_counts": decision_counts,
        "box_width_quantiles": dict(zip(map(str, quantiles), np.quantile(widths_array, quantiles).tolist())),
        "box_height_quantiles": dict(zip(map(str, quantiles), np.quantile(heights_array, quantiles).tolist())),
        "best_centered_shape_iou": {
            "mean": float(best_shape_iou.mean()),
            "median": float(np.median(best_shape_iou)),
            "coverage_at_0.5": float((best_shape_iou >= 0.5).mean()),
            "coverage_at_0.7": float((best_shape_iou >= 0.7).mean()),
            "note": "shape-only diagnostic; feature-grid center offsets are not included",
        },
        "suggested_anchor_config": {
            "anchor_sizes": [round(float(value), 3) for value in sizes],
            "anchor_scales": [round(float(value), 3) for value in scales],
            "anchor_ratios": [round(float(value), 3) for value in ratios],
        },
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["suggested_anchor_config"], indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
