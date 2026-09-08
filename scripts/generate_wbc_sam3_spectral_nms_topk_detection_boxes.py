#!/usr/bin/env python3
"""Filter SAM3 WBC candidates, suppress duplicates, and export detection boxes.

Pipeline
--------
1. Run the validated geometry + hyperspectral candidate filter.
2. Protect every positive spectral seed.
3. Suppress nested/duplicate boxes using box IoU or intersection-over-smaller.
4. Rank remaining candidates by spectral/SAM3 quality and cap each image at K.
5. Export per-image JSON, COCO-style JSON, reports, and final review figures.

The result remains a pseudo-label set pending human review; it is not final GT.

Example::

    conda run -n zsq_accl_mine \
      python scripts/generate_wbc_sam3_spectral_nms_topk_detection_boxes.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


DEFAULT_DATASET_NAME = (
    "2018WBC_cellcrop_512x512_noresize_contiguous20_entropy_"
    "multicandidate_manualoverride_filtered_minmax_20260830_2233"
)
DEFAULT_CANDIDATE_DIRNAME = "sam3_text_all_wbc"
DEFAULT_OUTPUT_DIRNAME = "sam3_spectral_nms_top5_detection_boxes"


def project_root_from_script() -> Path:
    root = Path(__file__).resolve().parents[1]
    if not (root / "data/original").is_dir():
        raise FileNotFoundError(f"Cannot identify project root from script path: {root}")
    return root


def parse_args() -> argparse.Namespace:
    project_root = project_root_from_script()
    data_root = project_root / "data" / DEFAULT_DATASET_NAME
    candidate_root = data_root / "sam_prompt_experiments" / DEFAULT_CANDIDATE_DIRNAME
    parser = argparse.ArgumentParser(
        description=(
            "Filter full-dataset SAM3 WBC candidates, suppress duplicate boxes, "
            "and export pending-review pseudo detection labels."
        )
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--data-root", type=Path, default=data_root)
    parser.add_argument(
        "--raw-root", type=Path, default=project_root / "data/original/2018WBC"
    )
    parser.add_argument("--candidate-root", type=Path, default=candidate_root)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=candidate_root.parent / DEFAULT_OUTPUT_DIRNAME,
        help="Defaults to a directory alongside sam3_text_all_wbc.",
    )

    # Validated spectral-filter settings.
    parser.add_argument("--spectral-score-threshold", type=float, default=1.0)
    parser.add_argument("--retain-quantile", type=float, default=0.12)
    parser.add_argument("--min-box-width", type=int, default=160)
    parser.add_argument("--min-box-height", type=int, default=160)
    parser.add_argument("--max-candidate-mask-fraction", type=float, default=0.10)
    parser.add_argument("--max-mask-pixels", type=int, default=6000)
    parser.add_argument("--max-ring-pixels", type=int, default=6000)

    # Final duplicate suppression and per-image cap.
    parser.add_argument("--box-iou-threshold", type=float, default=0.50)
    parser.add_argument(
        "--box-ios-threshold",
        type=float,
        default=0.80,
        help=(
            "Intersection divided by the smaller box area. This removes nested "
            "boxes that ordinary IoU can miss."
        ),
    )
    parser.add_argument("--max-boxes-per-image", type=int, default=5)

    parser.add_argument(
        "--max-sources", type=int, default=None,
        help="Debug/smoke-test only. Omit to process every candidate source.",
    )
    parser.add_argument(
        "--source-name",
        action="append",
        dest="source_names",
        help="Process only this source image; repeat to select an explicit subset.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reuse-filter-report",
        action="store_true",
        help="Reuse output-root/spectral_stage/spectral_filter_report.json.",
    )
    parser.add_argument(
        "--input-filter-report",
        type=Path,
        default=None,
        help=(
            "Use an existing full-dataset spectral_filter_report.json. Useful "
            "for representative random-subset review without refitting spectra."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Path:
    for name in ("project_root", "data_root", "raw_root", "candidate_root", "output_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.input_filter_report is not None:
        args.input_filter_report = args.input_filter_report.expanduser().resolve()
    helper = args.project_root / "data/original/wbc_sam3_spectral_filter_demo.py"
    required = (
        helper,
        args.data_root / "crop_coord_prompts.json",
        args.raw_root,
        args.candidate_root / "candidate_instances.json",
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required input is missing: {path}")
    if args.output_root == args.candidate_root:
        raise ValueError("output-root must differ from candidate-root")
    if args.spectral_score_threshold <= 0:
        raise ValueError("--spectral-score-threshold must be > 0")
    if not 0 < args.retain_quantile < 1:
        raise ValueError("--retain-quantile must be in (0, 1)")
    if args.min_box_width < 1 or args.min_box_height < 1:
        raise ValueError("box-size thresholds must be positive")
    if not 0 < args.box_iou_threshold <= 1:
        raise ValueError("--box-iou-threshold must be in (0, 1]")
    if not 0 < args.box_ios_threshold <= 1:
        raise ValueError("--box-ios-threshold must be in (0, 1]")
    if args.max_boxes_per_image < 1:
        raise ValueError("--max-boxes-per-image must be >= 1")
    if args.max_sources is not None and args.max_sources < 1:
        raise ValueError("--max-sources must be >= 1")
    if args.max_sources is not None and args.source_names:
        raise ValueError("--max-sources and --source-name are mutually exclusive")
    if args.input_filter_report is not None and args.reuse_filter_report:
        raise ValueError("--input-filter-report and --reuse-filter-report are mutually exclusive")
    if args.input_filter_report is not None and not args.input_filter_report.is_file():
        raise FileNotFoundError(
            f"Existing spectral filter report is missing: {args.input_filter_report}"
        )
    return helper


def spectral_stage_root(args: argparse.Namespace) -> Path:
    return args.output_root / "spectral_stage"


def spectral_report_path(args: argparse.Namespace) -> Path:
    return (
        args.input_filter_report
        if args.input_filter_report is not None
        else spectral_stage_root(args) / "spectral_filter_report.json"
    )


def build_filter_command(args: argparse.Namespace, helper: Path) -> list[str]:
    command = [
        sys.executable,
        str(helper),
        "--data-root", str(args.data_root),
        "--raw-root", str(args.raw_root),
        "--candidate-root", str(args.candidate_root),
        "--output-root", str(spectral_stage_root(args)),
        "--retain-quantile", str(args.retain_quantile),
        "--spectral-score-threshold", str(args.spectral_score_threshold),
        "--min-box-width", str(args.min_box_width),
        "--min-box-height", str(args.min_box_height),
        "--max-candidate-mask-fraction", str(args.max_candidate_mask_fraction),
        "--max-mask-pixels", str(args.max_mask_pixels),
        "--max-ring-pixels", str(args.max_ring_pixels),
    ]
    if args.max_sources is not None:
        command.extend(("--max-sources", str(args.max_sources)))
    for source_name in args.source_names or []:
        command.extend(("--source-name", source_name))
    if args.overwrite:
        command.append("--overwrite")
    return command


def source_image_metadata(candidate_root: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(
        (candidate_root / "candidate_instances.json").read_text(encoding="utf-8")
    )
    result: dict[str, dict[str, Any]] = {}
    for source in payload["sources"]:
        height, width = (int(value) for value in source["image_size_hw"])
        result[source["source_image"]] = {
            "height": height,
            "width": width,
            "source_image_path": source["source_image_path"],
        }
    return result


def box_overlap_metrics(a: list[int], b: list[int]) -> tuple[float, float]:
    ax1, ay1, ax2, ay2 = (float(value) for value in a)
    bx1, by1, bx2, by2 = (float(value) for value in b)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    iou = intersection / union if union > 0 else 0.0
    smaller = min(area_a, area_b)
    ios = intersection / smaller if smaller > 0 else 0.0
    return float(iou), float(ios)


def candidate_rank_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """Known positives first; then spectral, SAM3, and finally size evidence."""
    max_side = max(record["box_width_pixels"], record["box_height_pixels"])
    return (
        not bool(record["positive_seed"]),
        float(record["spectral_score"]),
        -float(record["sam3_score"]),
        -int(max_side),
        record["candidate_id"],
    )


def postprocess_source_records(
    records: list[dict[str, Any]],
    box_iou_threshold: float,
    box_ios_threshold: float,
    max_boxes_per_image: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Protect seeds, suppress duplicates, then apply a seed-safe top-K cap."""
    ranked = sorted(records, key=candidate_rank_key)
    nms_kept: list[dict[str, Any]] = []
    decisions: dict[str, dict[str, Any]] = {}
    for record in ranked:
        candidate_id = record["candidate_id"]
        if record["positive_seed"]:
            # Never discard a known positive seed, even if two prompts overlap.
            nms_kept.append(record)
            decisions[candidate_id] = {
                "candidate_id": candidate_id,
                "positive_seed": True,
                "duplicate_suppressed": False,
                "suppressed_by": None,
                "suppression_iou": None,
                "suppression_ios": None,
                "nms_keep": True,
            }
            continue

        suppressor = None
        suppress_iou = 0.0
        suppress_ios = 0.0
        for selected in nms_kept:
            iou, ios = box_overlap_metrics(
                record["predicted_box_xyxy"], selected["predicted_box_xyxy"]
            )
            if iou >= box_iou_threshold or ios >= box_ios_threshold:
                suppressor = selected
                suppress_iou, suppress_ios = iou, ios
                break
        if suppressor is None:
            nms_kept.append(record)
        decisions[candidate_id] = {
            "candidate_id": candidate_id,
            "positive_seed": False,
            "duplicate_suppressed": suppressor is not None,
            "suppressed_by": suppressor["candidate_id"] if suppressor else None,
            "suppression_iou": suppress_iou if suppressor else None,
            "suppression_ios": suppress_ios if suppressor else None,
            "nms_keep": suppressor is None,
        }

    seed_count = sum(bool(record["positive_seed"]) for record in nms_kept)
    final_limit = max(max_boxes_per_image, seed_count)
    final_records = nms_kept[:final_limit]
    final_ids = {record["candidate_id"] for record in final_records}
    nms_rank = {
        record["candidate_id"]: rank for rank, record in enumerate(nms_kept, start=1)
    }
    for record in ranked:
        decision = decisions[record["candidate_id"]]
        decision["nms_rank"] = nms_rank.get(record["candidate_id"])
        decision["topk_filtered"] = bool(
            decision["nms_keep"] and record["candidate_id"] not in final_ids
        )
        decision["final_keep"] = record["candidate_id"] in final_ids
    return final_records, [decisions[record["candidate_id"]] for record in ranked]


def save_final_review_visualization(
    source_name: str,
    source_image_path: str,
    spectral_records: list[dict[str, Any]],
    final_records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    image = np.asarray(Image.open(source_image_path).convert("RGB"))
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    axes[0].imshow(image)
    for record in spectral_records:
        x1, y1, x2, y2 = record["predicted_box_xyxy"]
        color = "yellow" if record["positive_seed"] else "cyan"
        axes[0].add_patch(Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color,
            linewidth=2.0 if record["positive_seed"] else 0.7, alpha=0.80,
        ))
    axes[0].set_title(
        f"Before duplicate suppression (n={len(spectral_records)}); yellow=seed"
    )
    axes[0].axis("off")

    overlay = image.astype(np.float32) / 255.0
    colors = plt.get_cmap("tab10", max(len(final_records), 1))
    loaded_masks: list[np.ndarray | None] = []
    for index, record in enumerate(final_records):
        mask_path = Path(record["mask_path"])
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0 if mask_path.is_file() else None
        loaded_masks.append(mask)
        if mask is not None and mask.shape == image.shape[:2]:
            color = np.asarray(colors(index)[:3], dtype=np.float32)
            overlay[mask] = 0.55 * overlay[mask] + 0.45 * color
    axes[1].imshow(np.clip(overlay, 0, 1))
    for index, record in enumerate(final_records):
        x1, y1, x2, y2 = record["predicted_box_xyxy"]
        color = "yellow" if record["positive_seed"] else colors(index)
        axes[1].add_patch(Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color,
            linewidth=2.4 if record["positive_seed"] else 1.5,
        ))
        short_id = record["candidate_id"].rsplit("_", 1)[-1]
        label = (
            f"{short_id} {record['box_width_pixels']}x{record['box_height_pixels']} "
            f"d={record['spectral_score']:.2f}"
        )
        axes[1].text(
            x1, y1, label, fontsize=6, color="black",
            bbox={"facecolor": color, "alpha": 0.80, "pad": 1},
        )
    axes[1].set_title(f"Final seed-protected NMS + Top-K boxes (n={len(final_records)})")
    axes[1].axis("off")
    fig.suptitle(source_name, fontsize=14)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def export_detection_annotations(args: argparse.Namespace) -> dict[str, Any]:
    report_path = spectral_report_path(args)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    image_metadata = source_image_metadata(args.candidate_root)
    report_source_names = [source["source_image"] for source in report["sources"]]
    if args.source_names:
        missing = [name for name in args.source_names if name not in report_source_names]
        if missing:
            raise KeyError(f"Selected images are absent from spectral report: {missing}")
        source_order = list(args.source_names)
    elif args.max_sources is not None:
        source_order = report_source_names[: args.max_sources]
    else:
        source_order = report_source_names
    selected_source_set = set(source_order)
    spectral_by_source: dict[str, list[dict[str, Any]]] = {
        source_name: [] for source_name in source_order
    }
    for record in report["candidates"]:
        if record["spectral_keep"] and record["source_image"] in selected_source_set:
            spectral_by_source[record["source_image"]].append(record)

    final_by_source: dict[str, list[dict[str, Any]]] = {}
    decisions_by_source: dict[str, list[dict[str, Any]]] = {}
    for source_name in source_order:
        final, decisions = postprocess_source_records(
            spectral_by_source[source_name],
            args.box_iou_threshold,
            args.box_ios_threshold,
            args.max_boxes_per_image,
        )
        final_by_source[source_name] = final
        decisions_by_source[source_name] = decisions

    annotations_root = args.output_root / "annotations"
    visualization_root = args.output_root / "review_visualizations"
    for generated_root in (annotations_root, visualization_root):
        if generated_root.exists():
            if not args.overwrite and not args.reuse_filter_report:
                raise FileExistsError(f"Generated output already exists: {generated_root}")
            shutil.rmtree(generated_root)
        generated_root.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).isoformat()
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    postprocess_sources: list[dict[str, Any]] = []
    postprocess_decisions: list[dict[str, Any]] = []
    review_lines = [
        "# SAM3 spectral + seed-protected NMS/Top-K review checklist",
        "",
        "> Pseudo-label candidates pending human review; not final GT.",
        "",
    ]
    global_annotation_id = 1
    for image_id, source_name in enumerate(source_order, start=1):
        metadata = image_metadata[source_name]
        source_records = final_by_source[source_name]
        image_entry = {
            "id": image_id,
            "file_name": source_name,
            "width": metadata["width"],
            "height": metadata["height"],
            "source_image_path": metadata["source_image_path"],
        }
        coco_images.append(image_entry)
        compact_annotations = []
        for local_id, record in enumerate(source_records, start=1):
            x1, y1, x2, y2 = (int(value) for value in record["predicted_box_xyxy"])
            box_width, box_height = x2 - x1, y2 - y1
            compact = {
                "id": local_id,
                "candidate_id": record["candidate_id"],
                "category_id": 1,
                "category_name": "WBC",
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xywh": [x1, y1, box_width, box_height],
                "box_width_pixels": box_width,
                "box_height_pixels": box_height,
                "mask_area_pixels": int(record["mask_area_pixels"]),
                "iscrowd": 0,
                "mask_path": record["mask_path"],
                "sam3_score": float(record["sam3_score"]),
                "spectral_score": float(record["spectral_score"]),
                "positive_seed": bool(record["positive_seed"]),
                "review_status": "pending",
            }
            compact_annotations.append(compact)
            coco_annotations.append({
                "id": global_annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": compact["bbox_xywh"],
                "area": compact["mask_area_pixels"],
                "iscrowd": 0,
                "candidate_id": compact["candidate_id"],
                "mask_path": compact["mask_path"],
                "sam3_score": compact["sam3_score"],
                "spectral_score": compact["spectral_score"],
                "positive_seed": compact["positive_seed"],
                "review_status": "pending",
            })
            global_annotation_id += 1

        per_image = {
            "schema_version": 2,
            "annotation_status": "pseudo_labels_pending_human_review",
            "postprocess": {
                "method": "positive-seed-protected box NMS/containment + top-k",
                "box_iou_threshold": args.box_iou_threshold,
                "box_ios_threshold": args.box_ios_threshold,
                "max_boxes_per_image": args.max_boxes_per_image,
            },
            "image": image_entry,
            "categories": [{"id": 1, "name": "WBC"}],
            "annotations": compact_annotations,
        }
        (annotations_root / f"{Path(source_name).stem}.json").write_text(
            json.dumps(per_image, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        save_final_review_visualization(
            source_name,
            metadata["source_image_path"],
            spectral_by_source[source_name],
            source_records,
            visualization_root / source_name,
        )

        decisions = decisions_by_source[source_name]
        postprocess_decisions.extend(
            {"source_image": source_name, **decision} for decision in decisions
        )
        postprocess_sources.append({
            "source_image": source_name,
            "n_spectral_retained": len(spectral_by_source[source_name]),
            "n_duplicate_suppressed": sum(
                decision["duplicate_suppressed"] for decision in decisions
            ),
            "n_topk_filtered": sum(decision["topk_filtered"] for decision in decisions),
            "n_positive_seeds": sum(
                record["positive_seed"] for record in spectral_by_source[source_name]
            ),
            "n_final_boxes": len(source_records),
            "review_visualization_path": str(
                (visualization_root / source_name).resolve()
            ),
        })
        review_lines.extend([
            f"- [ ] `{source_name}` — {len(spectral_by_source[source_name])} spectral "
            f"→ {len(source_records)} final boxes",
            f"  - visualization: `review_visualizations/{source_name}`",
            f"  - annotation: `annotations/{Path(source_name).stem}.json`",
        ])

    coco_payload = {
        "info": {
            "description": (
                "SAM3 + hyperspectral + seed-protected NMS/Top-K WBC pseudo boxes"
            ),
            "generated_at": generated_at,
            "annotation_status": "pseudo_labels_pending_human_review",
            "source_candidate_root": str(args.candidate_root),
            "spectral_filter_report": str(report_path),
            "spectral_score_threshold": report["applied_spectral_score_threshold"],
            "min_box_width": report["min_box_width"],
            "min_box_height": report["min_box_height"],
            "min_box_rule": report["min_box_rule"],
            "box_iou_threshold": args.box_iou_threshold,
            "box_ios_threshold": args.box_ios_threshold,
            "max_boxes_per_image": args.max_boxes_per_image,
            "positive_seed_policy": "always_keep_even_if_cap_is_exceeded",
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": 1, "name": "WBC", "supercategory": "blood_cell"}],
    }
    aggregate_path = args.output_root / "filtered_detection_boxes_coco.json"
    aggregate_path.write_text(
        json.dumps(coco_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    postprocess_report = {
        "status": "pseudo_labels_pending_human_review",
        "method": "positive-seed-protected box NMS/containment + top-k",
        "box_iou_threshold": args.box_iou_threshold,
        "box_ios_threshold": args.box_ios_threshold,
        "max_boxes_per_image": args.max_boxes_per_image,
        "ranking": [
            "positive_seed descending",
            "spectral_score ascending",
            "sam3_score descending",
            "max_box_side descending",
        ],
        "n_spectral_retained": sum(len(items) for items in spectral_by_source.values()),
        "n_duplicate_suppressed": sum(
            decision["duplicate_suppressed"] for decision in postprocess_decisions
        ),
        "n_topk_filtered": sum(
            decision["topk_filtered"] for decision in postprocess_decisions
        ),
        "n_final_boxes": len(coco_annotations),
        "n_positive_seeds_before": sum(
            record["positive_seed"] for items in spectral_by_source.values() for record in items
        ),
        "n_positive_seeds_after": sum(
            record["positive_seed"] for items in final_by_source.values() for record in items
        ),
        "sources": postprocess_sources,
        "candidates": postprocess_decisions,
    }
    (args.output_root / "postprocess_report.json").write_text(
        json.dumps(postprocess_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_root / "review_index.md").write_text(
        "\n".join(review_lines) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "pseudo_labels_pending_human_review",
        "n_images": len(coco_images),
        "n_images_with_boxes": sum(bool(items) for items in final_by_source.values()),
        "n_images_without_boxes": sum(not items for items in final_by_source.values()),
        "n_spectral_retained": postprocess_report["n_spectral_retained"],
        "n_duplicate_suppressed": postprocess_report["n_duplicate_suppressed"],
        "n_topk_filtered": postprocess_report["n_topk_filtered"],
        "n_detection_boxes": len(coco_annotations),
        "n_positive_seeds_before": postprocess_report["n_positive_seeds_before"],
        "n_positive_seeds_after": postprocess_report["n_positive_seeds_after"],
        "annotations_root": str(annotations_root),
        "coco_json": str(aggregate_path),
        "review_visualizations": str(visualization_root),
        "postprocess_report": str(args.output_root / "postprocess_report.json"),
    }
    (args.output_root / "detection_export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    helper = resolve_paths(args)
    command = build_filter_command(args, helper)
    candidate_payload = json.loads(
        (args.candidate_root / "candidate_instances.json").read_text(encoding="utf-8")
    )
    available_sources = len(candidate_payload["sources"])
    requested_sources = (
        len(args.source_names)
        if args.source_names
        else min(available_sources, args.max_sources)
        if args.max_sources is not None
        else available_sources
    )
    print("=" * 78)
    print("WBC SAM3 spectral + seed-protected NMS/Top-K detection-box export")
    print(f"Candidate root:       {args.candidate_root}")
    print(f"Available sources:    {available_sources}")
    print(f"Requested sources:    {requested_sources}")
    print(f"Output root:          {args.output_root}")
    print(f"Spectral threshold:   {args.spectral_score_threshold}")
    print(
        "Box-size rule:       reject only if width < "
        f"{args.min_box_width} AND height < {args.min_box_height}"
    )
    print(
        f"Duplicate rule:      IoU >= {args.box_iou_threshold} OR "
        f"IoS >= {args.box_ios_threshold}"
    )
    print(f"Per-image cap:        {args.max_boxes_per_image}, positive seeds protected")
    print(
        "Spectral report:      "
        f"{args.input_filter_report if args.input_filter_report else '<compute/reuse in output>'}"
    )
    print("Annotation status:    pseudo labels pending human review")
    print("=" * 78)
    if args.dry_run:
        print("Dry-run child command:")
        print(" ".join(command))
        return

    report_path = spectral_report_path(args)
    if args.input_filter_report is not None:
        pass
    elif args.reuse_filter_report:
        if not report_path.is_file():
            raise FileNotFoundError(
                f"--reuse-filter-report requested but report is missing: {report_path}"
            )
    else:
        subprocess.run(command, cwd=args.project_root, check=True)
    summary = export_detection_annotations(args)
    if summary["n_positive_seeds_before"] != summary["n_positive_seeds_after"]:
        raise RuntimeError(
            "Positive-seed invariant violated: "
            f"{summary['n_positive_seeds_before']} before vs "
            f"{summary['n_positive_seeds_after']} after"
        )
    print("\nDetection-box export completed successfully:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
