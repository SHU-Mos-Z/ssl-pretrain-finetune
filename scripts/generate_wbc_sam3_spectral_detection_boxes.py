#!/usr/bin/env python3
"""Batch-filter SAM3 WBC candidates and export reviewable detection boxes.

The upstream ``sam3_text_all_wbc`` directory contains high-recall candidate
masks.  This script runs the validated geometry + hyperspectral filter over all
sources in that directory, then exports:

* one compact JSON annotation per source image;
* a COCO-style aggregate JSON;
* spectral diagnostics and per-source review visualizations.

The exported boxes are deliberately marked ``pending_human_review``.  They are
pseudo-label candidates, not final ground truth, until a person accepts them.

Run from any working directory, for example::

    conda run -n zsq_accl_mine \
      python scripts/generate_wbc_sam3_spectral_detection_boxes.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DATASET_NAME = (
    "2018WBC_cellcrop_512x512_noresize_contiguous20_entropy_"
    "multicandidate_manualoverride_filtered_minmax_20260830_2233"
)
DEFAULT_CANDIDATE_DIRNAME = "sam3_text_all_wbc"
DEFAULT_OUTPUT_DIRNAME = "sam3_spectral_filtered_detection_boxes"


def project_root_from_script() -> Path:
    root = Path(__file__).resolve().parents[1]
    if not (root / "data/original").is_dir():
        raise FileNotFoundError(f"Cannot identify project root from script path: {root}")
    return root


def parse_args() -> argparse.Namespace:
    project_root = project_root_from_script()
    data_root = project_root / "data" / DEFAULT_DATASET_NAME
    candidate_root = (
        data_root / "sam_prompt_experiments" / DEFAULT_CANDIDATE_DIRNAME
    )
    parser = argparse.ArgumentParser(
        description=(
            "Filter full-dataset SAM3 WBC candidates spectrally and export "
            "pending-review detection boxes."
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
    parser.add_argument("--spectral-score-threshold", type=float, default=1.0)
    parser.add_argument("--retain-quantile", type=float, default=0.12)
    parser.add_argument("--min-box-width", type=int, default=160)
    parser.add_argument("--min-box-height", type=int, default=160)
    parser.add_argument("--max-candidate-mask-fraction", type=float, default=0.10)
    parser.add_argument("--max-mask-pixels", type=int, default=6000)
    parser.add_argument("--max-ring-pixels", type=int, default=6000)
    parser.add_argument(
        "--max-sources",
        type=int,
        default=None,
        help="Debug/smoke-test only. Omit to process every source candidate.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace prior output at output-root."
    )
    parser.add_argument(
        "--reuse-filter-report",
        action="store_true",
        help="Skip feature extraction and export boxes from an existing report.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate inputs and print the child command."
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Path:
    for name in ("project_root", "data_root", "raw_root", "candidate_root", "output_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
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
        raise ValueError("box size thresholds must be positive")
    if args.max_sources is not None and args.max_sources < 1:
        raise ValueError("--max-sources must be >= 1")
    return helper


def build_filter_command(args: argparse.Namespace, helper: Path) -> list[str]:
    command = [
        sys.executable,
        str(helper),
        "--data-root", str(args.data_root),
        "--raw-root", str(args.raw_root),
        "--candidate-root", str(args.candidate_root),
        "--output-root", str(args.output_root),
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


def export_detection_annotations(args: argparse.Namespace) -> dict[str, Any]:
    report_path = args.output_root / "spectral_filter_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    image_metadata = source_image_metadata(args.candidate_root)
    source_order = [source["source_image"] for source in report["sources"]]
    records_by_source: dict[str, list[dict[str, Any]]] = {
        source_name: [] for source_name in source_order
    }
    for record in report["candidates"]:
        if record["spectral_keep"]:
            records_by_source[record["source_image"]].append(record)

    annotations_root = args.output_root / "annotations"
    if annotations_root.exists():
        if not args.overwrite and not args.reuse_filter_report:
            raise FileExistsError(f"Annotation directory already exists: {annotations_root}")
        shutil.rmtree(annotations_root)
    annotations_root.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).isoformat()
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    review_lines = [
        "# SAM3 + hyperspectral filtering manual-review checklist",
        "",
        "> These are pseudo-label candidates pending human review, not final GT.",
        "",
    ]
    global_annotation_id = 1
    for image_id, source_name in enumerate(source_order, start=1):
        height = image_metadata[source_name]["height"]
        width = image_metadata[source_name]["width"]
        source_records = sorted(
            records_by_source[source_name], key=lambda item: item["candidate_id"]
        )
        image_entry = {
            "id": image_id,
            "file_name": source_name,
            "width": width,
            "height": height,
            "source_image_path": image_metadata[source_name]["source_image_path"],
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
            "schema_version": 1,
            "annotation_status": "pseudo_labels_pending_human_review",
            "image": image_entry,
            "categories": [{"id": 1, "name": "WBC"}],
            "annotations": compact_annotations,
        }
        (annotations_root / f"{Path(source_name).stem}.json").write_text(
            json.dumps(per_image, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        visualization = args.output_root / "review_visualizations" / source_name
        review_lines.extend([
            f"- [ ] `{source_name}` — {len(compact_annotations)} retained boxes",
            f"  - visualization: `{visualization.relative_to(args.output_root)}`",
            f"  - annotation: `annotations/{Path(source_name).stem}.json`",
        ])

    coco_payload = {
        "info": {
            "description": "SAM3 + hyperspectral WBC pseudo boxes pending human review",
            "generated_at": generated_at,
            "annotation_status": "pseudo_labels_pending_human_review",
            "source_candidate_root": str(args.candidate_root),
            "spectral_filter_report": str(report_path),
            "spectral_score_threshold": report["applied_spectral_score_threshold"],
            "min_box_width": report["min_box_width"],
            "min_box_height": report["min_box_height"],
            "min_box_rule": report["min_box_rule"],
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
    (args.output_root / "review_index.md").write_text(
        "\n".join(review_lines) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "pseudo_labels_pending_human_review",
        "n_images": len(coco_images),
        "n_images_with_boxes": sum(bool(items) for items in records_by_source.values()),
        "n_images_without_boxes": sum(not items for items in records_by_source.values()),
        "n_detection_boxes": len(coco_annotations),
        "annotations_root": str(annotations_root),
        "coco_json": str(aggregate_path),
        "review_visualizations": str(args.output_root / "review_visualizations"),
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
        min(available_sources, args.max_sources)
        if args.max_sources is not None else available_sources
    )
    print("=" * 78)
    print("WBC SAM3 hyperspectral filtering and detection-box export")
    print(f"Candidate root:       {args.candidate_root}")
    print(f"Available sources:    {available_sources}")
    print(f"Requested sources:    {requested_sources}")
    print(f"Output root:          {args.output_root}")
    print(f"Spectral threshold:   {args.spectral_score_threshold}")
    print(
        "Box-size rule:       reject only if width < "
        f"{args.min_box_width} AND height < {args.min_box_height}"
    )
    print("Annotation status:    pseudo labels pending human review")
    print("=" * 78)
    if args.dry_run:
        print("Dry-run child command:")
        print(" ".join(command))
        return

    report_path = args.output_root / "spectral_filter_report.json"
    if args.reuse_filter_report:
        if not report_path.is_file():
            raise FileNotFoundError(
                f"--reuse-filter-report requested but report is missing: {report_path}"
            )
    else:
        subprocess.run(command, cwd=args.project_root, check=True)
    summary = export_detection_annotations(args)
    print("\nDetection-box export completed successfully:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
