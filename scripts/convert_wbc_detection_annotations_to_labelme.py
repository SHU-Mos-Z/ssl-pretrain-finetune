#!/usr/bin/env python3
"""Convert WBC detection-box JSON files to editable LabelMe JSON files.

Each output directory contains a same-stem ``PNG + JSON`` pair.  The matching
pseudo-RGB PNG is also embedded in ``imageData`` as raw Base64 (without a
data-URI prefix).  This layout lets LabelMe open the directory as an image list,
load each same-stem annotation automatically, and move continuously between
images. Detection boxes are exported as ``rectangle`` shapes.

The conversion is format-only: pseudo labels remain pending human review.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_DATASET_NAME = (
    "2018WBC_cellcrop_512x512_noresize_contiguous20_entropy_"
    "multicandidate_manualoverride_filtered_minmax_20260830_2233"
)
DEFAULT_EXPERIMENT_NAME = "sam3_spectral_nms_top5_detection_boxes"
WBC_CLASS_PREFIXES = ("B", "E", "L", "M", "N")


def project_root_from_script() -> Path:
    root = Path(__file__).resolve().parents[1]
    if not (root / "data/original/2018WBC").is_dir():
        raise FileNotFoundError(f"Cannot identify project root from script path: {root}")
    return root


def parse_args() -> argparse.Namespace:
    project_root = project_root_from_script()
    experiment_root = (
        project_root
        / "data"
        / DEFAULT_DATASET_NAME
        / "sam_prompt_experiments"
        / DEFAULT_EXPERIMENT_NAME
    )
    parser = argparse.ArgumentParser(
        description=(
            "Convert per-image WBC detection JSON files into LabelMe rectangle "
            "annotations with embedded pseudo-RGB imageData."
        )
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--annotations-root", type=Path, default=experiment_root / "annotations"
    )
    parser.add_argument(
        "--rgb-root", type=Path, default=project_root / "data/original/2018WBC"
    )
    parser.add_argument(
        "--output-root", type=Path, default=experiment_root / "annotations_labelme"
    )
    parser.add_argument("--label", default="WBC")
    parser.add_argument("--labelme-version", default="6.3.0")
    parser.add_argument(
        "--image-placement",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help=(
            "How to place each pseudo-RGB PNG beside its LabelMe JSON. "
            "The default 'copy' is portable; hardlink/symlink save space."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Debug/smoke-test only. Omit to convert every annotation JSON.",
    )
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_and_validate_args(args: argparse.Namespace) -> None:
    for name in ("project_root", "annotations_root", "rgb_root", "output_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if not args.annotations_root.is_dir():
        raise FileNotFoundError(f"Annotation directory is missing: {args.annotations_root}")
    if not args.rgb_root.is_dir():
        raise FileNotFoundError(f"Pseudo-RGB root is missing: {args.rgb_root}")
    for class_prefix in WBC_CLASS_PREFIXES:
        class_dir = args.rgb_root / f"{class_prefix}_rgb"
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Pseudo-RGB class directory is missing: {class_dir}")
    if not args.label.strip():
        raise ValueError("--label must not be empty")
    if not args.labelme_version.strip():
        raise ValueError("--labelme-version must not be empty")
    if args.max_files is not None and args.max_files < 1:
        raise ValueError("--max-files must be >= 1")
    if args.log_interval < 1:
        raise ValueError("--log-interval must be >= 1")


def discover_annotation_files(args: argparse.Namespace) -> list[Path]:
    files = sorted(args.annotations_root.glob("*.json"), key=lambda path: path.name)
    if not files:
        raise FileNotFoundError(f"No JSON files found under {args.annotations_root}")
    if args.max_files is not None:
        files = files[: args.max_files]
    return files


def build_rgb_index(rgb_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for class_prefix in WBC_CLASS_PREFIXES:
        class_dir = rgb_root / f"{class_prefix}_rgb"
        for image_path in sorted(class_dir.glob("*.png")):
            if image_path.name in index:
                duplicates.setdefault(image_path.name, [index[image_path.name]]).append(
                    image_path
                )
            else:
                index[image_path.name] = image_path.resolve()
    if duplicates:
        formatted = {
            name: [str(path) for path in paths] for name, paths in duplicates.items()
        }
        raise RuntimeError(f"Duplicate pseudo-RGB basenames detected: {formatted}")
    return index


def load_detection_annotation(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ValueError(f"Cannot parse detection annotation {path}: {error}") from error
    for key in ("image", "annotations"):
        if key not in payload:
            raise KeyError(f"Missing {key!r} in {path}")
    image = payload["image"]
    for key in ("file_name", "width", "height"):
        if key not in image:
            raise KeyError(f"Missing image.{key} in {path}")
    if not isinstance(payload["annotations"], list):
        raise TypeError(f"annotations must be a list: {path}")
    return payload


def resolve_rgb_path(
    annotation_path: Path, payload: dict[str, Any], rgb_index: dict[str, Path]
) -> Path:
    image_name = str(payload["image"]["file_name"])
    class_prefix = image_name.split("-", 1)[0].upper()
    if class_prefix not in WBC_CLASS_PREFIXES:
        raise ValueError(
            f"Cannot infer WBC class prefix from {image_name!r}: {annotation_path}"
        )
    image_path = rgb_index.get(image_name)
    if image_path is None:
        raise FileNotFoundError(
            f"No pseudo-RGB PNG matches image.file_name={image_name!r}: {annotation_path}"
        )
    expected_parent = f"{class_prefix}_rgb"
    if image_path.parent.name != expected_parent:
        raise ValueError(
            f"Class-directory mismatch for {image_name}: expected {expected_parent}, "
            f"found {image_path.parent.name}"
        )
    return image_path


def inspect_png(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            if image.format != "PNG":
                raise ValueError(f"Expected PNG but Pillow detected {image.format}: {path}")
    except Exception as error:
        raise ValueError(f"Invalid pseudo-RGB PNG {path}: {error}") from error
    return int(width), int(height)


def validated_box(
    annotation: dict[str, Any], image_width: int, image_height: int, source: Path
) -> tuple[float, float, float, float]:
    box = annotation.get("bbox_xyxy")
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(
            f"Annotation {annotation.get('candidate_id')} has invalid bbox_xyxy in {source}"
        )
    x1, y1, x2, y2 = (float(value) for value in box)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError(f"Non-finite bbox in {source}: {box}")
    if not (0.0 <= x1 < x2 <= image_width and 0.0 <= y1 < y2 <= image_height):
        raise ValueError(
            f"Out-of-bounds or degenerate bbox in {source}: {box}; "
            f"image={image_width}x{image_height}"
        )
    return x1, y1, x2, y2


def shape_description(annotation: dict[str, Any]) -> str:
    candidate_id = annotation.get("candidate_id", "")
    spectral_score = annotation.get("spectral_score")
    sam3_score = annotation.get("sam3_score")
    positive_seed = bool(annotation.get("positive_seed", False))
    spectral_text = f"{float(spectral_score):.6g}" if spectral_score is not None else "null"
    sam3_text = f"{float(sam3_score):.6g}" if sam3_score is not None else "null"
    return (
        f"candidate_id={candidate_id}; spectral_score={spectral_text}; "
        f"sam3_score={sam3_text}; positive_seed={str(positive_seed).lower()}"
    )


def make_labelme_payload(
    args: argparse.Namespace,
    detection_payload: dict[str, Any],
    annotation_path: Path,
    image_path: Path,
) -> tuple[dict[str, Any], int, int]:
    png_bytes = image_path.read_bytes()
    image_data = base64.b64encode(png_bytes).decode("ascii")
    image_width, image_height = inspect_png(image_path)
    declared_width = int(detection_payload["image"]["width"])
    declared_height = int(detection_payload["image"]["height"])
    if (declared_width, declared_height) != (image_width, image_height):
        raise ValueError(
            f"Image-size mismatch for {annotation_path.name}: annotation="
            f"{declared_width}x{declared_height}, PNG={image_width}x{image_height}"
        )

    shapes = []
    for annotation in detection_payload["annotations"]:
        x1, y1, x2, y2 = validated_box(
            annotation, image_width, image_height, annotation_path
        )
        group_id = annotation.get("id")
        if group_id is not None:
            group_id = int(group_id)
        shapes.append({
            "label": args.label,
            "points": [[x1, y1], [x2, y2]],
            "group_id": group_id,
            "description": shape_description(annotation),
            "shape_type": "rectangle",
            "flags": {},
            "mask": None,
        })

    labelme_payload = {
        "version": args.labelme_version,
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": image_data,
        "imageHeight": image_height,
        "imageWidth": image_width,
    }
    return labelme_payload, len(png_bytes), len(image_data)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_place_image(source: Path, destination: Path, mode: str) -> None:
    """Place one image beside its JSON without exposing a partial output file."""
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        if mode == "copy":
            shutil.copy2(source, temporary)
        elif mode == "hardlink":
            os.link(source, temporary)
        elif mode == "symlink":
            temporary.symlink_to(source)
        else:  # guarded by argparse choices
            raise ValueError(f"Unsupported image placement mode: {mode}")
        temporary.replace(destination)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def preflight(
    args: argparse.Namespace,
    annotation_files: list[Path],
    rgb_index: dict[str, Path],
) -> list[dict[str, Any]]:
    tasks = []
    image_names: set[str] = set()
    output_names: set[str] = set()
    for annotation_path in annotation_files:
        payload = load_detection_annotation(annotation_path)
        image_name = str(payload["image"]["file_name"])
        if image_name in image_names:
            raise RuntimeError(f"Duplicate image.file_name in annotations: {image_name}")
        image_names.add(image_name)
        image_path = resolve_rgb_path(annotation_path, payload, rgb_index)
        image_width, image_height = inspect_png(image_path)
        if (int(payload["image"]["width"]), int(payload["image"]["height"])) != (
            image_width,
            image_height,
        ):
            raise ValueError(
                f"Image-size mismatch for {annotation_path.name}: annotation="
                f"{payload['image']['width']}x{payload['image']['height']}, "
                f"PNG={image_width}x{image_height}"
            )
        for annotation in payload["annotations"]:
            validated_box(annotation, image_width, image_height, annotation_path)
        output_name = annotation_path.name
        if output_name in output_names:
            raise RuntimeError(f"Duplicate output basename: {output_name}")
        output_names.add(output_name)
        tasks.append({
            "annotation_path": annotation_path,
            "image_path": image_path,
            "output_path": args.output_root / output_name,
            "output_image_path": args.output_root / image_path.name,
            "n_boxes": len(payload["annotations"]),
            "image_width": image_width,
            "image_height": image_height,
        })

    conflicts = [
        path
        for task in tasks
        for path in (task["output_path"], task["output_image_path"])
        if path.exists() or path.is_symlink()
    ]
    if conflicts and not args.overwrite:
        examples = ", ".join(str(path) for path in conflicts[:5])
        raise FileExistsError(
            f"{len(conflicts)} output LabelMe JSON/PNG files already exist; "
            f"use --overwrite. "
            f"Examples: {examples}"
        )
    return tasks


def main() -> None:
    args = parse_args()
    resolve_and_validate_args(args)
    annotation_files = discover_annotation_files(args)
    rgb_index = build_rgb_index(args.rgb_root)
    tasks = preflight(args, annotation_files, rgb_index)
    total_boxes = sum(task["n_boxes"] for task in tasks)
    print("=" * 78)
    print("WBC detection JSON -> LabelMe JSON conversion")
    print(f"Input annotations:    {args.annotations_root}")
    print(f"Pseudo-RGB root:      {args.rgb_root}")
    print(f"Output root:          {args.output_root}")
    print(f"Files to convert:     {len(tasks)}")
    print(f"Rectangles to export: {total_boxes}")
    print(f"Label / version:      {args.label} / LabelMe {args.labelme_version}")
    print("imageData:            embedded Base64 PNG bytes")
    print(f"Image placement:      {args.image_placement} beside same-stem JSON")
    print(f"Overwrite:            {args.overwrite}")
    print("=" * 78)
    if args.dry_run:
        print("Dry run completed: all paths, images, dimensions, and boxes are valid.")
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    converted = []
    raw_image_bytes = 0
    base64_characters = 0
    for index, task in enumerate(tasks, start=1):
        detection_payload = load_detection_annotation(task["annotation_path"])
        labelme_payload, png_size, encoded_size = make_labelme_payload(
            args,
            detection_payload,
            task["annotation_path"],
            task["image_path"],
        )
        atomic_place_image(
            task["image_path"], task["output_image_path"], args.image_placement
        )
        atomic_write_json(task["output_path"], labelme_payload)
        raw_image_bytes += png_size
        base64_characters += encoded_size
        converted.append({
            "source_annotation": str(task["annotation_path"]),
            "pseudo_rgb": str(task["image_path"]),
            "labelme_image": str(task["output_image_path"]),
            "labelme_json": str(task["output_path"]),
            "image_width": task["image_width"],
            "image_height": task["image_height"],
            "n_rectangles": task["n_boxes"],
            "png_bytes": png_size,
            "base64_characters": encoded_size,
        })
        if index == 1 or index % args.log_interval == 0 or index == len(tasks):
            print(
                f"[{index:>4}/{len(tasks)}] {task['annotation_path'].name}: "
                f"{task['n_boxes']} rectangles"
            )

    report = {
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "labelme_version": args.labelme_version,
        "label": args.label,
        "annotation_status": "pseudo_labels_pending_human_review",
        "directory_usage": "Open this output_root with LabelMe Open Dir",
        "image_placement": args.image_placement,
        "input_annotations_root": str(args.annotations_root),
        "rgb_root": str(args.rgb_root),
        "output_root": str(args.output_root),
        "n_input_annotations": len(tasks),
        "n_output_labelme_json": len(converted),
        "n_output_images": len(converted),
        "n_rectangles": total_boxes,
        "n_empty_annotations": sum(task["n_boxes"] == 0 for task in tasks),
        "raw_png_bytes_embedded": raw_image_bytes,
        "base64_characters": base64_characters,
        "files": converted,
    }
    report_path = args.output_root / "labelme_conversion_report.json"
    atomic_write_json(report_path, report)
    print("\nConversion completed successfully.")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
