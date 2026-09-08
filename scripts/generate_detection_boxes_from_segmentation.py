#!/usr/bin/env python3
"""Convert semantic segmentation masks stored as ``.npy`` files to COCO boxes.

This script is intentionally a *candidate annotation generator*, rather than a
claim that every semantic connected component is a valid object instance.
Semantic masks do not contain enough information to separate two touching
objects.  Therefore:

* every positive connected component is represented in the output;
* geometrically plausible components use ``iscrowd = 0``;
* tiny, giant, very thin, low-fill, or dataset-specific ambiguous components
  use ``iscrowd = 1`` and carry an explicit reason in ``attributes``;
* no morphology is applied, because erosion/dilation can invent instance
  boundaries or merge neighbouring objects;
* a CSV audit table records every generated component.

The main output is one COCO-style JSON file for the whole input directory.
COCO stores all image and box annotations in one JSON file; unlike Pascal VOC
or YOLO it does not normally create one JSON per image.  ``file_name`` points
to the hyperspectral ``.npy`` file.  A future detector data loader must
therefore load NumPy cubes instead of assuming RGB JPEG/PNG input.

Expected input layout::

    DATASET_ROOT/
    ├── images/
    │   ├── sample_a.npy
    │   └── sample_b.npy
    └── masks/
        ├── sample_a.npy
        └── sample_b.npy

Pairing is by filename stem.  The script reports the number of image files,
mask files, name-matched pairs, successfully readable pairs, and failures.

Examples::

    # Dataset policy is inferred from the directory name.
    python scripts/generate_detection_boxes_from_segmentation.py \
        data/GPCC_Resized_512_640_to_256_256_overlap_0_0_preprocessed

    # Explicit policy and output location.
    python scripts/generate_detection_boxes_from_segmentation.py \
        data/MDC_Resized_256_320_to_256_256_overlap_0_0_preprocessed \
        --dataset mdc \
        --output-dir records/detection_labels/MDC

    # Supply verified semantic class names for a multi-class dataset.
    python scripts/generate_detection_boxes_from_segmentation.py \
        data/LUAD_patch_400x400_overlap_0x0_to_256x256_minmax/Training \
        --dataset luad \
        --category-names-json '{"1":"class_1","2":"class_2","3":"class_3"}'

The built-in class names for LUAD/TMA deliberately remain ``class_1`` etc.
Replace them only after the original datasets' class-ID semantics have been
verified.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class PairingReport:
    """Result of matching files under ``images/`` and ``masks/``."""

    dataset_dir: str
    images_dir: str
    masks_dir: str
    n_images: int
    n_masks: int
    n_name_matched_pairs: int
    images_without_mask: tuple[str, ...]
    masks_without_image: tuple[str, ...]


@dataclass(frozen=True)
class ComponentRules:
    """Rules deciding whether a connected component is a trainable instance."""

    min_area_px: int = 64
    min_side_px: int = 4
    max_relative_area: float = 0.25
    max_aspect_ratio: float = 8.0
    min_fill_ratio: float = 0.20
    touching_border_is_crowd: bool = False
    force_crowd: bool = False


@dataclass(frozen=True)
class DatasetPolicy:
    """Dataset-specific category definitions and component rules."""

    key: str
    description: str
    category_names: Mapping[int, str]
    default_rules: ComponentRules = field(default_factory=ComponentRules)
    rules_by_category: Mapping[int, ComponentRules] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def expected_category_ids(self) -> set[int]:
        return set(self.category_names)

    def rules_for(self, category_id: int) -> ComponentRules:
        return self.rules_by_category.get(category_id, self.default_rules)


@dataclass(frozen=True)
class ComponentBox:
    """Geometry and quality decision for one connected component."""

    category_id: int
    component_id: int
    x: int
    y: int
    width: int
    height: int
    pixel_area: int
    bbox_area: int
    relative_area: float
    fill_ratio: float
    aspect_ratio: float
    touches_border: bool
    iscrowd: int
    reasons: tuple[str, ...]


BASE_RULES = ComponentRules()

POLICIES: dict[str, DatasetPolicy] = {
    "gpcc": DatasetPolicy(
        key="gpcc",
        description=(
            "Binary GPCC masks: connected components are usually compact and "
            "are retained as ordinary boxes unless their geometry is suspicious."
        ),
        category_names={1: "foreground_region"},
        notes=(
            "Border-touching components are retained as truncated objects.",
            "Visual/statistical review previously graded GPCC as a strong candidate.",
        ),
    ),
    "mdc": DatasetPolicy(
        key="mdc",
        description=(
            "Binary MDC masks: ordinary components are retained; large, thin, "
            "or low-fill regions are preserved as crowd/ignore candidates."
        ),
        category_names={1: "foreground_region"},
        notes=(
            "MDC has many border-touching components; they are retained and tagged truncated.",
            "Giant or fused semantic regions cannot be reliably split into instances.",
        ),
    ),
    "luad": DatasetPolicy(
        key="luad",
        description=(
            "Three-class LUAD masks: class 1 is frequently merged or patch-sized "
            "and is therefore crowd-only by default; classes 2/3 use geometric rules."
        ),
        category_names={1: "class_1", 2: "class_2", 3: "class_3"},
        rules_by_category={
            1: ComponentRules(force_crowd=True),
            2: BASE_RULES,
            3: BASE_RULES,
        },
        notes=(
            "Verify the original semantic meaning of class IDs before renaming categories.",
            "Class 1 needs manual splitting, crowd treatment, or a cluster-level target definition.",
        ),
    ),
    "tma": DatasetPolicy(
        key="tma",
        description=(
            "Three-class TMA masks: class 1 uses geometric rules, class 2 is "
            "conditional, and irregular class 3 regions are crowd-only by default."
        ),
        category_names={1: "class_1", 2: "class_2", 3: "class_3"},
        rules_by_category={
            1: BASE_RULES,
            2: BASE_RULES,
            3: ComponentRules(force_crowd=True),
        },
        notes=(
            "Verify the original semantic meaning of class IDs before renaming categories.",
            "Class 3 connected regions are not treated as reliable object instances.",
        ),
    ),
}


def _index_npy_files(directory: Path) -> dict[str, Path]:
    """Index immediate ``.npy`` children by stem and reject ambiguous duplicates."""

    indexed: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.suffix.lower() != ".npy":
            continue
        if path.stem in indexed:
            duplicates.append(path.stem)
        else:
            indexed[path.stem] = path
    if duplicates:
        duplicate_text = ", ".join(sorted(set(duplicates))[:10])
        raise ValueError(
            f"{directory} contains duplicate .npy stems; pairing is ambiguous: "
            f"{duplicate_text}"
        )
    return indexed


def inspect_image_mask_pairs(dataset_dir: str | Path) -> tuple[PairingReport, list[tuple[Path, Path]]]:
    """Validate a dataset root and return all stem-matched image/mask pairs."""

    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {root}")

    images_dir = root / "images"
    masks_dir = root / "masks"
    missing = [str(path) for path in (images_dir, masks_dir) if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "The dataset root must contain both images/ and masks/. Missing: "
            + ", ".join(missing)
        )

    images = _index_npy_files(images_dir)
    masks = _index_npy_files(masks_dir)
    paired_stems = sorted(images.keys() & masks.keys())
    image_only = tuple(sorted(images.keys() - masks.keys()))
    mask_only = tuple(sorted(masks.keys() - images.keys()))
    pairs = [(images[stem], masks[stem]) for stem in paired_stems]
    report = PairingReport(
        dataset_dir=str(root),
        images_dir=str(images_dir),
        masks_dir=str(masks_dir),
        n_images=len(images),
        n_masks=len(masks),
        n_name_matched_pairs=len(pairs),
        images_without_mask=image_only,
        masks_without_image=mask_only,
    )
    return report, pairs


def _load_mask(mask_path: Path) -> np.ndarray:
    """Read a mask and validate that it is a finite 2-D integer label map."""

    raw = np.load(mask_path, allow_pickle=False)
    mask = np.asarray(raw).squeeze()
    if mask.ndim != 2:
        raise ValueError(
            f"Mask must become 2-D after squeeze, got shape {raw.shape}: {mask_path}"
        )
    if not np.issubdtype(mask.dtype, np.number):
        raise TypeError(f"Mask dtype must be numeric, got {mask.dtype}: {mask_path}")
    if not np.isfinite(mask).all():
        raise ValueError(f"Mask contains NaN or Inf: {mask_path}")
    if not np.allclose(mask, np.rint(mask), atol=0.0):
        raise ValueError(f"Mask contains non-integer class values: {mask_path}")

    mask = np.rint(mask).astype(np.int64, copy=False)
    if np.any(mask < 0):
        raise ValueError(f"Mask contains negative class IDs: {mask_path}")
    return mask


def _validate_image_shape(image_path: Path, mask_shape: tuple[int, int]) -> tuple[int, ...]:
    """Read only the NumPy header/data mapping and verify spatial compatibility."""

    image = np.load(image_path, mmap_mode="r", allow_pickle=False)
    image_shape = tuple(int(value) for value in image.shape)
    if image.ndim < 2:
        raise ValueError(f"Image must have at least 2 dimensions: {image_path}")

    compatible = image_shape[:2] == mask_shape
    if image.ndim >= 3:
        compatible = compatible or image_shape[-2:] == mask_shape
    if not compatible:
        raise ValueError(
            f"Image/mask spatial shape mismatch: image={image_shape}, "
            f"mask={mask_shape}, image_path={image_path}"
        )
    return image_shape


def _component_reasons(
    *,
    rules: ComponentRules,
    area: int,
    width: int,
    height: int,
    image_area: int,
    fill_ratio: float,
    aspect_ratio: float,
    touches_border: bool,
) -> tuple[str, ...]:
    """Return all reasons why a component should be treated as crowd/ignore."""

    reasons: list[str] = []
    if rules.force_crowd:
        reasons.append("dataset_class_not_instance_reliable")
    if area < rules.min_area_px:
        reasons.append("area_below_min")
    if min(width, height) < rules.min_side_px:
        reasons.append("side_below_min")
    if area / image_area > rules.max_relative_area:
        reasons.append("relative_area_above_max")
    if aspect_ratio > rules.max_aspect_ratio:
        reasons.append("aspect_ratio_above_max")
    if fill_ratio < rules.min_fill_ratio:
        reasons.append("fill_ratio_below_min")
    if touches_border and rules.touching_border_is_crowd:
        reasons.append("touches_image_border")
    return tuple(reasons)


def mask_to_component_boxes(
    mask: np.ndarray,
    policy: DatasetPolicy,
    *,
    connectivity: int = 8,
) -> list[ComponentBox]:
    """Convert every positive semantic connected component to a candidate box."""

    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    positive_ids = {int(value) for value in np.unique(mask) if int(value) != 0}
    if policy.key != "generic":
        unexpected = positive_ids - policy.expected_category_ids
        if unexpected:
            raise ValueError(
                f"Mask contains class IDs not defined by the {policy.key} policy: "
                f"{sorted(unexpected)}"
            )

    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    height, width = (int(mask.shape[0]), int(mask.shape[1]))
    image_area = height * width
    boxes: list[ComponentBox] = []

    for category_id in sorted(positive_ids):
        labels, n_components = ndimage.label(mask == category_id, structure=structure)
        slices = ndimage.find_objects(labels)
        rules = policy.rules_for(category_id)

        for component_id in range(1, n_components + 1):
            component_slice = slices[component_id - 1]
            if component_slice is None:
                continue
            y_slice, x_slice = component_slice
            x = int(x_slice.start)
            y = int(y_slice.start)
            box_width = int(x_slice.stop - x_slice.start)
            box_height = int(y_slice.stop - y_slice.start)
            local_component = labels[component_slice] == component_id
            area = int(np.count_nonzero(local_component))
            bbox_area = box_width * box_height
            fill_ratio = float(area / bbox_area)
            aspect_ratio = float(
                max(box_width / box_height, box_height / box_width)
            )
            touches_border = bool(
                x == 0
                or y == 0
                or x + box_width == width
                or y + box_height == height
            )
            reasons = _component_reasons(
                rules=rules,
                area=area,
                width=box_width,
                height=box_height,
                image_area=image_area,
                fill_ratio=fill_ratio,
                aspect_ratio=aspect_ratio,
                touches_border=touches_border,
            )
            boxes.append(
                ComponentBox(
                    category_id=category_id,
                    component_id=component_id,
                    x=x,
                    y=y,
                    width=box_width,
                    height=box_height,
                    pixel_area=area,
                    bbox_area=bbox_area,
                    relative_area=float(area / image_area),
                    fill_ratio=fill_ratio,
                    aspect_ratio=aspect_ratio,
                    touches_border=touches_border,
                    iscrowd=int(bool(reasons)),
                    reasons=reasons,
                )
            )
    return boxes


def _infer_policy_key(dataset_dir: Path) -> str:
    """Infer a known policy from all path components."""

    normalized = dataset_dir.as_posix().lower()
    matches = [key for key in POLICIES if key in normalized]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return "generic"
    raise ValueError(
        f"Could not infer one unambiguous dataset policy from {dataset_dir}; "
        f"matches={matches}. Pass --dataset explicitly."
    )


def _generic_policy(category_names: Mapping[int, str] | None = None) -> DatasetPolicy:
    """Build a generic policy when no dataset-specific policy is available."""

    return DatasetPolicy(
        key="generic",
        description=(
            "Generic semantic-mask conversion using only geometry. Review every "
            "category before using the generated boxes for training."
        ),
        category_names=dict(category_names or {}),
        notes=(
            "No dataset-specific validity assumption was made.",
            "Unknown positive class IDs are named class_<id>.",
        ),
    )


def _with_category_names(
    policy: DatasetPolicy,
    category_names: Mapping[int, str] | None,
) -> DatasetPolicy:
    """Return a policy whose category display names include user overrides."""

    if not category_names:
        return policy
    merged = dict(policy.category_names)
    for category_id, name in category_names.items():
        if int(category_id) <= 0:
            raise ValueError("COCO foreground category IDs must be positive")
        if not str(name).strip():
            raise ValueError(f"Empty category name for ID {category_id}")
        merged[int(category_id)] = str(name).strip()
    return DatasetPolicy(
        key=policy.key,
        description=policy.description,
        category_names=merged,
        default_rules=policy.default_rules,
        rules_by_category=policy.rules_by_category,
        notes=policy.notes,
    )


def _atomic_json_dump(payload: Any, output_path: Path) -> None:
    """Write formatted JSON atomically within the destination directory."""

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary_path.replace(output_path)


def _write_audit_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    """Write the component-level audit table."""

    columns = [
        "image_id",
        "annotation_id",
        "stem",
        "image_file",
        "mask_file",
        "category_id",
        "category_name",
        "component_id",
        "x",
        "y",
        "width",
        "height",
        "pixel_area",
        "bbox_area",
        "relative_area",
        "fill_ratio",
        "aspect_ratio",
        "touches_border",
        "iscrowd",
        "decision",
        "reasons",
    ]
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(output_path)


def generate_detection_labels(
    dataset_dir: str | Path,
    *,
    dataset: str = "auto",
    output_dir: str | Path | None = None,
    category_names: Mapping[int, str] | None = None,
    connectivity: int = 8,
    overwrite: bool = False,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Generate COCO candidate boxes and audit files for one dataset directory."""

    root = Path(dataset_dir).expanduser().resolve()
    report, pairs = inspect_image_mask_pairs(root)
    policy_key = _infer_policy_key(root) if dataset == "auto" else dataset.lower()
    if policy_key == "generic":
        policy = _generic_policy(category_names)
    else:
        if policy_key not in POLICIES:
            raise ValueError(
                f"Unknown dataset policy {policy_key!r}; choices are "
                f"{sorted(POLICIES)} plus generic/auto"
            )
        policy = _with_category_names(POLICIES[policy_key], category_names)

    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else root / "detection_annotations" / policy.key
    )
    coco_path = destination / "instances_candidates.json"
    audit_path = destination / "component_audit.csv"
    summary_path = destination / "generation_summary.json"
    existing = [path for path in (coco_path, audit_path, summary_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs. Pass --overwrite if intended: "
            + ", ".join(str(path) for path in existing)
        )
    destination.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).isoformat()
    coco: dict[str, Any] = {
        "info": {
            "description": "Candidate detection boxes derived from semantic masks",
            "version": "1.0",
            "date_created": generated_at,
            "dataset_root": str(root),
            "dataset_policy": policy.key,
            "policy_description": policy.description,
            "policy_notes": list(policy.notes),
            "box_convention": "[x, y, width, height], zero-based, half-open extent",
            "image_storage": "hyperspectral NumPy array; custom loader required",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }
    audit_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    observed_category_ids: set[int] = set()
    decision_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    next_annotation_id = 1

    for pair_index, (image_path, mask_path) in enumerate(pairs, start=1):
        try:
            mask = _load_mask(mask_path)
            image_shape = _validate_image_shape(image_path, tuple(mask.shape))
            boxes = mask_to_component_boxes(
                mask,
                policy,
                connectivity=connectivity,
            )
        except Exception as exc:
            failures.append(
                {
                    "stem": image_path.stem,
                    "image_file": str(image_path),
                    "mask_file": str(mask_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if fail_fast:
                raise
            continue

        image_id = len(coco["images"]) + 1
        relative_image = image_path.relative_to(root).as_posix()
        relative_mask = mask_path.relative_to(root).as_posix()
        coco["images"].append(
            {
                "id": image_id,
                "file_name": relative_image,
                "width": int(mask.shape[1]),
                "height": int(mask.shape[0]),
                "mask_file_name": relative_mask,
                "numpy_shape": list(image_shape),
                "hyperspectral": len(image_shape) >= 3,
            }
        )

        for box in boxes:
            observed_category_ids.add(box.category_id)
            category_name = policy.category_names.get(
                box.category_id, f"class_{box.category_id}"
            )
            decision = "crowd_or_ignore_candidate" if box.iscrowd else "trainable_box"
            decision_counts[decision] += 1
            reason_counts.update(box.reasons)
            coco["annotations"].append(
                {
                    "id": next_annotation_id,
                    "image_id": image_id,
                    "category_id": box.category_id,
                    "bbox": [box.x, box.y, box.width, box.height],
                    "area": box.pixel_area,
                    "iscrowd": box.iscrowd,
                    "segmentation": [],
                    "attributes": {
                        "source": "semantic_connected_component",
                        "component_id_within_class": box.component_id,
                        "bbox_area": box.bbox_area,
                        "fill_ratio": box.fill_ratio,
                        "relative_area": box.relative_area,
                        "aspect_ratio": box.aspect_ratio,
                        "truncated": box.touches_border,
                        "decision": decision,
                        "reasons": list(box.reasons),
                    },
                }
            )
            audit_rows.append(
                {
                    "image_id": image_id,
                    "annotation_id": next_annotation_id,
                    "stem": image_path.stem,
                    "image_file": relative_image,
                    "mask_file": relative_mask,
                    "category_id": box.category_id,
                    "category_name": category_name,
                    "component_id": box.component_id,
                    "x": box.x,
                    "y": box.y,
                    "width": box.width,
                    "height": box.height,
                    "pixel_area": box.pixel_area,
                    "bbox_area": box.bbox_area,
                    "relative_area": f"{box.relative_area:.8f}",
                    "fill_ratio": f"{box.fill_ratio:.8f}",
                    "aspect_ratio": f"{box.aspect_ratio:.8f}",
                    "touches_border": int(box.touches_border),
                    "iscrowd": box.iscrowd,
                    "decision": decision,
                    "reasons": "|".join(box.reasons),
                }
            )
            next_annotation_id += 1

        if pair_index % 100 == 0 or pair_index == len(pairs):
            print(
                f"  processed name-matched pairs: {pair_index}/{len(pairs)} "
                f"(successful={len(coco['images'])}, failed={len(failures)})"
            )

    category_ids = policy.expected_category_ids | observed_category_ids
    if policy.key == "generic" and not category_ids:
        category_ids = set(policy.category_names)
    coco["categories"] = [
        {
            "id": category_id,
            "name": policy.category_names.get(category_id, f"class_{category_id}"),
            "supercategory": "pathology_region",
        }
        for category_id in sorted(category_ids)
    ]

    summary: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "dataset_root": str(root),
        "output_dir": str(destination),
        "policy": {
            "key": policy.key,
            "description": policy.description,
            "category_names": {
                str(key): value for key, value in sorted(policy.category_names.items())
            },
            "default_rules": asdict(policy.default_rules),
            "rules_by_category": {
                str(key): asdict(value)
                for key, value in sorted(policy.rules_by_category.items())
            },
            "notes": list(policy.notes),
        },
        "pairing": asdict(report),
        "processing": {
            "connectivity": connectivity,
            "n_successful_pairs": len(coco["images"]),
            "n_failed_pairs": len(failures),
            "n_images_with_no_boxes": sum(
                1
                for image in coco["images"]
                if not any(
                    annotation["image_id"] == image["id"]
                    for annotation in coco["annotations"]
                )
            ),
            "n_annotations": len(coco["annotations"]),
            "decision_counts": dict(sorted(decision_counts.items())),
            "reason_counts": dict(sorted(reason_counts.items())),
            "observed_category_ids": sorted(observed_category_ids),
        },
        "failures": failures,
        "outputs": {
            "coco_json": str(coco_path),
            "component_audit_csv": str(audit_path),
            "summary_json": str(summary_path),
        },
    }

    _atomic_json_dump(coco, coco_path)
    _write_audit_csv(audit_rows, audit_path)
    _atomic_json_dump(summary, summary_path)
    return summary


def generate_gpcc_detection_labels(
    dataset_dir: str | Path, **kwargs: Any
) -> dict[str, Any]:
    """Generate GPCC candidate boxes from a directory containing images/masks."""

    return generate_detection_labels(dataset_dir, dataset="gpcc", **kwargs)


def generate_mdc_detection_labels(
    dataset_dir: str | Path, **kwargs: Any
) -> dict[str, Any]:
    """Generate MDC candidate boxes from a directory containing images/masks."""

    return generate_detection_labels(dataset_dir, dataset="mdc", **kwargs)


def generate_luad_detection_labels(
    dataset_dir: str | Path, **kwargs: Any
) -> dict[str, Any]:
    """Generate LUAD candidate boxes from a directory containing images/masks."""

    return generate_detection_labels(dataset_dir, dataset="luad", **kwargs)


def generate_tma_detection_labels(
    dataset_dir: str | Path, **kwargs: Any
) -> dict[str, Any]:
    """Generate TMA candidate boxes from a directory containing images/masks."""

    return generate_detection_labels(dataset_dir, dataset="tma", **kwargs)


def _parse_category_names(raw: str | None) -> dict[int, str] | None:
    """Parse category names from an inline JSON object or a JSON file."""

    if raw is None:
        return None
    candidate_path = Path(raw).expanduser()
    if candidate_path.is_file():
        with candidate_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("--category-names-json must be a JSON object")
    return {int(key): str(value) for key, value in payload.items()}


def _print_pairing_summary(summary: Mapping[str, Any]) -> None:
    """Print the final pairing/conversion result for one dataset."""

    pairing = summary["pairing"]
    processing = summary["processing"]
    print("\n" + "=" * 80)
    print(f"Dataset: {summary['dataset_root']}")
    print(f"Policy:  {summary['policy']['key']}")
    print(
        "Files:   "
        f"images={pairing['n_images']}, masks={pairing['n_masks']}, "
        f"name-matched={pairing['n_name_matched_pairs']}"
    )
    print(
        "Read:    "
        f"successful={processing['n_successful_pairs']}, "
        f"failed={processing['n_failed_pairs']}"
    )
    print(
        "Boxes:   "
        f"total={processing['n_annotations']}, "
        f"decisions={processing['decision_counts']}"
    )
    if pairing["images_without_mask"]:
        print(
            f"Warning: {len(pairing['images_without_mask'])} image stems have no mask"
        )
    if pairing["masks_without_image"]:
        print(
            f"Warning: {len(pairing['masks_without_image'])} mask stems have no image"
        )
    print(f"COCO:    {summary['outputs']['coco_json']}")
    print(f"Audit:   {summary['outputs']['component_audit_csv']}")
    print(f"Summary: {summary['outputs']['summary_json']}")
    print("=" * 80)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate auditable COCO detection-box candidates from .npy "
            "semantic segmentation masks."
        )
    )
    parser.add_argument(
        "dataset_dir",
        help="dataset root containing images/ and masks/",
    )
    parser.add_argument(
        "--dataset",
        choices=["auto", "gpcc", "mdc", "luad", "tma", "generic"],
        default="auto",
        help="dataset-specific conversion policy (default: infer from path)",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "output directory (default: "
            "DATASET_ROOT/detection_annotations/POLICY/)"
        ),
    )
    parser.add_argument(
        "--category-names-json",
        help=(
            "JSON object or JSON file mapping positive mask IDs to verified names, "
            'for example \'{"1":"tumor"}\''
        ),
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=[4, 8],
        default=8,
        help="connected-component pixel connectivity (default: 8)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing generated JSON/CSV files",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop at the first unreadable or invalid image/mask pair",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""

    args = build_argument_parser().parse_args(argv)
    try:
        category_names = _parse_category_names(args.category_names_json)
        summary = generate_detection_labels(
            args.dataset_dir,
            dataset=args.dataset,
            output_dir=args.output_dir,
            category_names=category_names,
            connectivity=args.connectivity,
            overwrite=args.overwrite,
            fail_fast=args.fail_fast,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    _print_pairing_summary(summary)
    return 1 if summary["processing"]["n_failed_pairs"] else 0


if __name__ == "__main__":
    sys.exit(main())
