#!/usr/bin/env python3
"""Build the reconstructed official PUAD/LUAD segmentation protocol.

The script intentionally keeps the original scene-level split:

* subject-1..60  -> train, 3,138 centre-balanced 224x224 patches;
* subject-61..70 -> val,     522 centre-balanced 224x224 patches;
* subject-71..100 -> test, 30 complete 1300x1800 scenes.

The paper states that 3,660 train/validation patches are sampled according to
class proportions and a labelled centre pixel, but it does not publish the
train/validation patch totals, alpha values, integer scene quotas, or random
seed.  The defaults below are therefore a deterministic reconstruction: each
split is class-balanced, with 1,046 patches per class in train and 174 per
class in validation.

Only train patches are used to fit the scalar dataset-level Min-Max transform.
The same transform is then applied to train, validation, and complete test
scenes.  Offline NMF is deliberately not run here; run it separately on each
output split after preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from spectral import envi
from tqdm import tqdm


# ============================================================================
# Macro-style user configuration
# ============================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

SOURCE_ROOT = PROJECT_ROOT / "data/original/LUAD_HDR"
OUTPUT_ROOT = PROJECT_ROOT / "data/LUAD_PUAD_official224_centerbalanced_3660"

TRAIN_SCENE_IDS = tuple(range(1, 61))
VAL_SCENE_IDS = tuple(range(61, 71))
TEST_SCENE_IDS = tuple(range(71, 101))

PATCH_SIZE = 224
SAMPLING_SEED = 20260823
TARGET_PATCHES_PER_CLASS = {
    "train": {1: 1046, 2: 1046, 3: 1046},
    "val": {1: 174, 2: 174, 3: 174},
}

# Smoke mode uses real source scenes but much smaller quotas.  It still writes
# one complete test scene, so the exact full-scene materialisation path is tested.
SMOKE_TRAIN_SCENE_IDS = (10,)
SMOKE_VAL_SCENE_IDS = (61,)
SMOKE_TEST_SCENE_IDS = (71,)
SMOKE_TARGET_PATCHES_PER_CLASS = {
    "train": {1: 2, 2: 2, 3: 2},
    "val": {1: 1, 2: 1, 3: 1},
}

NORMALIZATION_METHOD = "dataset_global_minmax"
NORMALIZATION_FIT_SCOPE = "train_patches_only"
NORMALIZATION_CLIP = True
NORMALIZATION_EPS = 1e-12
OUTPUT_IMAGE_DTYPE = np.float32
OUTPUT_MASK_DTYPE = np.uint8
TEST_WRITE_CHUNK_ROWS = 64

# Save one review PNG for every output sample. Train/Val patches are shown at
# native 224x224 resolution; complete Test scenes are resized for manageable
# previews while the underlying .npy data remains full resolution.
SAVE_REVIEW_VISUALIZATIONS = True
REVIEW_DIR_NAME = "review_patch_visualizations"
REVIEW_RGB_WAVELENGTHS_NM = (650.0, 550.0, 473.0)
REVIEW_MASK_ALPHA = 0.42
REVIEW_SEPARATOR_PIXELS = 6
REVIEW_TEST_MAX_PANEL_SIDE = 1200
REVIEW_PNG_COMPRESS_LEVEL = 4

# False protects an existing dataset.  It can be changed here, or overridden
# explicitly with --overwrite on the command line.
OVERWRITE_OUTPUT = False

EXPECTED_BANDS = 40
EXPECTED_SCENE_SHAPE = (1300, 1800, 40)
ENFORCE_EXPECTED_SCENE_SHAPE = True
VALID_CLASSES = (0, 1, 2, 3)
CLASS_NAMES = {
    0: "background",
    1: "tumor_or_red",
    2: "hyperplasia_or_green",
    3: "normal_or_blue",
}
CLASS_COLORS_RGB = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
}


@dataclass(frozen=True)
class RunConfig:
    source_root: Path
    output_root: Path
    train_scene_ids: tuple[int, ...]
    val_scene_ids: tuple[int, ...]
    test_scene_ids: tuple[int, ...]
    target_patches_per_class: dict[str, dict[int, int]]
    patch_size: int
    seed: int
    overwrite: bool
    smoke_test: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_ready(value: Any) -> Any:
    """Convert Path/NumPy/tuple-rich objects into JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def relative_to_output(path: Path, output_root: Path) -> str:
    return path.relative_to(output_root).as_posix()


def scene_stem(scene_id: int) -> str:
    return f"subject-{scene_id}"


def image_paths(source_root: Path, source_split: str, scene_id: int) -> tuple[Path, Path]:
    stem = scene_stem(scene_id)
    base = source_root / source_split / "imgs" / stem
    return base.with_suffix(".hdr"), base.with_suffix(".raw")


def label_path(source_root: Path, source_split: str, scene_id: int) -> Path:
    stem = scene_stem(scene_id)
    if source_split == "Training":
        return source_root / "Training" / "label" / f"{stem}-label.png"
    return source_root / "Testing" / "label_rgb" / f"{stem}.png"


def open_hsi(source_root: Path, source_split: str, scene_id: int):
    hdr_path, raw_path = image_paths(source_root, source_split, scene_id)
    if not hdr_path.is_file() or not raw_path.is_file():
        raise FileNotFoundError(f"missing ENVI pair: {hdr_path}, {raw_path}")
    image = envi.open(str(hdr_path), str(raw_path))
    if len(image.shape) != 3:
        raise ValueError(f"{hdr_path}: expected 3D HSI, got {image.shape}")
    return image, hdr_path, raw_path


def extract_wavelengths(image, hdr_path: Path) -> np.ndarray:
    values = image.metadata.get("wavelength")
    if values is None:
        raise ValueError(f"{hdr_path}: missing wavelength metadata")
    wavelengths = np.asarray([float(value) for value in values], dtype=np.float32)
    if wavelengths.size != int(image.shape[2]):
        raise ValueError(
            f"{hdr_path}: {wavelengths.size} wavelengths for {image.shape[2]} bands"
        )
    if not np.all(np.isfinite(wavelengths)) or not np.all(np.diff(wavelengths) > 0):
        raise ValueError(f"{hdr_path}: wavelengths must be finite and strictly increasing")
    return wavelengths


def rgb_label_to_mask(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected RGB label, got {rgb.shape}")
    channels = rgb.astype(np.int16, copy=False)
    maximum = channels.max(axis=2)
    dominant = channels.argmax(axis=2)
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    foreground = maximum > 127
    mask[foreground] = dominant[foreground].astype(np.uint8) + 1
    return mask


def load_aligned_mask(
    source_root: Path,
    source_split: str,
    scene_id: int,
    target_hw: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    path = label_path(source_root, source_split, scene_id)
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as label_image:
        original_size_wh = label_image.size
        original_hw = (original_size_wh[1], original_size_wh[0])
        resized = original_hw != target_hw
        if resized:
            label_image = label_image.resize(
                (target_hw[1], target_hw[0]), resample=Image.Resampling.NEAREST
            )
        if source_split == "Training":
            raw = np.asarray(label_image)
            if raw.ndim == 3:
                raw = raw[..., 0]
            mask = raw.astype(np.uint8, copy=False)
            label_kind = "grayscale_class_index"
        else:
            raw = np.asarray(label_image.convert("RGB"), dtype=np.uint8)
            mask = rgb_label_to_mask(raw)
            label_kind = "rgb_palette"
    unexpected = sorted(set(np.unique(mask).tolist()) - set(VALID_CLASSES))
    if unexpected:
        raise ValueError(f"{path}: unexpected class values {unexpected}")
    if mask.shape != target_hw:
        raise AssertionError(f"{path}: aligned mask shape {mask.shape}, expected {target_hw}")
    return mask, {
        "source_label": str(path),
        "label_kind": label_kind,
        "label_original_height": int(original_hw[0]),
        "label_original_width": int(original_hw[1]),
        "label_resized": bool(resized),
        "label_aligned_height": int(target_hw[0]),
        "label_aligned_width": int(target_hw[1]),
    }


def crop_geometry(patch_size: int) -> tuple[int, int]:
    before = patch_size // 2
    after = patch_size - before
    return before, after


def valid_center_bounds(height: int, width: int, patch_size: int) -> tuple[int, int, int, int]:
    before, after = crop_geometry(patch_size)
    y_min, y_max = before, height - after
    x_min, x_max = before, width - after
    if y_min > y_max or x_min > x_max:
        raise ValueError(f"patch size {patch_size} exceeds scene {(height, width)}")
    return y_min, y_max, x_min, x_max


def class_counts(mask: np.ndarray) -> np.ndarray:
    return np.bincount(mask.reshape(-1), minlength=4)[:4].astype(np.int64)


def eligible_class_counts(mask: np.ndarray, patch_size: int) -> np.ndarray:
    y_min, y_max, x_min, x_max = valid_center_bounds(*mask.shape, patch_size)
    # y_max/x_max are inclusive legal centre coordinates.
    inner = mask[y_min : y_max + 1, x_min : x_max + 1]
    return class_counts(inner)


def largest_remainder_allocation(raw_values: np.ndarray, target_total: int) -> np.ndarray:
    raw_values = np.asarray(raw_values, dtype=np.float64)
    allocation = np.floor(raw_values).astype(np.int64)
    remaining = int(target_total - allocation.sum())
    if remaining < 0:
        raise ValueError("floored allocation exceeds target")
    fractions = raw_values - allocation
    order = np.argsort(-fractions, kind="stable")
    allocation[order[:remaining]] += 1
    if int(allocation.sum()) != int(target_total):
        raise AssertionError("largest-remainder allocation failed")
    return allocation


def capacity_aware_allocation(
    raw_values: np.ndarray,
    target_total: int,
    capacities: np.ndarray,
) -> np.ndarray:
    """Round alpha*p while respecting each scene's number of legal centres."""
    raw_values = np.asarray(raw_values, dtype=np.float64)
    capacities = np.asarray(capacities, dtype=np.int64)
    if int(capacities.sum()) < int(target_total):
        raise ValueError(
            f"only {int(capacities.sum())} legal centres for target {target_total}"
        )
    allocation = np.minimum(np.floor(raw_values).astype(np.int64), capacities)
    remaining = int(target_total - allocation.sum())
    while remaining > 0:
        available = allocation < capacities
        if not available.any():
            raise RuntimeError("capacity-aware allocation exhausted all legal centres")
        deficits = raw_values - allocation
        deficits[~available] = -np.inf
        allocation[int(np.argmax(deficits))] += 1
        remaining -= 1
    if int(allocation.sum()) != int(target_total) or np.any(allocation > capacities):
        raise AssertionError("capacity-aware allocation failed")
    return allocation


def stable_rng(seed: int, split_name: str, scene_id: int, class_id: int) -> np.random.Generator:
    split_code = {"train": 11, "val": 23}[split_name]
    sequence = np.random.SeedSequence([seed, split_code, scene_id, class_id])
    return np.random.default_rng(sequence)


def inspect_scene_headers(config: RunConfig) -> tuple[dict[int, dict[str, Any]], np.ndarray]:
    records: dict[int, dict[str, Any]] = {}
    reference_wavelengths: np.ndarray | None = None
    groups = (
        ("train", "Training", config.train_scene_ids),
        ("val", "Training", config.val_scene_ids),
        ("test", "Testing", config.test_scene_ids),
    )
    for split_name, source_split, scene_ids in groups:
        for scene_id in tqdm(scene_ids, desc=f"Preflight headers: {split_name}"):
            image, hdr_path, raw_path = open_hsi(config.source_root, source_split, scene_id)
            shape = tuple(int(value) for value in image.shape)
            if ENFORCE_EXPECTED_SCENE_SHAPE and shape != EXPECTED_SCENE_SHAPE:
                raise ValueError(
                    f"{hdr_path}: expected scene shape {EXPECTED_SCENE_SHAPE}, got {shape}"
                )
            if shape[2] != EXPECTED_BANDS:
                raise ValueError(f"{hdr_path}: expected {EXPECTED_BANDS} bands, got {shape}")
            wavelengths = extract_wavelengths(image, hdr_path)
            if reference_wavelengths is None:
                reference_wavelengths = wavelengths
            elif not np.allclose(wavelengths, reference_wavelengths, rtol=0.0, atol=1e-5):
                raise ValueError(f"{hdr_path}: wavelength table differs from the first scene")
            label = label_path(config.source_root, source_split, scene_id)
            if not label.is_file():
                raise FileNotFoundError(label)
            records[scene_id] = {
                "split": split_name,
                "source_split": source_split,
                "scene_id": scene_id,
                "stem": scene_stem(scene_id),
                "source_hdr": str(hdr_path),
                "source_raw": str(raw_path),
                "source_label": str(label),
                "height": shape[0],
                "width": shape[1],
                "bands": shape[2],
            }
    if reference_wavelengths is None:
        raise RuntimeError("no source scenes found")
    return records, reference_wavelengths


def collect_sampling_statistics(
    config: RunConfig,
    header_records: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scene_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for split_name, scene_ids in (
        ("train", config.train_scene_ids),
        ("val", config.val_scene_ids),
    ):
        for scene_id in tqdm(scene_ids, desc=f"Mask statistics: {split_name}"):
            header = header_records[scene_id]
            target_hw = (header["height"], header["width"])
            mask, label_info = load_aligned_mask(
                config.source_root, "Training", scene_id, target_hw
            )
            counts = class_counts(mask)
            eligible = eligible_class_counts(mask, config.patch_size)
            row: dict[str, Any] = {
                "split": split_name,
                "scene_id": scene_id,
                "stem": scene_stem(scene_id),
                "height": target_hw[0],
                "width": target_hw[1],
                "pixels": int(mask.size),
            }
            for class_id in VALID_CLASSES:
                row[f"class_{class_id}_pixels"] = int(counts[class_id])
                row[f"class_{class_id}_proportion"] = float(counts[class_id] / mask.size)
                row[f"class_{class_id}_eligible_centers"] = int(eligible[class_id])
            scene_rows.append(row)
            label_rows.append({"split": split_name, "scene_id": scene_id, **label_info})
    return scene_rows, label_rows


def build_scene_quotas(
    config: RunConfig,
    scene_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    alpha_rows: list[dict[str, Any]] = []
    quota_rows: list[dict[str, Any]] = []
    for split_name in ("train", "val"):
        split_rows = sorted(
            (row for row in scene_rows if row["split"] == split_name),
            key=lambda row: row["scene_id"],
        )
        for class_id in (1, 2, 3):
            target = int(config.target_patches_per_class[split_name][class_id])
            proportions = np.asarray(
                [row[f"class_{class_id}_proportion"] for row in split_rows],
                dtype=np.float64,
            )
            capacities = np.asarray(
                [row[f"class_{class_id}_eligible_centers"] for row in split_rows],
                dtype=np.int64,
            )
            sum_proportions = float(proportions.sum())
            if sum_proportions <= 0:
                raise ValueError(f"{split_name}/class-{class_id}: no labelled pixels")
            alpha = float(target / sum_proportions)
            raw = alpha * proportions
            unconstrained = largest_remainder_allocation(raw, target)
            allocated = capacity_aware_allocation(raw, target, capacities)
            alpha_rows.append(
                {
                    "split": split_name,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "scene_count": len(split_rows),
                    "sum_scene_proportions": sum_proportions,
                    "target_patches": target,
                    "alpha": alpha,
                    "allocated_patches": int(allocated.sum()),
                    "capacity_adjusted_scene_count": int(np.count_nonzero(allocated != unconstrained)),
                }
            )
            for index, scene_row in enumerate(split_rows):
                quota_rows.append(
                    {
                        "split": split_name,
                        "scene_id": int(scene_row["scene_id"]),
                        "stem": scene_row["stem"],
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "proportion": float(proportions[index]),
                        "alpha": alpha,
                        "raw_quota": float(raw[index]),
                        "unconstrained_quota": int(unconstrained[index]),
                        "allocated_quota": int(allocated[index]),
                        "eligible_centers": int(capacities[index]),
                        "capacity_adjusted": bool(allocated[index] != unconstrained[index]),
                    }
                )
    return alpha_rows, quota_rows


def sample_patch_centres(
    config: RunConfig,
    header_records: dict[int, dict[str, Any]],
    quota_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in quota_rows:
        grouped[(row["split"], row["scene_id"])].append(row)

    samples: list[dict[str, Any]] = []
    for (split_name, scene_id), rows in tqdm(
        sorted(grouped.items()), desc="Sample legal patch centres"
    ):
        header = header_records[scene_id]
        mask, _ = load_aligned_mask(
            config.source_root,
            "Training",
            scene_id,
            (header["height"], header["width"]),
        )
        y_min, y_max, x_min, x_max = valid_center_bounds(
            header["height"], header["width"], config.patch_size
        )
        inner = mask[y_min : y_max + 1, x_min : x_max + 1]
        before, after = crop_geometry(config.patch_size)
        for quota_row in sorted(rows, key=lambda row: row["class_id"]):
            class_id = int(quota_row["class_id"])
            quota = int(quota_row["allocated_quota"])
            if quota == 0:
                continue
            positions = np.argwhere(inner == class_id)
            if len(positions) < quota:
                raise AssertionError(
                    f"{split_name}/{scene_stem(scene_id)}/class-{class_id}: "
                    f"quota {quota} exceeds {len(positions)} legal centres"
                )
            rng = stable_rng(config.seed, split_name, scene_id, class_id)
            chosen = rng.choice(len(positions), size=quota, replace=False)
            for position in positions[chosen]:
                center_y = int(position[0] + y_min)
                center_x = int(position[1] + x_min)
                y1, x1 = center_y - before, center_x - before
                y2, x2 = center_y + after, center_x + after
                if int(mask[center_y, center_x]) != class_id:
                    raise AssertionError("sampled centre class mismatch")
                samples.append(
                    {
                        "split": split_name,
                        "scene_id": scene_id,
                        "stem": scene_stem(scene_id),
                        "sampling_class": class_id,
                        "center_y": center_y,
                        "center_x": center_x,
                        "crop_y1": y1,
                        "crop_x1": x1,
                        "crop_y2": y2,
                        "crop_x2": x2,
                        "alpha": float(quota_row["alpha"]),
                        "raw_quota": float(quota_row["raw_quota"]),
                        "unconstrained_quota": int(quota_row["unconstrained_quota"]),
                        "allocated_scene_class_quota": quota,
                        "capacity_adjusted": bool(quota_row["capacity_adjusted"]),
                    }
                )
    samples.sort(
        key=lambda row: (
            row["split"], row["scene_id"], row["sampling_class"],
            row["center_y"], row["center_x"],
        )
    )
    seen: set[tuple[str, int, int, int]] = set()
    for row in samples:
        key = (row["split"], row["scene_id"], row["center_y"], row["center_x"])
        if key in seen:
            raise AssertionError(f"duplicate sampled centre: {key}")
        seen.add(key)
    return samples


def fit_train_minmax(
    config: RunConfig,
    samples: list[dict[str, Any]],
) -> tuple[float, float, int]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        if row["split"] == "train":
            grouped[row["scene_id"]].append(row)
    global_min = math.inf
    global_max = -math.inf
    value_count = 0
    for scene_id, rows in tqdm(sorted(grouped.items()), desc="Fit Train-only Min-Max"):
        image, _, _ = open_hsi(config.source_root, "Training", scene_id)
        cube = image.open_memmap()
        for row in rows:
            patch = np.asarray(
                cube[row["crop_y1"] : row["crop_y2"], row["crop_x1"] : row["crop_x2"], :]
            )
            if patch.shape != (config.patch_size, config.patch_size, image.shape[2]):
                raise AssertionError(f"unexpected patch shape {patch.shape}")
            if not np.all(np.isfinite(patch)):
                raise ValueError(f"non-finite intensity in {scene_stem(scene_id)}")
            global_min = min(global_min, float(patch.min()))
            global_max = max(global_max, float(patch.max()))
            value_count += int(patch.size)
    if not np.isfinite(global_min) or not np.isfinite(global_max):
        raise RuntimeError("failed to fit normalization parameters")
    if global_max - global_min <= NORMALIZATION_EPS:
        raise ValueError(f"degenerate Min-Max range: {global_min}, {global_max}")
    return global_min, global_max, value_count


def normalize_array(array: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    normalized = (np.asarray(array, dtype=np.float32) - minimum) / (maximum - minimum)
    if NORMALIZATION_CLIP:
        np.clip(normalized, 0.0, 1.0, out=normalized)
    return normalized.astype(OUTPUT_IMAGE_DTYPE, copy=False)


def review_rgb_band_indices(wavelengths: np.ndarray) -> tuple[int, int, int]:
    wavelengths = np.asarray(wavelengths, dtype=np.float32).reshape(-1)
    return tuple(
        int(np.argmin(np.abs(wavelengths - target)))
        for target in REVIEW_RGB_WAVELENGTHS_NM
    )


def robust_uint8(channel: np.ndarray) -> np.ndarray:
    values = np.asarray(channel, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    lower, upper = np.percentile(finite, (1.0, 99.0))
    if float(upper - lower) <= 1e-8:
        lower, upper = float(finite.min()), float(finite.max())
    if float(upper - lower) <= 1e-8:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def pseudo_rgb_uint8(cube_hws: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    indices = review_rgb_band_indices(wavelengths)
    return np.stack(
        [robust_uint8(cube_hws[..., band_index]) for band_index in indices],
        axis=-1,
    )


def mask_overlay_uint8(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color_mask = np.zeros_like(rgb, dtype=np.uint8)
    for class_id, color in CLASS_COLORS_RGB.items():
        color_mask[mask == class_id] = np.asarray(color, dtype=np.uint8)
    overlay = rgb.astype(np.float32).copy()
    foreground = mask != 0
    overlay[foreground] = (
        (1.0 - REVIEW_MASK_ALPHA) * overlay[foreground]
        + REVIEW_MASK_ALPHA * color_mask[foreground].astype(np.float32)
    )
    return np.clip(np.rint(overlay), 0, 255).astype(np.uint8)


def save_review_visualization(
    cube_hws: np.ndarray,
    mask: np.ndarray,
    wavelengths: np.ndarray,
    output_path: Path,
    max_panel_side: int | None = None,
) -> None:
    """Save side-by-side pseudo RGB and foreground-mask overlay panels."""
    rgb_image = Image.fromarray(pseudo_rgb_uint8(cube_hws, wavelengths), mode="RGB")
    overlay_image = Image.fromarray(
        mask_overlay_uint8(np.asarray(rgb_image), mask), mode="RGB"
    )
    if max_panel_side is not None and max(rgb_image.size) > max_panel_side:
        scale = max_panel_side / max(rgb_image.size)
        resized_size = (
            max(1, int(round(rgb_image.width * scale))),
            max(1, int(round(rgb_image.height * scale))),
        )
        rgb_image = rgb_image.resize(resized_size, resample=Image.Resampling.BILINEAR)
        overlay_image = overlay_image.resize(
            resized_size, resample=Image.Resampling.BILINEAR
        )
    canvas = Image.new(
        "RGB",
        (rgb_image.width * 2 + REVIEW_SEPARATOR_PIXELS, rgb_image.height),
        color=(255, 255, 255),
    )
    canvas.paste(rgb_image, (0, 0))
    canvas.paste(overlay_image, (rgb_image.width + REVIEW_SEPARATOR_PIXELS, 0))
    canvas.save(output_path, format="PNG", compress_level=REVIEW_PNG_COMPRESS_LEVEL)


def prepare_output_root(config: RunConfig) -> Path:
    output = config.output_root.resolve()
    protected = {
        PROJECT_ROOT.resolve(),
        config.source_root.resolve(),
        (PROJECT_ROOT / "data").resolve(),
    }
    if output in protected:
        raise ValueError(f"unsafe output root: {output}")
    if output.exists():
        if not config.overwrite:
            raise FileExistsError(
                f"output exists: {output}; choose another path or pass --overwrite"
            )
        shutil.rmtree(output)
    for split_name in ("train", "val", "test"):
        (output / split_name / "images").mkdir(parents=True, exist_ok=False)
        (output / split_name / "masks").mkdir(parents=True, exist_ok=False)
        if SAVE_REVIEW_VISUALIZATIONS:
            (output / split_name / REVIEW_DIR_NAME).mkdir(parents=True, exist_ok=False)
    (output / "PREPROCESSING_INCOMPLETE").write_text(utc_now() + "\n", encoding="utf-8")
    return output


def patch_filename(row: dict[str, Any]) -> str:
    return (
        f"{row['stem']}_y{int(row['center_y']):05d}_x{int(row['center_x']):05d}"
        f"_c{int(row['sampling_class'])}"
    )


def materialize_patch_splits(
    config: RunConfig,
    output_root: Path,
    samples: list[dict[str, Any]],
    wavelengths: np.ndarray,
    minimum: float,
    maximum: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, tuple[float, float]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        grouped[(row["split"], row["scene_id"])].append(row)
    manifests: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    observed = {"train": [math.inf, -math.inf], "val": [math.inf, -math.inf]}
    for (split_name, scene_id), rows in tqdm(
        sorted(grouped.items()), desc="Write Train/Val patches"
    ):
        image, hdr_path, raw_path = open_hsi(config.source_root, "Training", scene_id)
        cube = image.open_memmap()
        mask, label_info = load_aligned_mask(
            config.source_root, "Training", scene_id, (image.shape[0], image.shape[1])
        )
        for row in rows:
            image_patch = np.asarray(
                cube[row["crop_y1"] : row["crop_y2"], row["crop_x1"] : row["crop_x2"], :]
            )
            mask_patch = mask[
                row["crop_y1"] : row["crop_y2"], row["crop_x1"] : row["crop_x2"]
            ]
            normalized = normalize_array(image_patch, minimum, maximum)
            counts = class_counts(mask_patch)
            base = patch_filename(row)
            image_path = output_root / split_name / "images" / f"{base}.npy"
            mask_path = output_root / split_name / "masks" / f"{base}.npy"
            np.save(image_path, normalized, allow_pickle=False)
            np.save(mask_path, mask_patch.astype(OUTPUT_MASK_DTYPE), allow_pickle=False)
            review_path: Path | None = None
            if SAVE_REVIEW_VISUALIZATIONS:
                review_path = (
                    output_root / split_name / REVIEW_DIR_NAME / f"{base}.png"
                )
                save_review_visualization(
                    normalized, mask_patch, wavelengths, review_path
                )
            observed[split_name][0] = min(observed[split_name][0], float(normalized.min()))
            observed[split_name][1] = max(observed[split_name][1], float(normalized.max()))
            manifest_row = {
                "stem": base,
                "split": split_name,
                "scene_id": scene_id,
                "source_hdr": str(hdr_path),
                "source_raw": str(raw_path),
                "source_label": label_info["source_label"],
                "image_file": relative_to_output(image_path, output_root),
                "mask_file": relative_to_output(mask_path, output_root),
                "review_visualization_file": (
                    relative_to_output(review_path, output_root)
                    if review_path is not None else ""
                ),
                **{key: row[key] for key in (
                    "sampling_class", "center_y", "center_x", "crop_y1", "crop_x1",
                    "crop_y2", "crop_x2", "alpha", "raw_quota",
                    "unconstrained_quota", "allocated_scene_class_quota",
                    "capacity_adjusted",
                )},
                "center_label": int(mask[row["center_y"], row["center_x"]]),
                "height": config.patch_size,
                "width": config.patch_size,
                "bands": int(image.shape[2]),
                "image_dtype": str(normalized.dtype),
                "mask_dtype": str(mask_patch.astype(OUTPUT_MASK_DTYPE).dtype),
                "image_min": float(normalized.min()),
                "image_max": float(normalized.max()),
            }
            for class_id in VALID_CLASSES:
                manifest_row[f"class_{class_id}_pixels"] = int(counts[class_id])
                manifest_row[f"class_{class_id}_fraction"] = float(
                    counts[class_id] / mask_patch.size
                )
            manifests[split_name].append(manifest_row)
    return manifests, {
        key: (float(value[0]), float(value[1])) for key, value in observed.items()
    }


def write_normalized_full_scene(
    cube,
    output_path: Path,
    minimum: float,
    maximum: float,
) -> tuple[float, float]:
    height, width, bands = (int(value) for value in cube.shape)
    destination = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=OUTPUT_IMAGE_DTYPE,
        shape=(height, width, bands),
    )
    observed_min, observed_max = math.inf, -math.inf
    for y1 in range(0, height, TEST_WRITE_CHUNK_ROWS):
        y2 = min(height, y1 + TEST_WRITE_CHUNK_ROWS)
        source_chunk = np.asarray(cube[y1:y2, :, :])
        if not np.all(np.isfinite(source_chunk)):
            raise ValueError(f"non-finite intensity while writing {output_path}")
        chunk = normalize_array(source_chunk, minimum, maximum)
        destination[y1:y2, :, :] = chunk
        observed_min = min(observed_min, float(chunk.min()))
        observed_max = max(observed_max, float(chunk.max()))
    destination.flush()
    del destination
    return float(observed_min), float(observed_max)


def materialize_test_split(
    config: RunConfig,
    output_root: Path,
    wavelengths: np.ndarray,
    minimum: float,
    maximum: float,
) -> tuple[list[dict[str, Any]], tuple[float, float]]:
    manifest: list[dict[str, Any]] = []
    overall_min, overall_max = math.inf, -math.inf
    for scene_id in tqdm(config.test_scene_ids, desc="Write complete Test scenes"):
        image, hdr_path, raw_path = open_hsi(config.source_root, "Testing", scene_id)
        cube = image.open_memmap()
        target_hw = (int(image.shape[0]), int(image.shape[1]))
        mask, label_info = load_aligned_mask(
            config.source_root, "Testing", scene_id, target_hw
        )
        stem = scene_stem(scene_id)
        image_path = output_root / "test" / "images" / f"{stem}.npy"
        mask_path = output_root / "test" / "masks" / f"{stem}.npy"
        image_min, image_max = write_normalized_full_scene(
            cube, image_path, minimum, maximum
        )
        np.save(mask_path, mask.astype(OUTPUT_MASK_DTYPE), allow_pickle=False)
        review_path: Path | None = None
        if SAVE_REVIEW_VISUALIZATIONS:
            review_path = output_root / "test" / REVIEW_DIR_NAME / f"{stem}.png"
            normalized_cube = np.load(image_path, mmap_mode="r")
            save_review_visualization(
                normalized_cube,
                mask,
                wavelengths,
                review_path,
                max_panel_side=REVIEW_TEST_MAX_PANEL_SIDE,
            )
        overall_min = min(overall_min, image_min)
        overall_max = max(overall_max, image_max)
        counts = class_counts(mask)
        row: dict[str, Any] = {
            "stem": stem,
            "split": "test",
            "scene_id": scene_id,
            "source_hdr": str(hdr_path),
            "source_raw": str(raw_path),
            **label_info,
            "image_file": relative_to_output(image_path, output_root),
            "mask_file": relative_to_output(mask_path, output_root),
            "review_visualization_file": (
                relative_to_output(review_path, output_root)
                if review_path is not None else ""
            ),
            "height": target_hw[0],
            "width": target_hw[1],
            "bands": int(image.shape[2]),
            "image_dtype": np.dtype(OUTPUT_IMAGE_DTYPE).name,
            "mask_dtype": np.dtype(OUTPUT_MASK_DTYPE).name,
            "image_min": image_min,
            "image_max": image_max,
        }
        for class_id in VALID_CLASSES:
            row[f"class_{class_id}_pixels"] = int(counts[class_id])
            row[f"class_{class_id}_fraction"] = float(counts[class_id] / mask.size)
        manifest.append(row)
    return manifest, (float(overall_min), float(overall_max))


def validate_outputs(
    config: RunConfig,
    output_root: Path,
    manifests: dict[str, list[dict[str, Any]]],
    wavelengths: np.ndarray,
) -> dict[str, Any]:
    expected_counts = {
        "train": sum(config.target_patches_per_class["train"].values()),
        "val": sum(config.target_patches_per_class["val"].values()),
        "test": len(config.test_scene_ids),
    }
    split_reports: dict[str, Any] = {}
    for split_name in ("train", "val", "test"):
        image_dir = output_root / split_name / "images"
        mask_dir = output_root / split_name / "masks"
        image_stems = {path.stem for path in image_dir.glob("*.npy")}
        mask_stems = {path.stem for path in mask_dir.glob("*.npy")}
        if image_stems != mask_stems:
            raise AssertionError(f"{split_name}: image/mask stem mismatch")
        if len(image_stems) != expected_counts[split_name]:
            raise AssertionError(
                f"{split_name}: got {len(image_stems)} pairs, expected {expected_counts[split_name]}"
            )
        if len(manifests[split_name]) != expected_counts[split_name]:
            raise AssertionError(f"{split_name}: manifest count mismatch")
        if SAVE_REVIEW_VISUALIZATIONS:
            review_dir = output_root / split_name / REVIEW_DIR_NAME
            review_stems = {path.stem for path in review_dir.glob("*.png")}
            if review_stems != image_stems:
                raise AssertionError(
                    f"{split_name}: review PNG stems do not match image stems"
                )
        split_wavelengths = np.load(output_root / split_name / "wavelengths.npy")
        if not np.array_equal(split_wavelengths, wavelengths):
            raise AssertionError(f"{split_name}: wavelength file mismatch")
        for row in manifests[split_name]:
            image = np.load(output_root / row["image_file"], mmap_mode="r")
            mask = np.load(output_root / row["mask_file"], mmap_mode="r")
            expected_image_shape = (
                (config.patch_size, config.patch_size, EXPECTED_BANDS)
                if split_name != "test"
                else (row["height"], row["width"], EXPECTED_BANDS)
            )
            expected_mask_shape = expected_image_shape[:2]
            if image.shape != expected_image_shape or mask.shape != expected_mask_shape:
                raise AssertionError(
                    f"{row['stem']}: shapes {image.shape}/{mask.shape}, "
                    f"expected {expected_image_shape}/{expected_mask_shape}"
                )
            if image.dtype != np.dtype(OUTPUT_IMAGE_DTYPE):
                raise AssertionError(f"{row['stem']}: unexpected image dtype {image.dtype}")
            if mask.dtype != np.dtype(OUTPUT_MASK_DTYPE):
                raise AssertionError(f"{row['stem']}: unexpected mask dtype {mask.dtype}")
            if split_name != "test" and row["center_label"] != row["sampling_class"]:
                raise AssertionError(f"{row['stem']}: centre-label mismatch")
            if SAVE_REVIEW_VISUALIZATIONS:
                review_path = output_root / row["review_visualization_file"]
                if not review_path.is_file():
                    raise AssertionError(f"{row['stem']}: missing review visualization")
        split_reports[split_name] = {
            "expected_pairs": expected_counts[split_name],
            "actual_pairs": len(image_stems),
            "image_mask_pairing": "ok",
            "manifest_count": len(manifests[split_name]),
            "wavelengths": "ok",
            "shape_and_dtype_headers": "ok",
            "review_visualizations": (
                len(image_stems) if SAVE_REVIEW_VISUALIZATIONS else 0
            ),
        }
    return {"status": "ok", "splits": split_reports}


def preprocessing_config_payload(config: RunConfig) -> dict[str, Any]:
    return {
        "generated_at": utc_now(),
        "script": str(SCRIPT_PATH),
        "protocol_name": "LUAD_PUAD_official224_centerbalanced_3660",
        "protocol_status": "deterministic reconstruction of incompletely published sampling details",
        "source_root": str(config.source_root),
        "output_root": str(config.output_root),
        "scene_splits": {
            "train": config.train_scene_ids,
            "val": config.val_scene_ids,
            "test": config.test_scene_ids,
        },
        "patch_size": config.patch_size,
        "sampling_seed": config.seed,
        "target_patches_per_class": config.target_patches_per_class,
        "sampling": {
            "eligible_center_rule": "mask[center_y,center_x]==class and complete patch inside scene",
            "replacement": False,
            "quota_formula": "raw_B_sc = alpha_dc * class_pixels_sc / scene_pixels_s",
            "integer_allocation": "capacity-aware largest-remainder-style deterministic allocation",
            "mask_output": "complete four-class patch mask; cN in filename is sampling stratum only",
        },
        "normalization": {
            "method": NORMALIZATION_METHOD,
            "fit_scope": NORMALIZATION_FIT_SCOPE,
            "clip": NORMALIZATION_CLIP,
            "output_dtype": np.dtype(OUTPUT_IMAGE_DTYPE).name,
        },
        "review_visualizations": {
            "enabled": SAVE_REVIEW_VISUALIZATIONS,
            "directory_name": REVIEW_DIR_NAME,
            "layout": "pseudo RGB | foreground mask overlay",
            "target_wavelengths_nm": REVIEW_RGB_WAVELENGTHS_NM,
            "mask_alpha": REVIEW_MASK_ALPHA,
            "test_max_panel_side": REVIEW_TEST_MAX_PANEL_SIDE,
        },
        "class_names": CLASS_NAMES,
        "class_colors_rgb": CLASS_COLORS_RGB,
        "expected_bands": EXPECTED_BANDS,
        "expected_source_scene_shape": EXPECTED_SCENE_SHAPE,
        "enforce_expected_source_scene_shape": ENFORCE_EXPECTED_SCENE_SHAPE,
        "test_output": "complete source scenes; tiled model inference is required downstream",
        "offline_nmf_included": False,
        "smoke_test": config.smoke_test,
    }


def run(config: RunConfig) -> Path:
    if config.patch_size <= 0:
        raise ValueError("patch size must be positive")
    for split_name in ("train", "val"):
        if set(config.target_patches_per_class[split_name]) != {1, 2, 3}:
            raise ValueError(f"{split_name}: targets must contain classes 1,2,3")
        if any(value <= 0 for value in config.target_patches_per_class[split_name].values()):
            raise ValueError(f"{split_name}: all class targets must be positive")

    print("=" * 80)
    print("LUAD/PUAD official-224 centre-balanced preprocessing")
    print(f"Source:       {config.source_root}")
    print(f"Output:       {config.output_root}")
    print(f"Patch size:   {config.patch_size}")
    print(f"Seed:         {config.seed}")
    print(f"Smoke test:   {config.smoke_test}")
    print(f"Train scenes: {config.train_scene_ids}")
    print(f"Val scenes:   {config.val_scene_ids}")
    print(f"Test scenes:  {config.test_scene_ids}")
    print(f"Targets:      {config.target_patches_per_class}")
    print("=" * 80)

    header_records, wavelengths = inspect_scene_headers(config)
    scene_rows, label_rows = collect_sampling_statistics(config, header_records)
    alpha_rows, quota_rows = build_scene_quotas(config, scene_rows)
    samples = sample_patch_centres(config, header_records, quota_rows)

    expected_train = sum(config.target_patches_per_class["train"].values())
    expected_val = sum(config.target_patches_per_class["val"].values())
    actual_train = sum(row["split"] == "train" for row in samples)
    actual_val = sum(row["split"] == "val" for row in samples)
    if (actual_train, actual_val) != (expected_train, expected_val):
        raise AssertionError(
            f"sample totals {(actual_train, actual_val)} != {(expected_train, expected_val)}"
        )

    minimum, maximum, fit_value_count = fit_train_minmax(config, samples)
    print(
        f"Train-only Min-Max: min={minimum:.8g}, max={maximum:.8g}, "
        f"values={fit_value_count:,}"
    )

    output_root = prepare_output_root(config)
    write_json(output_root / "preprocessing_config.json", preprocessing_config_payload(config))
    np.save(output_root / "wavelengths.npy", wavelengths.astype(np.float32), allow_pickle=False)
    for split_name in ("train", "val", "test"):
        np.save(
            output_root / split_name / "wavelengths.npy",
            wavelengths.astype(np.float32),
            allow_pickle=False,
        )

    patch_manifests, patch_ranges = materialize_patch_splits(
        config, output_root, samples, wavelengths, minimum, maximum
    )
    test_manifest, test_range = materialize_test_split(
        config, output_root, wavelengths, minimum, maximum
    )
    manifests = {
        "train": patch_manifests["train"],
        "val": patch_manifests["val"],
        "test": test_manifest,
    }
    for split_name, rows in manifests.items():
        write_csv(output_root / split_name / "manifest.csv", rows)

    normalization_payload = {
        "method": NORMALIZATION_METHOD,
        "fit_scope": NORMALIZATION_FIT_SCOPE,
        "global_min": minimum,
        "global_max": maximum,
        "denominator": maximum - minimum,
        "formula": "clip((x - global_min) / (global_max - global_min), 0, 1)",
        "clip": NORMALIZATION_CLIP,
        "fit_value_count": fit_value_count,
        "output_dtype": np.dtype(OUTPUT_IMAGE_DTYPE).name,
        "observed_output_ranges": {
            "train": patch_ranges["train"],
            "val": patch_ranges["val"],
            "test": test_range,
        },
    }
    write_json(output_root / "normalization_params.json", normalization_payload)

    split_manifest = {
        "generated_at": utc_now(),
        "split_unit": "complete source scene",
        "train": {
            "source_scene_ids": config.train_scene_ids,
            "source_scene_count": len(config.train_scene_ids),
            "output_type": "224x224 center-balanced patches",
            "output_count": len(manifests["train"]),
        },
        "val": {
            "source_scene_ids": config.val_scene_ids,
            "source_scene_count": len(config.val_scene_ids),
            "output_type": "224x224 center-balanced patches",
            "output_count": len(manifests["val"]),
        },
        "test": {
            "source_scene_ids": config.test_scene_ids,
            "source_scene_count": len(config.test_scene_ids),
            "output_type": "complete HSI scenes",
            "output_count": len(manifests["test"]),
        },
    }
    write_json(output_root / "split_manifest.json", split_manifest)

    validation = validate_outputs(config, output_root, manifests, wavelengths)
    sampling_report = {
        "generated_at": utc_now(),
        "alpha": alpha_rows,
        "scene_class_quotas": quota_rows,
        "scene_statistics": scene_rows,
        "label_alignment": label_rows,
        "sample_totals": {"train": actual_train, "val": actual_val},
        "capacity_adjustments": [row for row in quota_rows if row["capacity_adjusted"]],
        "validation": validation,
    }
    write_json(output_root / "sampling_report.json", sampling_report)
    write_csv(output_root / "sampling_scene_class_quotas.csv", quota_rows)

    incomplete = output_root / "PREPROCESSING_INCOMPLETE"
    incomplete.unlink()
    print("=" * 80)
    print("Preprocessing completed and validated")
    print(f"Output: {output_root}")
    print(f"Train/Val/Test: {actual_train}/{actual_val}/{len(test_manifest)}")
    print(f"Wavelengths: {wavelengths.size}, {wavelengths[0]:.3f}-{wavelengths[-1]:.3f} nm")
    print("Offline NMF has not been run; generate one cache directory per split next.")
    print("=" * 80)
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the LUAD/PUAD official-224 centre-balanced dataset"
    )
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=SAMPLING_SEED)
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE_OUTPUT)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="use one real scene per split and tiny class-balanced quotas",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        train_ids = SMOKE_TRAIN_SCENE_IDS
        val_ids = SMOKE_VAL_SCENE_IDS
        test_ids = SMOKE_TEST_SCENE_IDS
        targets = SMOKE_TARGET_PATCHES_PER_CLASS
    else:
        train_ids = TRAIN_SCENE_IDS
        val_ids = VAL_SCENE_IDS
        test_ids = TEST_SCENE_IDS
        targets = TARGET_PATCHES_PER_CLASS
    config = RunConfig(
        source_root=args.source_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        train_scene_ids=tuple(train_ids),
        val_scene_ids=tuple(val_ids),
        test_scene_ids=tuple(test_ids),
        target_patches_per_class={
            split_name: {int(key): int(value) for key, value in split_targets.items()}
            for split_name, split_targets in targets.items()
        },
        patch_size=PATCH_SIZE,
        seed=args.seed,
        overwrite=bool(args.overwrite),
        smoke_test=bool(args.smoke_test),
    )
    run(config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; PREPROCESSING_INCOMPLETE is retained if output began.", file=sys.stderr)
        raise
