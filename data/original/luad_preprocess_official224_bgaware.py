#!/usr/bin/env python3
"""Build a background-aware PUAD segmentation dataset.

This is an additive successor to ``luad_preprocess_official224_centerbalanced.py``;
the historical 3,660-patch reconstruction is not modified.  Subject-level
splits and foreground-centred patches are retained, while deterministic pure
and context-background crops are added.  Complete validation scenes are also
materialised so fine-tuning can select checkpoints under the same spatial
distribution used by complete-scene testing.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

import luad_preprocess_official224_centerbalanced as base


# =============================================================================
# User-editable defaults
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

SOURCE_ROOT = PROJECT_ROOT / "data/original/LUAD_HDR"
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data/LUAD_PUAD_official224_bgaware_fg3138_bg1569_fullsceneval"
)

TRAIN_SCENE_IDS = tuple(range(1, 61))
VAL_SCENE_IDS = tuple(range(61, 71))
TEST_SCENE_IDS = tuple(range(71, 101))

PATCH_SIZE = 224
SAMPLING_SEED = 20260914

# Preserve every foreground-centred crop from the historical reconstruction.
FOREGROUND_PATCHES_PER_CLASS = {
    "train": {1: 1046, 2: 1046, 3: 1046},
    "val": {1: 174, 2: 174, 3: 174},
}

# Add one background-centred crop per two foreground-centred crops.
BACKGROUND_PATCH_COUNTS = {"train": 1569, "val": 261}
BACKGROUND_PURE_RATIO = 0.70
PURE_BG_MAX_FOREGROUND_FRACTION = 0.01
CONTEXT_BG_MAX_FOREGROUND_FRACTION = 0.25
BACKGROUND_MIN_CENTER_DISTANCE = PATCH_SIZE // 2

NORMALIZATION_FIT_SCOPE = "complete_train_scenes"
NORMALIZATION_CHUNK_ROWS = 64
OVERWRITE_OUTPUT = False

SMOKE_TRAIN_SCENE_IDS = (10,)
SMOKE_VAL_SCENE_IDS = (61,)
SMOKE_TEST_SCENE_IDS = (71,)
SMOKE_FOREGROUND_PATCHES_PER_CLASS = {
    "train": {1: 2, 2: 2, 3: 2},
    "val": {1: 1, 2: 1, 3: 1},
}
SMOKE_BACKGROUND_PATCH_COUNTS = {"train": 3, "val": 2}


@dataclass(frozen=True)
class RunConfig:
    source_root: Path
    output_root: Path
    train_scene_ids: tuple[int, ...]
    val_scene_ids: tuple[int, ...]
    test_scene_ids: tuple[int, ...]
    target_patches_per_class: dict[str, dict[int, int]]
    background_patch_counts: dict[str, int]
    patch_size: int
    seed: int
    pure_background_ratio: float
    pure_bg_max_foreground_fraction: float
    context_bg_max_foreground_fraction: float
    background_min_center_distance: int
    overwrite: bool
    smoke_test: bool


def _window_foreground_counts(foreground: np.ndarray, patch_size: int) -> np.ndarray:
    """Return the foreground count for every legal top-left patch position."""
    values = np.asarray(foreground, dtype=np.int64)
    integral = np.pad(values, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    p = int(patch_size)
    return (
        integral[p:, p:]
        - integral[:-p, p:]
        - integral[p:, :-p]
        + integral[:-p, :-p]
    )


def _background_candidate_positions(
    mask: np.ndarray,
    patch_size: int,
    pure_max: float,
    context_max: float,
) -> dict[str, np.ndarray]:
    before, after = base.crop_geometry(patch_size)
    center_labels = mask[
        before : mask.shape[0] - after + 1,
        before : mask.shape[1] - after + 1,
    ]
    foreground_counts = _window_foreground_counts(mask > 0, patch_size)
    if center_labels.shape != foreground_counts.shape:
        raise AssertionError(
            f"center/count geometry mismatch: {center_labels.shape} vs "
            f"{foreground_counts.shape}"
        )
    fractions = foreground_counts.astype(np.float64) / float(patch_size**2)
    is_background_center = center_labels == 0
    selectors = {
        "background_pure": is_background_center & (fractions <= pure_max),
        "background_context": (
            is_background_center
            & (fractions > pure_max)
            & (fractions <= context_max)
        ),
    }
    result: dict[str, np.ndarray] = {}
    offset = np.asarray([before, before], dtype=np.int64)
    for stratum, selector in selectors.items():
        result[stratum] = np.argwhere(selector).astype(np.int64) + offset
    return result


def _allocate_quotas(capacities: np.ndarray, target: int) -> np.ndarray:
    capacities = np.asarray(capacities, dtype=np.int64)
    if target == 0:
        return np.zeros_like(capacities)
    if int(capacities.sum()) < target:
        raise ValueError(
            f"only {int(capacities.sum())} background centres for target {target}"
        )
    # Square-root capacity weighting retains area information without allowing
    # the largest mostly-empty scenes to dominate the background pool.
    weights = np.sqrt(capacities.astype(np.float64))
    raw = target * weights / weights.sum()
    return base.capacity_aware_allocation(raw, target, capacities)


def _select_spaced_positions(
    positions: np.ndarray,
    quota: int,
    rng: np.random.Generator,
    occupied: list[tuple[int, int]],
    min_distance: int,
) -> tuple[list[tuple[int, int]], bool]:
    if quota <= 0:
        return [], False
    if len(positions) < quota:
        raise ValueError(f"candidate count {len(positions)} is below quota {quota}")
    trial_count = min(len(positions), max(10_000, quota * 500))
    trial_indices = rng.choice(len(positions), size=trial_count, replace=False)
    selected: list[tuple[int, int]] = []
    minimum_squared = int(min_distance) ** 2
    for index in trial_indices:
        point = tuple(int(value) for value in positions[int(index)])
        if all(
            (point[0] - other[0]) ** 2 + (point[1] - other[1]) ** 2
            >= minimum_squared
            for other in (*occupied, *selected)
        ):
            selected.append(point)
            if len(selected) == quota:
                return selected, False

    # Very small/fragmented candidate regions may not support the requested
    # spacing. Fill deterministically without duplicate centres and report it.
    used = set(occupied) | set(selected)
    fallback_indices = rng.permutation(len(positions))
    for index in fallback_indices:
        point = tuple(int(value) for value in positions[int(index)])
        if point not in used:
            selected.append(point)
            used.add(point)
            if len(selected) == quota:
                return selected, True
    raise RuntimeError(f"could not select {quota} unique positions")


def sample_background_centres(
    config: RunConfig,
    header_records: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    quota_report: list[dict[str, Any]] = []
    before, after = base.crop_geometry(config.patch_size)
    for split_name, scene_ids in (
        ("train", config.train_scene_ids),
        ("val", config.val_scene_ids),
    ):
        total = int(config.background_patch_counts[split_name])
        pure_total = int(round(total * config.pure_background_ratio))
        stratum_totals = {
            "background_pure": pure_total,
            "background_context": total - pure_total,
        }
        candidates_by_scene: dict[int, dict[str, np.ndarray]] = {}
        for scene_id in tqdm(scene_ids, desc=f"Background candidates: {split_name}"):
            header = header_records[scene_id]
            mask, _ = base.load_aligned_mask(
                config.source_root,
                "Training",
                scene_id,
                (header["height"], header["width"]),
            )
            candidates_by_scene[scene_id] = _background_candidate_positions(
                mask,
                config.patch_size,
                config.pure_bg_max_foreground_fraction,
                config.context_bg_max_foreground_fraction,
            )

        quotas_by_stratum: dict[str, np.ndarray] = {}
        for stratum, target in stratum_totals.items():
            capacities = np.asarray(
                [len(candidates_by_scene[scene_id][stratum]) for scene_id in scene_ids],
                dtype=np.int64,
            )
            quotas = _allocate_quotas(capacities, target)
            quotas_by_stratum[stratum] = quotas
            for scene_index, scene_id in enumerate(scene_ids):
                quota_report.append(
                    {
                        "split": split_name,
                        "scene_id": scene_id,
                        "stem": base.scene_stem(scene_id),
                        "sampling_class": 0,
                        "sampling_stratum": stratum,
                        "candidate_centres": int(capacities[scene_index]),
                        "allocated_quota": int(quotas[scene_index]),
                    }
                )

        for scene_index, scene_id in enumerate(scene_ids):
            header = header_records[scene_id]
            mask, _ = base.load_aligned_mask(
                config.source_root,
                "Training",
                scene_id,
                (header["height"], header["width"]),
            )
            foreground_counts = _window_foreground_counts(mask > 0, config.patch_size)
            # Spacing is enforced between background samples. Foreground-centred
            # crops are deliberately not exclusion zones: context negatives must
            # remain available near annotated lesions.
            occupied: list[tuple[int, int]] = []
            for stratum_index, stratum in enumerate(
                ("background_pure", "background_context")
            ):
                quota = int(quotas_by_stratum[stratum][scene_index])
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [config.seed, 101 if split_name == "train" else 211,
                         scene_id, stratum_index]
                    )
                )
                chosen, relaxed = _select_spaced_positions(
                    candidates_by_scene[scene_id][stratum],
                    quota,
                    rng,
                    occupied,
                    config.background_min_center_distance,
                )
                occupied.extend(chosen)
                for center_y, center_x in chosen:
                    y1, x1 = center_y - before, center_x - before
                    y2, x2 = center_y + after, center_x + after
                    count = int(foreground_counts[y1, x1])
                    fraction = count / float(config.patch_size**2)
                    samples.append(
                        {
                            "split": split_name,
                            "scene_id": scene_id,
                            "stem": base.scene_stem(scene_id),
                            "sampling_class": 0,
                            "sampling_stratum": stratum,
                            "center_y": center_y,
                            "center_x": center_x,
                            "crop_y1": y1,
                            "crop_x1": x1,
                            "crop_y2": y2,
                            "crop_x2": x2,
                            "foreground_fraction_at_sampling": fraction,
                            "spacing_relaxed": relaxed,
                            # Compatibility fields consumed by the historical
                            # patch materialiser and its manifest schema.
                            "alpha": 0.0,
                            "raw_quota": float(quota),
                            "unconstrained_quota": quota,
                            "allocated_scene_class_quota": quota,
                            "capacity_adjusted": False,
                        }
                    )
    samples.sort(
        key=lambda row: (
            row["split"], row["scene_id"], row["center_y"], row["center_x"]
        )
    )
    return samples, quota_report


def fit_complete_train_scene_minmax(
    config: RunConfig,
) -> tuple[float, float, int]:
    global_min, global_max = math.inf, -math.inf
    value_count = 0
    for scene_id in tqdm(config.train_scene_ids, desc="Fit complete Train-scene Min-Max"):
        image, _, _ = base.open_hsi(config.source_root, "Training", scene_id)
        cube = image.open_memmap()
        for y1 in range(0, int(image.shape[0]), NORMALIZATION_CHUNK_ROWS):
            y2 = min(int(image.shape[0]), y1 + NORMALIZATION_CHUNK_ROWS)
            chunk = np.asarray(cube[y1:y2, :, :])
            if not np.all(np.isfinite(chunk)):
                raise ValueError(f"non-finite intensity in {base.scene_stem(scene_id)}")
            global_min = min(global_min, float(chunk.min()))
            global_max = max(global_max, float(chunk.max()))
            value_count += int(chunk.size)
    if not np.isfinite(global_min) or not np.isfinite(global_max):
        raise RuntimeError("failed to fit complete-train-scene Min-Max")
    if global_max - global_min <= base.NORMALIZATION_EPS:
        raise ValueError(f"degenerate Min-Max range: {global_min}, {global_max}")
    return global_min, global_max, value_count


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
    for split_name in ("train", "val", "val_scenes", "test"):
        (output / split_name / "images").mkdir(parents=True, exist_ok=False)
        (output / split_name / "masks").mkdir(parents=True, exist_ok=False)
        if base.SAVE_REVIEW_VISUALIZATIONS:
            (output / split_name / base.REVIEW_DIR_NAME).mkdir(
                parents=True, exist_ok=False
            )
    (output / "PREPROCESSING_INCOMPLETE").write_text(
        base.utc_now() + "\n", encoding="utf-8"
    )
    return output


def _materialize_complete_scenes(
    config: RunConfig,
    output_root: Path,
    output_split: str,
    source_split: str,
    scene_ids: tuple[int, ...],
    wavelengths: np.ndarray,
    minimum: float,
    maximum: float,
) -> tuple[list[dict[str, Any]], tuple[float, float]]:
    manifest: list[dict[str, Any]] = []
    observed_min, observed_max = math.inf, -math.inf
    for scene_id in tqdm(scene_ids, desc=f"Write complete {output_split} scenes"):
        image, hdr_path, raw_path = base.open_hsi(
            config.source_root, source_split, scene_id
        )
        cube = image.open_memmap()
        target_hw = (int(image.shape[0]), int(image.shape[1]))
        mask, label_info = base.load_aligned_mask(
            config.source_root, source_split, scene_id, target_hw
        )
        stem = base.scene_stem(scene_id)
        image_path = output_root / output_split / "images" / f"{stem}.npy"
        mask_path = output_root / output_split / "masks" / f"{stem}.npy"
        image_min, image_max = base.write_normalized_full_scene(
            cube, image_path, minimum, maximum
        )
        np.save(mask_path, mask.astype(base.OUTPUT_MASK_DTYPE), allow_pickle=False)
        review_path: Path | None = None
        if base.SAVE_REVIEW_VISUALIZATIONS:
            review_path = (
                output_root / output_split / base.REVIEW_DIR_NAME / f"{stem}.png"
            )
            normalized = np.load(image_path, mmap_mode="r")
            base.save_review_visualization(
                normalized,
                mask,
                wavelengths,
                review_path,
                max_panel_side=base.REVIEW_TEST_MAX_PANEL_SIDE,
            )
        counts = base.class_counts(mask)
        row: dict[str, Any] = {
            "stem": stem,
            "split": output_split,
            "scene_id": scene_id,
            "source_hdr": str(hdr_path),
            "source_raw": str(raw_path),
            **label_info,
            "image_file": base.relative_to_output(image_path, output_root),
            "mask_file": base.relative_to_output(mask_path, output_root),
            "review_visualization_file": (
                base.relative_to_output(review_path, output_root) if review_path else ""
            ),
            "height": target_hw[0],
            "width": target_hw[1],
            "bands": int(image.shape[2]),
            "image_min": image_min,
            "image_max": image_max,
        }
        for class_id in base.VALID_CLASSES:
            row[f"class_{class_id}_pixels"] = int(counts[class_id])
            row[f"class_{class_id}_fraction"] = float(counts[class_id] / mask.size)
        manifest.append(row)
        observed_min = min(observed_min, image_min)
        observed_max = max(observed_max, image_max)
    return manifest, (float(observed_min), float(observed_max))


def _enrich_patch_manifests(
    manifests: dict[str, list[dict[str, Any]]],
    samples: list[dict[str, Any]],
) -> None:
    metadata = {base.patch_filename(row): row for row in samples}
    if len(metadata) != len(samples):
        raise AssertionError("duplicate output patch stems")
    for rows in manifests.values():
        for row in rows:
            source = metadata[row["stem"]]
            row["sampling_stratum"] = source.get(
                "sampling_stratum", f"foreground_class_{source['sampling_class']}"
            )
            row["foreground_fraction_at_sampling"] = source.get(
                "foreground_fraction_at_sampling", ""
            )
            row["spacing_relaxed"] = bool(source.get("spacing_relaxed", False))


def _pixel_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = np.asarray(
        [sum(int(row[f"class_{c}_pixels"]) for row in rows) for c in range(4)],
        dtype=np.int64,
    )
    total = int(counts.sum())
    return {
        f"class_{class_id}": {
            "pixels": int(counts[class_id]),
            "fraction": float(counts[class_id] / total),
        }
        for class_id in range(4)
    }


def validate_outputs(
    config: RunConfig,
    output_root: Path,
    manifests: dict[str, list[dict[str, Any]]],
    wavelengths: np.ndarray,
) -> dict[str, Any]:
    expected = {
        "train": sum(config.target_patches_per_class["train"].values())
        + config.background_patch_counts["train"],
        "val": sum(config.target_patches_per_class["val"].values())
        + config.background_patch_counts["val"],
        "val_scenes": len(config.val_scene_ids),
        "test": len(config.test_scene_ids),
    }
    report: dict[str, Any] = {"status": "ok", "splits": {}}
    for split_name, expected_count in expected.items():
        image_stems = {
            path.stem for path in (output_root / split_name / "images").glob("*.npy")
        }
        mask_stems = {
            path.stem for path in (output_root / split_name / "masks").glob("*.npy")
        }
        if image_stems != mask_stems or len(image_stems) != expected_count:
            raise AssertionError(
                f"{split_name}: pairs={len(image_stems)}/{len(mask_stems)}, "
                f"expected={expected_count}"
            )
        if len(manifests[split_name]) != expected_count:
            raise AssertionError(f"{split_name}: manifest count mismatch")
        saved_wavelengths = np.load(output_root / split_name / "wavelengths.npy")
        if not np.array_equal(saved_wavelengths, wavelengths):
            raise AssertionError(f"{split_name}: wavelengths mismatch")
        if base.SAVE_REVIEW_VISUALIZATIONS:
            review_stems = {
                path.stem
                for path in (
                    output_root / split_name / base.REVIEW_DIR_NAME
                ).glob("*.png")
            }
            if review_stems != image_stems:
                raise AssertionError(f"{split_name}: review pairing mismatch")
        complete_scene = split_name in {"val_scenes", "test"}
        for row in manifests[split_name]:
            image = np.load(output_root / row["image_file"], mmap_mode="r")
            mask = np.load(output_root / row["mask_file"], mmap_mode="r")
            expected_hw = (
                (int(row["height"]), int(row["width"]))
                if complete_scene
                else (config.patch_size, config.patch_size)
            )
            expected_image_shape = (*expected_hw, base.EXPECTED_BANDS)
            if image.shape != expected_image_shape or mask.shape != expected_hw:
                raise AssertionError(
                    f"{row['stem']}: shapes {image.shape}/{mask.shape}, expected "
                    f"{expected_image_shape}/{expected_hw}"
                )
            if image.dtype != np.dtype(base.OUTPUT_IMAGE_DTYPE):
                raise AssertionError(f"{row['stem']}: image dtype {image.dtype}")
            if mask.dtype != np.dtype(base.OUTPUT_MASK_DTYPE):
                raise AssertionError(f"{row['stem']}: mask dtype {mask.dtype}")
            unexpected = set(np.unique(mask).tolist()) - set(base.VALID_CLASSES)
            if unexpected:
                raise AssertionError(f"{row['stem']}: unexpected labels {unexpected}")
        report["splits"][split_name] = {
            "pairs": len(image_stems),
            "image_mask_pairing": "ok",
            "wavelengths": "ok",
            "review_visualizations": len(image_stems),
        }
    for split_name in ("train", "val"):
        for row in manifests[split_name]:
            if int(row["center_label"]) != int(row["sampling_class"]):
                raise AssertionError(f"{row['stem']}: centre-label mismatch")
    return report


def run(config: RunConfig) -> Path:
    if not 0.0 <= config.pure_background_ratio <= 1.0:
        raise ValueError("pure background ratio must be in [0,1]")
    if not (
        0.0 <= config.pure_bg_max_foreground_fraction
        < config.context_bg_max_foreground_fraction
        <= 1.0
    ):
        raise ValueError("invalid background foreground-fraction thresholds")
    for split_name in ("train", "val"):
        if set(config.target_patches_per_class[split_name]) != {1, 2, 3}:
            raise ValueError(f"{split_name}: foreground targets must contain 1,2,3")
        if config.background_patch_counts[split_name] < 0:
            raise ValueError("background patch counts must be nonnegative")

    print("=" * 88)
    print("PUAD official-224 background-aware preprocessing")
    print(f"Source:                   {config.source_root}")
    print(f"Output:                   {config.output_root}")
    print(f"Foreground targets:       {config.target_patches_per_class}")
    print(f"Background targets:       {config.background_patch_counts}")
    print(f"Pure/context ratio:       {config.pure_background_ratio:.2f}/"
          f"{1.0-config.pure_background_ratio:.2f}")
    print(f"Min-Max fit scope:        {NORMALIZATION_FIT_SCOPE}")
    print(f"Complete scene validation:{config.val_scene_ids}")
    print("=" * 88)

    header_records, wavelengths = base.inspect_scene_headers(config)
    scene_rows, label_rows = base.collect_sampling_statistics(config, header_records)
    alpha_rows, foreground_quota_rows = base.build_scene_quotas(config, scene_rows)
    foreground_samples = base.sample_patch_centres(
        config, header_records, foreground_quota_rows
    )
    background_samples, background_quota_rows = sample_background_centres(
        config, header_records
    )
    all_samples = sorted(
        [*foreground_samples, *background_samples],
        key=lambda row: (
            row["split"], row["scene_id"], row["center_y"], row["center_x"],
            row["sampling_class"],
        ),
    )

    minimum, maximum, fit_value_count = fit_complete_train_scene_minmax(config)
    print(
        f"Complete-Train Min-Max: min={minimum:.8g}, max={maximum:.8g}, "
        f"values={fit_value_count:,}"
    )
    output_root = prepare_output_root(config)
    np.save(output_root / "wavelengths.npy", wavelengths.astype(np.float32))
    for split_name in ("train", "val", "val_scenes", "test"):
        np.save(
            output_root / split_name / "wavelengths.npy",
            wavelengths.astype(np.float32),
            allow_pickle=False,
        )

    patch_manifests, patch_ranges = base.materialize_patch_splits(
        config, output_root, all_samples, wavelengths, minimum, maximum
    )
    _enrich_patch_manifests(patch_manifests, all_samples)
    val_scene_manifest, val_scene_range = _materialize_complete_scenes(
        config,
        output_root,
        "val_scenes",
        "Training",
        config.val_scene_ids,
        wavelengths,
        minimum,
        maximum,
    )
    test_manifest, test_range = _materialize_complete_scenes(
        config,
        output_root,
        "test",
        "Testing",
        config.test_scene_ids,
        wavelengths,
        minimum,
        maximum,
    )
    manifests = {
        "train": patch_manifests["train"],
        "val": patch_manifests["val"],
        "val_scenes": val_scene_manifest,
        "test": test_manifest,
    }
    for split_name, rows in manifests.items():
        base.write_csv(output_root / split_name / "manifest.csv", rows)

    foreground_only = {
        split_name: [
            row for row in manifests[split_name] if int(row["sampling_class"]) > 0
        ]
        for split_name in ("train", "val")
    }
    distributions = {
        "train_foreground_centred_only": _pixel_distribution(foreground_only["train"]),
        "train_after_background_addition": _pixel_distribution(manifests["train"]),
        "val_foreground_centred_only": _pixel_distribution(foreground_only["val"]),
        "val_after_background_addition": _pixel_distribution(manifests["val"]),
        "val_complete_scenes": _pixel_distribution(manifests["val_scenes"]),
        "test_complete_scenes": _pixel_distribution(manifests["test"]),
    }
    validation = validate_outputs(config, output_root, manifests, wavelengths)

    preprocessing_payload = {
        "generated_at": base.utc_now(),
        "script": str(SCRIPT_PATH),
        "protocol_name": "LUAD_PUAD_official224_bgaware_fullsceneval",
        "source_root": str(config.source_root),
        "output_root": str(output_root),
        "scene_splits": {
            "train": config.train_scene_ids,
            "val": config.val_scene_ids,
            "test": config.test_scene_ids,
        },
        "patch_size": config.patch_size,
        "sampling_seed": config.seed,
        "foreground_targets": config.target_patches_per_class,
        "background_sampling": {
            "targets": config.background_patch_counts,
            "pure_ratio": config.pure_background_ratio,
            "pure_max_foreground_fraction": config.pure_bg_max_foreground_fraction,
            "context_max_foreground_fraction": config.context_bg_max_foreground_fraction,
            "minimum_center_distance": config.background_min_center_distance,
        },
        "normalization_fit_scope": NORMALIZATION_FIT_SCOPE,
        "offline_nmf_included": False,
        "smoke_test": config.smoke_test,
    }
    base.write_json(output_root / "preprocessing_config.json", preprocessing_payload)
    base.write_json(
        output_root / "normalization_params.json",
        {
            "method": base.NORMALIZATION_METHOD,
            "fit_scope": NORMALIZATION_FIT_SCOPE,
            "global_min": minimum,
            "global_max": maximum,
            "denominator": maximum - minimum,
            "fit_value_count": fit_value_count,
            "clip": base.NORMALIZATION_CLIP,
            "observed_output_ranges": {
                "train": patch_ranges["train"],
                "val": patch_ranges["val"],
                "val_scenes": val_scene_range,
                "test": test_range,
            },
        },
    )
    base.write_json(
        output_root / "split_manifest.json",
        {
            "generated_at": base.utc_now(),
            "split_unit": "complete source scene",
            "train": {"scene_ids": config.train_scene_ids, "output": "patches"},
            "val": {"scene_ids": config.val_scene_ids, "output": "patches"},
            "val_scenes": {
                "scene_ids": config.val_scene_ids,
                "output": "complete scenes for checkpoint selection",
            },
            "test": {"scene_ids": config.test_scene_ids, "output": "complete scenes"},
        },
    )
    base.write_json(
        output_root / "sampling_report.json",
        {
            "generated_at": base.utc_now(),
            "foreground_alpha": alpha_rows,
            "foreground_scene_class_quotas": foreground_quota_rows,
            "background_scene_stratum_quotas": background_quota_rows,
            "scene_statistics": scene_rows,
            "label_alignment": label_rows,
            "sample_counts": {key: len(value) for key, value in manifests.items()},
            "pixel_distributions": distributions,
            "background_spacing_relaxed_samples": sum(
                bool(row.get("spacing_relaxed")) for row in background_samples
            ),
            "validation": validation,
        },
    )
    base.write_csv(
        output_root / "sampling_scene_class_quotas.csv",
        [*foreground_quota_rows, *background_quota_rows],
    )
    (output_root / "PREPROCESSING_INCOMPLETE").unlink()
    print("=" * 88)
    print("Background-aware preprocessing completed and validated")
    print(f"Output: {output_root}")
    print("Counts: " + json.dumps({k: len(v) for k, v in manifests.items()}))
    print("Run offline NMF independently for train, val, val_scenes, and test.")
    print("=" * 88)
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build background-aware PUAD official-224 segmentation data"
    )
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=SAMPLING_SEED)
    parser.add_argument(
        "--foreground-train-patches",
        type=int,
        default=sum(FOREGROUND_PATCHES_PER_CLASS["train"].values()),
        help=(
            "Total foreground-centred training patches. The total is split "
            "as evenly as possible across classes 1, 2, and 3."
        ),
    )
    parser.add_argument(
        "--foreground-val-patches",
        type=int,
        default=sum(FOREGROUND_PATCHES_PER_CLASS["val"].values()),
        help=(
            "Total foreground-centred validation patches. The total is split "
            "as evenly as possible across classes 1, 2, and 3."
        ),
    )
    parser.add_argument(
        "--background-train-patches", type=int, default=BACKGROUND_PATCH_COUNTS["train"]
    )
    parser.add_argument(
        "--background-val-patches", type=int, default=BACKGROUND_PATCH_COUNTS["val"]
    )
    parser.add_argument("--pure-background-ratio", type=float, default=BACKGROUND_PURE_RATIO)
    parser.add_argument(
        "--pure-bg-max-foreground-fraction",
        type=float,
        default=PURE_BG_MAX_FOREGROUND_FRACTION,
    )
    parser.add_argument(
        "--context-bg-max-foreground-fraction",
        type=float,
        default=CONTEXT_BG_MAX_FOREGROUND_FRACTION,
    )
    parser.add_argument(
        "--background-min-center-distance",
        type=int,
        default=BACKGROUND_MIN_CENTER_DISTANCE,
    )
    parser.add_argument("--overwrite", action="store_true", default=OVERWRITE_OUTPUT)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def balanced_foreground_targets(total: int) -> dict[int, int]:
    """Distribute an exact total deterministically and nearly evenly over 3 classes."""
    total = int(total)
    foreground_classes = tuple(class_id for class_id in base.VALID_CLASSES if class_id > 0)
    if total < len(foreground_classes):
        raise ValueError(
            "foreground patch total must be at least the number of foreground "
            f"classes ({len(foreground_classes)}), got {total}"
        )
    quotient, remainder = divmod(total, len(foreground_classes))
    return {
        int(class_id): quotient + int(index < remainder)
        for index, class_id in enumerate(foreground_classes)
    }


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        train_ids = SMOKE_TRAIN_SCENE_IDS
        val_ids = SMOKE_VAL_SCENE_IDS
        test_ids = SMOKE_TEST_SCENE_IDS
        foreground_targets = SMOKE_FOREGROUND_PATCHES_PER_CLASS
        background_targets = SMOKE_BACKGROUND_PATCH_COUNTS
    else:
        train_ids = TRAIN_SCENE_IDS
        val_ids = VAL_SCENE_IDS
        test_ids = TEST_SCENE_IDS
        foreground_targets = {
            "train": balanced_foreground_targets(args.foreground_train_patches),
            "val": balanced_foreground_targets(args.foreground_val_patches),
        }
        background_targets = {
            "train": args.background_train_patches,
            "val": args.background_val_patches,
        }
    config = RunConfig(
        source_root=args.source_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        train_scene_ids=tuple(train_ids),
        val_scene_ids=tuple(val_ids),
        test_scene_ids=tuple(test_ids),
        target_patches_per_class={
            split: {int(key): int(value) for key, value in targets.items()}
            for split, targets in foreground_targets.items()
        },
        background_patch_counts={
            split: int(value) for split, value in background_targets.items()
        },
        patch_size=PATCH_SIZE,
        seed=int(args.seed),
        pure_background_ratio=float(args.pure_background_ratio),
        pure_bg_max_foreground_fraction=float(
            args.pure_bg_max_foreground_fraction
        ),
        context_bg_max_foreground_fraction=float(
            args.context_bg_max_foreground_fraction
        ),
        background_min_center_distance=int(args.background_min_center_distance),
        overwrite=bool(args.overwrite),
        smoke_test=bool(args.smoke_test),
    )
    run(config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "Interrupted; PREPROCESSING_INCOMPLETE is retained if output began.",
            file=sys.stderr,
        )
        raise
