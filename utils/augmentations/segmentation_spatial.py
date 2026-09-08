"""Deterministic online spatial augmentation for HSI segmentation pairs.

The defaults in the fine-tuning Dataset keep this module completely disabled.
When enabled, the same geometry is applied to the hyperspectral cube and its
integer segmentation mask.  Interpolating geometry is performed in optical
density space for the HSI and with nearest-neighbour interpolation for labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from utils.augmentations.hsi_spatial import (
    PERSPECTIVE_PADDING_MODES,
    derive_perspective_seed,
    sample_inward_four_point_perspective,
    warp_chw_with_perspective,
)
from utils.physics.beer_lambert import intensity_to_od_np


SEGMENTATION_AUGMENTATION_POLICIES = (
    "dihedral",
    "dihedral_affine",
    "dihedral_perspective",
)


@dataclass(frozen=True)
class SegmentationAugmentationResult:
    intensity: np.ndarray
    mask: np.ndarray
    transform_id: int
    interpolating_transform_applied: bool
    transform_seed: int


def _dihedral(array: np.ndarray, transform_id: int) -> np.ndarray:
    if not 0 <= transform_id < 8:
        raise ValueError("dihedral transform_id must be in [0,7]")
    rotated = np.rot90(array, k=transform_id % 4, axes=(-2, -1))
    if transform_id >= 4:
        rotated = np.flip(rotated, axis=-1)
    return np.ascontiguousarray(rotated)


def _deterministic_transform_id(
    base_seed: int, epoch: int, sample_index: int, copy_index: int
) -> int:
    seed = derive_perspective_seed(
        base_seed,
        epoch,
        sample_index,
        copy_index,
        namespace="segmentation-dihedral",
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return int(torch.randint(0, 8, (), generator=generator).item())


def _should_apply(
    probability: float,
    *,
    base_seed: int,
    epoch: int,
    sample_index: int,
    copy_index: int,
    namespace: str,
) -> bool:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("augmentation probability must be in [0,1]")
    if probability in {0.0, 1.0}:
        return bool(probability)
    seed = derive_perspective_seed(
        base_seed, epoch, sample_index, copy_index, namespace=namespace
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return bool(torch.rand((), generator=generator).item() < probability)


def _warp_affine(
    chw: torch.Tensor,
    mask_hw: torch.Tensor,
    *,
    rotation_degrees: float,
    scale_delta: float,
    translate_fraction: float,
    seed: int,
    padding_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    unit = torch.rand(4, generator=generator, dtype=torch.float64)
    angle = (2.0 * float(unit[0]) - 1.0) * rotation_degrees
    scale = 1.0 + (2.0 * float(unit[1]) - 1.0) * scale_delta
    translate_x = (2.0 * float(unit[2]) - 1.0) * translate_fraction
    translate_y = (2.0 * float(unit[3]) - 1.0) * translate_fraction
    radians = np.deg2rad(angle)
    cosine = float(np.cos(radians)) / scale
    sine = float(np.sin(radians)) / scale
    theta = torch.tensor(
        [[cosine, sine, translate_x], [-sine, cosine, translate_y]],
        dtype=torch.float32,
        device=chw.device,
    ).unsqueeze(0)
    grid = F.affine_grid(
        theta, size=(1, chw.shape[0], chw.shape[1], chw.shape[2]), align_corners=True
    )
    warped = F.grid_sample(
        chw.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    ).squeeze(0)
    mask_warped = F.grid_sample(
        mask_hw.to(torch.float32).unsqueeze(0).unsqueeze(0),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(0).squeeze(0)
    return warped, mask_warped


def augment_hsi_segmentation_pair(
    intensity_chw: np.ndarray,
    mask_hw: np.ndarray,
    *,
    policy: str,
    probability: float = 1.0,
    base_seed: int = 42,
    epoch: int = 0,
    sample_index: int = 0,
    copy_index: int = 0,
    transform_id: int | None = None,
    affine_rotation_degrees: float = 15.0,
    affine_scale_delta: float = 0.1,
    affine_translate_fraction: float = 0.05,
    perspective_scale: float = 0.05,
    padding_mode: str = "reflection",
    od_max: float | None = 3.0,
) -> SegmentationAugmentationResult:
    """Augment one aligned ``(S,H,W)`` HSI and ``(H,W)`` integer mask."""

    if policy not in SEGMENTATION_AUGMENTATION_POLICIES:
        raise ValueError(
            f"policy must be one of {SEGMENTATION_AUGMENTATION_POLICIES}"
        )
    if padding_mode not in PERSPECTIVE_PADDING_MODES:
        raise ValueError(f"padding_mode must be one of {PERSPECTIVE_PADDING_MODES}")
    intensity = np.asarray(intensity_chw, dtype=np.float32)
    mask = np.asarray(mask_hw)
    if intensity.ndim != 3 or mask.ndim != 2:
        raise ValueError("expected intensity (S,H,W) and mask (H,W)")
    if intensity.shape[-2:] != mask.shape:
        raise ValueError("HSI and segmentation mask spatial shapes differ")

    transform_id = (
        _deterministic_transform_id(base_seed, epoch, sample_index, copy_index)
        if transform_id is None
        else int(transform_id)
    )
    intensity = _dihedral(intensity, transform_id)
    mask = _dihedral(mask, transform_id)
    apply_interpolation = policy != "dihedral" and _should_apply(
        probability,
        base_seed=base_seed,
        epoch=epoch,
        sample_index=sample_index,
        copy_index=copy_index,
        namespace=f"segmentation-{policy}-apply",
    )
    if not apply_interpolation:
        return SegmentationAugmentationResult(
            intensity=np.ascontiguousarray(intensity),
            mask=np.ascontiguousarray(mask),
            transform_id=transform_id,
            interpolating_transform_applied=False,
            transform_seed=-1,
        )

    od = torch.from_numpy(
        np.clip(intensity_to_od_np(intensity), 0.0, od_max).astype(np.float32)
    )
    mask_tensor = torch.from_numpy(mask.astype(np.int64, copy=False))
    height, width = mask.shape
    transform_seed = derive_perspective_seed(
        base_seed,
        epoch,
        sample_index,
        copy_index,
        namespace=f"segmentation-{policy}-parameters",
    )
    if policy == "dihedral_affine":
        od_warped, mask_warped = _warp_affine(
            od,
            mask_tensor,
            rotation_degrees=affine_rotation_degrees,
            scale_delta=affine_scale_delta,
            translate_fraction=affine_translate_fraction,
            seed=transform_seed,
            padding_mode=padding_mode,
        )
    else:
        parameters = sample_inward_four_point_perspective(
            height,
            width,
            scale=perspective_scale,
            base_seed=base_seed,
            epoch=epoch,
            sample_index=sample_index,
            copy_index=copy_index,
        )
        transform_seed = int(parameters.seed)
        od_warped = warp_chw_with_perspective(
            od, parameters, interpolation="bilinear", padding_mode=padding_mode
        )
        mask_warped = warp_chw_with_perspective(
            mask_tensor.unsqueeze(0),
            parameters,
            interpolation="nearest",
            padding_mode="zeros",
        ).squeeze(0)

    if od_max is not None:
        od_warped = od_warped.clamp(0.0, float(od_max))
    intensity_warped = torch.exp(-od_warped)
    return SegmentationAugmentationResult(
        intensity=np.ascontiguousarray(intensity_warped.numpy().astype(np.float32)),
        mask=np.ascontiguousarray(mask_warped.numpy().astype(mask.dtype)),
        transform_id=transform_id,
        interpolating_transform_applied=True,
        transform_seed=transform_seed,
    )


__all__ = [
    "SEGMENTATION_AUGMENTATION_POLICIES",
    "SegmentationAugmentationResult",
    "augment_hsi_segmentation_pair",
]
