"""Deterministic spatial augmentation for hyperspectral cubes.

Perspective interpolation is performed in optical-density (OD) space.  This is
important because bilinear interpolation and the logarithm used to convert
intensity to OD do not commute.  A single sampling grid can optionally be
applied to an NMF abundance map so that ``OD ~= E* @ C*`` remains aligned.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


ArrayLike = np.ndarray | torch.Tensor
PERSPECTIVE_PADDING_MODES = ("zeros", "border", "reflection")


@dataclass(frozen=True)
class PerspectiveParameters:
    """Four-point transform parameters in TL, TR, BR, BL point order."""

    source_points: torch.Tensor
    destination_points: torch.Tensor
    homography_output_to_input: torch.Tensor
    height: int
    width: int
    seed: int
    scale: float


def derive_perspective_seed(
    base_seed: int,
    epoch: int,
    sample_index: int,
    copy_index: int,
    *,
    namespace: str = "corners",
) -> int:
    """Derive a stable seed independent of process-local RNG state."""

    payload = (
        f"{int(base_seed)}|{int(epoch)}|{int(sample_index)}|"
        f"{int(copy_index)}|{namespace}"
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**63 - 1)


def should_apply_perspective(
    probability: float,
    *,
    base_seed: int,
    epoch: int,
    sample_index: int,
    copy_index: int,
) -> bool:
    """Make a deterministic per-view Bernoulli decision."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("perspective probability must be in [0,1]")
    if probability == 0.0:
        return False
    if probability == 1.0:
        return True
    seed = derive_perspective_seed(
        base_seed,
        epoch,
        sample_index,
        copy_index,
        namespace="apply",
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return bool(torch.rand((), generator=generator).item() < probability)


def _solve_homography(
    source: torch.Tensor, destination: torch.Tensor
) -> torch.Tensor:
    """Return H such that ``destination ~ H @ source`` for four pairs."""

    source = source.to(dtype=torch.float64, device="cpu")
    destination = destination.to(dtype=torch.float64, device="cpu")
    if source.shape != (4, 2) or destination.shape != (4, 2):
        raise ValueError("source and destination must both have shape (4,2)")

    rows: list[torch.Tensor] = []
    rhs: list[torch.Tensor] = []
    for (x_coord, y_coord), (u_coord, v_coord) in zip(source, destination):
        zero = x_coord.new_tensor(0.0)
        one = x_coord.new_tensor(1.0)
        rows.append(
            torch.stack(
                [
                    x_coord,
                    y_coord,
                    one,
                    zero,
                    zero,
                    zero,
                    -u_coord * x_coord,
                    -u_coord * y_coord,
                ]
            )
        )
        rows.append(
            torch.stack(
                [
                    zero,
                    zero,
                    zero,
                    x_coord,
                    y_coord,
                    one,
                    -v_coord * x_coord,
                    -v_coord * y_coord,
                ]
            )
        )
        rhs.extend([u_coord, v_coord])
    coefficients = torch.linalg.solve(torch.stack(rows), torch.stack(rhs))
    return torch.cat([coefficients, coefficients.new_ones(1)]).reshape(3, 3)


def sample_inward_four_point_perspective(
    height: int,
    width: int,
    *,
    scale: float = 0.05,
    base_seed: int = 42,
    epoch: int = 0,
    sample_index: int = 0,
    copy_index: int = 0,
) -> PerspectiveParameters:
    """Sample a convex inward perturbation measured against full image size."""

    if height < 2 or width < 2:
        raise ValueError("height and width must be at least 2")
    if not 0.0 <= scale < 0.5:
        raise ValueError("perspective scale must be in [0,0.5)")

    seed = derive_perspective_seed(
        base_seed,
        epoch,
        sample_index,
        copy_index,
        namespace="corners",
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.rand(8, generator=generator, dtype=torch.float64)
    x_max, y_max = float(width - 1), float(height - 1)
    max_dx, max_dy = scale * x_max, scale * y_max
    source = torch.tensor(
        [[0.0, 0.0], [x_max, 0.0], [x_max, y_max], [0.0, y_max]],
        dtype=torch.float64,
    )
    destination = torch.stack(
        [
            torch.stack([values[0] * max_dx, values[1] * max_dy]),
            torch.stack(
                [
                    values[2].new_tensor(x_max) - values[2] * max_dx,
                    values[3] * max_dy,
                ]
            ),
            torch.stack(
                [
                    values[4].new_tensor(x_max) - values[4] * max_dx,
                    values[5].new_tensor(y_max) - values[5] * max_dy,
                ]
            ),
            torch.stack(
                [
                    values[6] * max_dx,
                    values[7].new_tensor(y_max) - values[7] * max_dy,
                ]
            ),
        ]
    )
    # grid_sample consumes an output -> input mapping.
    output_to_input = _solve_homography(destination, source)
    return PerspectiveParameters(
        source_points=source,
        destination_points=destination,
        homography_output_to_input=output_to_input,
        height=int(height),
        width=int(width),
        seed=seed,
        scale=float(scale),
    )


def _perspective_sampling_grid(
    parameters: PerspectiveParameters,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    height, width = parameters.height, parameters.width
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    output_points = torch.stack(
        [x_coords, y_coords, torch.ones_like(x_coords)], dim=-1
    )
    homography = parameters.homography_output_to_input.to(
        device=device, dtype=dtype
    )
    input_points = output_points @ homography.T
    denominator = input_points[..., 2]
    epsilon = torch.finfo(dtype).eps
    denominator = torch.where(
        denominator.abs() < epsilon,
        torch.where(denominator < 0, -epsilon, epsilon),
        denominator,
    )
    input_x = input_points[..., 0] / denominator
    input_y = input_points[..., 1] / denominator
    normalized_x = 2.0 * input_x / (width - 1) - 1.0
    normalized_y = 2.0 * input_y / (height - 1) - 1.0
    return torch.stack([normalized_x, normalized_y], dim=-1).unsqueeze(0)


def warp_chw_with_perspective(
    chw: ArrayLike,
    parameters: PerspectiveParameters,
    *,
    interpolation: str = "bilinear",
    padding_mode: str = "reflection",
) -> torch.Tensor:
    """Warp a ``(channels,height,width)`` array without changing its size."""

    tensor = torch.as_tensor(chw, dtype=torch.float32)
    if tensor.ndim != 3:
        raise ValueError(f"expected CHW data, got {tuple(tensor.shape)}")
    if tuple(tensor.shape[-2:]) != (parameters.height, parameters.width):
        raise ValueError("tensor shape and perspective parameters disagree")
    if interpolation not in {"bilinear", "nearest"}:
        raise ValueError("interpolation must be 'bilinear' or 'nearest'")
    if padding_mode not in PERSPECTIVE_PADDING_MODES:
        raise ValueError(
            f"padding_mode must be one of {PERSPECTIVE_PADDING_MODES}"
        )
    grid = _perspective_sampling_grid(
        parameters, device=tensor.device, dtype=tensor.dtype
    )
    return F.grid_sample(
        tensor.unsqueeze(0),
        grid,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=True,
    ).squeeze(0)


def random_four_point_perspective_od_and_abundance(
    od_chw: ArrayLike,
    c_star_khw: ArrayLike | None = None,
    *,
    scale: float = 0.05,
    base_seed: int = 42,
    epoch: int = 0,
    sample_index: int = 0,
    copy_index: int = 0,
    od_max: float | None = 3.0,
    eps: float = 1e-6,
    padding_mode: str = "reflection",
) -> dict[str, object]:
    """Apply one shared random perspective grid to OD and optional abundance."""

    od = torch.as_tensor(od_chw, dtype=torch.float32)
    if od.ndim != 3:
        raise ValueError("OD must have shape (S,H,W)")
    if not torch.isfinite(od).all():
        raise ValueError("OD contains NaN or Inf")
    if od_max is not None:
        if od_max <= 0:
            raise ValueError("od_max must be positive or None")
        od = od.clamp(min=0.0, max=float(od_max))

    abundance: torch.Tensor | None = None
    if c_star_khw is not None:
        abundance = torch.as_tensor(
            c_star_khw, dtype=torch.float32, device=od.device
        )
        if abundance.ndim != 3 or abundance.shape[-2:] != od.shape[-2:]:
            raise ValueError("C* must have shape (K,H,W) aligned with OD")
        if not torch.isfinite(abundance).all():
            raise ValueError("C* contains NaN or Inf")

    height, width = map(int, od.shape[-2:])
    parameters = sample_inward_four_point_perspective(
        height,
        width,
        scale=scale,
        base_seed=base_seed,
        epoch=epoch,
        sample_index=sample_index,
        copy_index=copy_index,
    )
    joint = od if abundance is None else torch.cat([od, abundance], dim=0)
    warped = warp_chw_with_perspective(
        joint,
        parameters,
        interpolation="bilinear",
        padding_mode=padding_mode,
    )
    od_augmented = warped[: od.shape[0]]
    result: dict[str, object] = {
        "od": od_augmented,
        "intensity": torch.exp(-od_augmented),
        "parameters": parameters,
    }
    if abundance is not None:
        abundance_raw = warped[od.shape[0] :]
        abundance_nonnegative = abundance_raw.clamp_min(0.0)
        abundance_augmented = abundance_nonnegative / abundance_nonnegative.sum(
            dim=0, keepdim=True
        ).clamp_min(eps)
        result["c_star"] = abundance_augmented
        result["c_star_before_simplex_correction"] = abundance_raw
    return result


def random_four_point_perspective_hsi_and_abundance(
    intensity_chw: ArrayLike,
    c_star_khw: ArrayLike | None = None,
    **kwargs,
) -> dict[str, object]:
    """Convert intensity to OD, then run the physically consistent transform."""

    intensity = torch.as_tensor(intensity_chw, dtype=torch.float32)
    if intensity.ndim != 3:
        raise ValueError("intensity must have shape (S,H,W)")
    if not torch.isfinite(intensity).all():
        raise ValueError("intensity contains NaN or Inf")
    eps = float(kwargs.get("eps", 1e-6))
    od = -torch.log(intensity.clamp_min(eps))
    return random_four_point_perspective_od_and_abundance(
        od, c_star_khw, **kwargs
    )
