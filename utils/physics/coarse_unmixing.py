"""Patch-level coarse unmixing from visible OD bands and per-sample endmembers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class CoarseUnmixingResult:
    c0_low: torch.Tensor
    c0: torch.Tensor
    x0: torch.Tensor
    rho_low: torch.Tensor
    rho: torch.Tensor
    visible_fraction_low: torch.Tensor
    fit_error_low: torch.Tensor
    condition_number_low: torch.Tensor


def project_simplex(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Project vectors on the last axis onto {x >= 0, sum(x) = 1}."""
    if x.shape[-1] == 0:
        raise ValueError("cannot project an empty vector")
    sorted_x, _ = torch.sort(x, dim=-1, descending=True)
    cssv = sorted_x.cumsum(dim=-1) - 1.0
    idx = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=x.dtype)
    view_shape = [1] * (x.ndim - 1) + [x.shape[-1]]
    idx = idx.view(view_shape)
    support = sorted_x - cssv / idx > 0
    rho = support.sum(dim=-1, keepdim=True).clamp(min=1)
    theta = cssv.gather(-1, rho - 1) / rho.to(x.dtype)
    projected = (x - theta).clamp(min=0.0)
    return projected / projected.sum(dim=-1, keepdim=True).clamp(min=eps)


def _patch_visible_means(
    od: torch.Tensor,
    voxel_visible: torch.Tensor,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if od.shape != voxel_visible.shape:
        raise ValueError(f"od {od.shape} and voxel_visible {voxel_visible.shape} must match")
    b, s, h, w = od.shape
    if h % patch_size or w % patch_size:
        raise ValueError(f"H/W ({h}, {w}) must be divisible by patch_size={patch_size}")

    hp, wp = h // patch_size, w // patch_size
    area = float(patch_size * patch_size)
    visible = voxel_visible.to(dtype=od.dtype)
    visible_count = F.avg_pool2d(visible, patch_size, stride=patch_size) * area
    visible_sum = F.avg_pool2d(od * visible, patch_size, stride=patch_size) * area
    patch_mean = visible_sum / visible_count.clamp(min=1.0)
    band_visibility = visible_count / area

    # (B,S,Hp,Wp) -> (B,L,S), L=Hp*Wp
    patch_mean = patch_mean.permute(0, 2, 3, 1).reshape(b, hp * wp, s)
    band_visibility = band_visibility.permute(0, 2, 3, 1).reshape(b, hp * wp, s)
    return patch_mean, band_visibility, hp, wp


def coarse_unmix(
    od: torch.Tensor,
    e_star: torch.Tensor,
    voxel_visible: torch.Tensor,
    patch_size: int,
    ridge_lambda: float = 1e-3,
    confidence_temperature: float = 0.05,
    eps: float = 1e-8,
) -> CoarseUnmixingResult:
    """Run the analytic solve in FP32 even when the caller uses AMP."""
    with torch.autocast(device_type=od.device.type, enabled=False):
        return _coarse_unmix_fp32(
            od.float(), e_star.float(), voxel_visible, patch_size,
            ridge_lambda, confidence_temperature, eps,
        )


def _coarse_unmix_fp32(
    od: torch.Tensor,
    e_star: torch.Tensor,
    voxel_visible: torch.Tensor,
    patch_size: int,
    ridge_lambda: float = 1e-3,
    confidence_temperature: float = 0.05,
    eps: float = 1e-8,
) -> CoarseUnmixingResult:
    """Compute C0, its physical reconstruction, and patch reliability.

    Args:
        od: (B,S,H,W) full target OD. Only entries selected by voxel_visible
            participate in C0 estimation.
        e_star: (B,K,S) detached per-sample endmembers.
        voxel_visible: (B,S,H,W), one for visible and zero for masked.
    """
    if od.ndim != 4 or e_star.ndim != 3:
        raise ValueError("od must be 4D and e_star must be 3D")
    b, s, h, w = od.shape
    if e_star.shape[0] != b or e_star.shape[2] != s:
        raise ValueError(f"e_star {e_star.shape} is incompatible with od {od.shape}")
    if ridge_lambda <= 0:
        raise ValueError("ridge_lambda must be positive")

    e = e_star.detach().to(dtype=od.dtype)
    patch_mean, visibility, hp, wp = _patch_visible_means(od, voxel_visible, patch_size)
    k = e.shape[1]

    # A[b,l,k,j] = sum_s E[b,k,s] * visibility[b,l,s] * E[b,j,s]
    system = torch.einsum("bks,bls,bjs->blkj", e, visibility, e)
    eye = torch.eye(k, device=od.device, dtype=od.dtype).view(1, 1, k, k)
    system = system + ridge_lambda * eye
    rhs = torch.einsum("bks,bls,bls->blk", e, visibility, patch_mean)
    solution = torch.linalg.solve(system, rhs.unsqueeze(-1)).squeeze(-1)
    c0_flat = project_simplex(solution, eps=eps)

    pred_patch = torch.einsum("blk,bks->bls", c0_flat, e)
    weighted_sq = visibility * (pred_patch - patch_mean).pow(2)
    fit_error = weighted_sq.sum(dim=-1) / visibility.sum(dim=-1).clamp(min=1.0)
    visible_fraction = visibility.mean(dim=-1)

    eigvals = torch.linalg.eigvalsh(system.float()).clamp(min=eps)
    condition_number = (eigvals[..., -1] / eigvals[..., 0]).to(od.dtype)
    condition_score = 1.0 / (1.0 + torch.log(condition_number.clamp(min=1.0)))
    fit_score = torch.exp(-fit_error / max(confidence_temperature, eps))
    rho_flat = (visible_fraction * fit_score * condition_score).clamp(0.0, 1.0)

    c0_low = c0_flat.view(b, hp, wp, k).permute(0, 3, 1, 2).contiguous()
    c0 = F.interpolate(c0_low, size=(h, w), mode="bilinear", align_corners=False)
    c0 = c0 / c0.sum(dim=1, keepdim=True).clamp(min=eps)
    x0 = torch.einsum("bkhw,bks->bshw", c0, e)

    def patch_scalar(x: torch.Tensor) -> torch.Tensor:
        return x.view(b, 1, hp, wp)

    rho_low = patch_scalar(rho_flat)
    rho = F.interpolate(rho_low, size=(h, w), mode="bilinear", align_corners=False)
    return CoarseUnmixingResult(
        c0_low=c0_low,
        c0=c0,
        x0=x0,
        rho_low=rho_low,
        rho=rho,
        visible_fraction_low=patch_scalar(visible_fraction),
        fit_error_low=patch_scalar(fit_error),
        condition_number_low=patch_scalar(condition_number),
    )
