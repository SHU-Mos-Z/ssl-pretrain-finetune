"""Hybrid spectral-group and fully-spatial patch masking."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HybridMaskConfig:
    spectral_mask_ratio: float = 0.3
    spatial_mask_ratio: float = 0.2

    def validate(self) -> None:
        for name, value in (
            ("spectral_mask_ratio", self.spectral_mask_ratio),
            ("spatial_mask_ratio", self.spatial_mask_ratio),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass
class HybridMaskResult:
    token_visible: torch.Tensor
    spatial_masked: torch.Tensor
    voxel_visible: torch.Tensor


def expand_token_visibility(
    token_visible: torch.Tensor,
    patch_size: int,
    spectral_patch_size: int,
    num_bands: int | None = None,
) -> torch.Tensor:
    """Expand (B,Hp,Wp,G) visibility to (B,S,H,W)."""
    if token_visible.ndim != 4:
        raise ValueError("token_visible must have shape (B,Hp,Wp,G)")
    visible = token_visible.permute(0, 3, 1, 2)
    visible = visible.repeat_interleave(spectral_patch_size, dim=1)
    visible = visible.repeat_interleave(patch_size, dim=2)
    visible = visible.repeat_interleave(patch_size, dim=3)
    if num_bands is not None:
        if num_bands > visible.shape[1]:
            raise ValueError("num_bands exceeds represented spectral groups")
        visible = visible[:, :num_bands]
    return visible


def sample_hybrid_mask(
    batch_size: int,
    h_p: int,
    w_p: int,
    num_groups: int,
    patch_size: int,
    spectral_patch_size: int,
    config: HybridMaskConfig,
    *,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    difficulty: torch.Tensor | None = None,
    curriculum_weight: float = 0.0,
    num_bands: int | None = None,
) -> HybridMaskResult:
    """Sample exact-count masks; True/one always means visible."""
    config.validate()
    if min(batch_size, h_p, w_p, num_groups) <= 0:
        raise ValueError("all dimensions must be positive")
    if difficulty is not None and difficulty.shape != (batch_size, h_p, w_p, num_groups):
        raise ValueError("difficulty must match (B,Hp,Wp,G)")

    token_visible = torch.ones(
        batch_size, h_p, w_p, num_groups, dtype=torch.bool, device=device
    )
    spatial_masked = torch.zeros(batch_size, h_p, w_p, dtype=torch.bool, device=device)
    n_spatial = round(config.spatial_mask_ratio * h_p * w_p)

    for batch_idx in range(batch_size):
        if n_spatial:
            order = torch.randperm(h_p * w_p, generator=generator, device=device)
            chosen = order[:n_spatial]
            spatial_masked[batch_idx].view(-1)[chosen] = True
            token_visible[batch_idx].view(h_p * w_p, num_groups)[chosen] = False

        candidates = (~spatial_masked[batch_idx]).unsqueeze(-1).expand(-1, -1, num_groups)
        candidate_idx = candidates.reshape(-1).nonzero(as_tuple=False).squeeze(1)
        n_spectral = round(config.spectral_mask_ratio * candidate_idx.numel())
        if not n_spectral:
            continue

        if difficulty is None or curriculum_weight <= 0.0:
            order = torch.randperm(candidate_idx.numel(), generator=generator, device=device)
            selected = candidate_idx[order[:n_spectral]]
        else:
            scores = difficulty[batch_idx].reshape(-1)[candidate_idx].float()
            scores = scores - scores.max()
            guided = torch.softmax(scores, dim=0)
            uniform = torch.full_like(guided, 1.0 / guided.numel())
            weight = min(max(curriculum_weight, 0.0), 1.0)
            probs = (1.0 - weight) * uniform + weight * guided
            local = torch.multinomial(probs, n_spectral, replacement=False, generator=generator)
            selected = candidate_idx[local]
        token_visible[batch_idx].view(-1)[selected] = False

    voxel_visible = expand_token_visibility(
        token_visible,
        patch_size,
        spectral_patch_size,
        num_bands=num_bands,
    )
    return HybridMaskResult(token_visible, spatial_masked, voxel_visible)
