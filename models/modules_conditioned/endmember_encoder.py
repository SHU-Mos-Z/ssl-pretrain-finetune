"""Permutation-equivariant encoding of per-sample endmember sets."""

from __future__ import annotations

import torch
import torch.nn as nn


class EndmemberSetEncoder(nn.Module):
    def __init__(self, spectral_patch_size: int, embed_dim: int):
        super().__init__()
        self.spectral_patch_size = spectral_patch_size
        self.point_encoder = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.group_norm = nn.LayerNorm(embed_dim)
        self.global_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self, e_star: torch.Tensor, wavelengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return group tokens (B,K,G,D) and endmember tokens (B,K,D)."""
        if e_star.ndim != 3 or wavelengths.ndim != 2:
            raise ValueError("e_star and wavelengths must have shapes (B,K,S) and (B,S)")
        b, k, s = e_star.shape
        if wavelengths.shape != (b, s):
            raise ValueError("wavelengths must match the batch and spectral dimensions")
        if s % self.spectral_patch_size:
            raise ValueError("the number of bands must be divisible by spectral_patch_size")

        g = s // self.spectral_patch_size
        e = e_star.detach().reshape(b, k, g, self.spectral_patch_size)
        wave = wavelengths.reshape(b, 1, g, self.spectral_patch_size).expand(-1, k, -1, -1)
        wave_min = wavelengths.amin(dim=1, keepdim=True)
        wave_span = (wavelengths.amax(dim=1, keepdim=True) - wave_min).clamp(min=1e-6)
        wave = (wave - wave_min[:, :, None, None]) / wave_span[:, :, None, None]
        points = torch.stack((e, wave), dim=-1)
        group_tokens = self.group_norm(self.point_encoder(points).mean(dim=-2))
        endmember_tokens = self.global_encoder(group_tokens.mean(dim=2))
        return group_tokens, endmember_tokens
