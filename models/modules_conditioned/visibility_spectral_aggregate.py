"""Aggregate spectral-group tokens without failing on fully masked patches."""

from __future__ import annotations

import torch
import torch.nn as nn


class VisibilitySpectralAggregate(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.score = nn.Linear(embed_dim, 1)

    def forward(self, z_grid: torch.Tensor, token_visible: torch.Tensor) -> torch.Tensor:
        visible = token_visible.to(z_grid.dtype)
        logits = self.score(z_grid).squeeze(-1)
        logits = logits.masked_fill(~token_visible, -1e4)
        weights = torch.softmax(logits, dim=3) * visible
        denom = weights.sum(dim=3, keepdim=True)
        uniform = torch.full_like(weights, 1.0 / weights.shape[3])
        weights = torch.where(denom > 0, weights / denom.clamp(min=1e-8), uniform)
        return (z_grid * weights[..., None]).sum(dim=3)
