"""Dynamic endmember-conditioned residual abundance prediction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ResidualAbundanceHead(nn.Module):
    def __init__(self, feature_dim: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.feature_projection = nn.Conv2d(feature_dim, hidden_dim, 1)
        self.endmember_projection = nn.Linear(embed_dim, hidden_dim)
        self.shared_residual = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        features: torch.Tensor,
        endmember_tokens: torch.Tensor,
        c0: torch.Tensor,
        rho: torch.Tensor,
        alpha_min: float,
        alpha_extra: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.feature_projection(features)
        e = self.endmember_projection(endmember_tokens)
        interaction = torch.einsum("bdhw,bkd->bkhw", z, e) / math.sqrt(z.shape[1])
        inputs = torch.stack((interaction, c0, rho.expand_as(c0)), dim=-1)
        delta = self.shared_residual(inputs).squeeze(-1)
        alpha = alpha_min + alpha_extra * (1.0 - rho)
        logits = torch.log(c0.clamp(min=1e-8)) + alpha * torch.tanh(delta)
        return delta, torch.softmax(logits, dim=1)
