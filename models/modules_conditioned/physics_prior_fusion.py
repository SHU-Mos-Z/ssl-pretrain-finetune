"""Inject coarse abundance and reliability evidence into OD tokens."""

from __future__ import annotations

import torch
import torch.nn as nn


class PhysicsPriorFusion(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.rho_encoder = nn.Sequential(nn.Linear(1, embed_dim), nn.Sigmoid())
        self.gate = nn.Sequential(
            nn.Linear(2 * embed_dim + 1, embed_dim), nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        endmember_evidence: torch.Tensor,
        endmember_tokens: torch.Tensor,
        c0_low: torch.Tensor,
        rho_low: torch.Tensor,
        num_groups: int,
    ) -> torch.Tensor:
        b, _, d = tokens.shape
        c_prior = torch.einsum("bkhw,bkd->bhwd", c0_low, endmember_tokens)
        c_prior = c_prior.unsqueeze(3).expand(-1, -1, -1, num_groups, -1).reshape(b, -1, d)
        rho = rho_low.permute(0, 2, 3, 1)
        rho = rho.unsqueeze(3).expand(-1, -1, -1, num_groups, -1).reshape(b, -1, 1)
        gate = self.gate(torch.cat((tokens, endmember_evidence, rho), dim=-1))
        return self.norm(tokens + c_prior + self.rho_encoder(rho) + gate * endmember_evidence)
