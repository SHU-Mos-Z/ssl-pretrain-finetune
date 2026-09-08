"""Reliability-gated bidirectional fusion of ViT and CNN features."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.modules_cinet.ciam import CIAM


class ConfidenceGatedFusion(nn.Module):
    def __init__(self, cnn_dim: int, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.cnn_projection = nn.Linear(cnn_dim, embed_dim)
        self.ciam = CIAM(embed_dim, num_heads, dropout)
        self.gate = nn.Sequential(
            nn.Linear(2 * embed_dim + 1, embed_dim), nn.Sigmoid()
        )

    def forward(
        self,
        z_vit: torch.Tensor,
        e_cnn: torch.Tensor,
        rho_low: torch.Tensor,
        num_groups: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h_p, w_p = e_cnn.shape
        cnn_tokens = self.cnn_projection(
            e_cnn.permute(0, 2, 3, 1).reshape(b, h_p * w_p, c)
        )
        z_cross, cnn_semantic = self.ciam(z_vit, cnn_tokens)
        rho = rho_low.permute(0, 2, 3, 1)
        rho = rho.unsqueeze(3).expand(-1, -1, -1, num_groups, -1).reshape(b, -1, 1)
        gate = self.gate(torch.cat((z_vit, z_cross, rho), dim=-1))
        return z_vit + gate * (z_cross - z_vit), cnn_semantic
