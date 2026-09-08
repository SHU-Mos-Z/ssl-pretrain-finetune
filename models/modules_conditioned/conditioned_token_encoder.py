"""OD token embedding with early endmember cross-attention."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.modules_vit.positional_encoding import SinePositionalEncoding


class ConditionedTokenEncoder(nn.Module):
    def __init__(
        self,
        patch_size: int,
        spectral_patch_size: int,
        embed_dim: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        token_dim = patch_size * patch_size * spectral_patch_size
        self.patch_projection = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, embed_dim)
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.position = SinePositionalEncoding(embed_dim)
        self.query_norm = nn.LayerNorm(embed_dim)
        self.endmember_norm = nn.LayerNorm(embed_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        token_raw: torch.Tensor,
        token_visible: torch.Tensor,
        pe_spatial: torch.Tensor,
        pe_spectral: torch.Tensor,
        endmember_group_tokens: torch.Tensor,
        h_p: int,
        w_p: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = token_raw.shape
        _, k, g, d = endmember_group_tokens.shape
        if t != h_p * w_p * g:
            raise ValueError("token count is inconsistent with patch grid and spectral groups")
        if token_visible.shape == (b, h_p, w_p, g):
            token_visible = token_visible.reshape(b, t)
        if token_visible.shape != (b, t):
            raise ValueError("token_visible must have shape (B,T) or (B,Hp,Wp,G)")

        raw = self.patch_projection(token_raw)
        raw = torch.where(token_visible[..., None], raw, self.mask_token.expand(b, t, d))
        raw = raw + self.position(pe_spatial, pe_spectral)

        # Flatten order is (spatial patch, spectral group); each OD token attends
        # only to the K endmembers in its corresponding spectral group.
        evidence = (
            endmember_group_tokens.permute(0, 2, 1, 3)
            .unsqueeze(1)
            .expand(-1, h_p * w_p, -1, -1, -1)
            .reshape(b * t, k, d)
        )
        query = self.query_norm(raw).reshape(b * t, 1, d)
        conditioned, _ = self.cross_attention(
            query, self.endmember_norm(evidence), self.endmember_norm(evidence)
        )
        conditioned = conditioned.reshape(b, t, d)
        return self.output_norm(raw + conditioned), conditioned
