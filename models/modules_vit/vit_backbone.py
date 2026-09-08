"""ViT 编码器：Pre-LN Transformer，输入 (B, T, D) 序列。"""

import torch
import torch.nn as nn


class ViTBackbone(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        dim_ff = int(embed_dim * mlp_ratio)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
            key_padding_mask: (B, T) bool，True=忽略（预留，固定 T 时通常为 None）
        Returns:
            (B, T, D)
        """
        x = self.blocks(x, src_key_padding_mask=key_padding_mask)
        return self.norm(x)
