"""规则 Patch × 谱段组 Token 嵌入（Step 4）。"""

import torch
import torch.nn as nn

from models.modules_vit.positional_encoding import SinePositionalEncoding


class TokenEncoder(nn.Module):
    """
    可见 Token：Linear(P*P*s_p → D)（标准 ViT 式展平投影）；掩膜 Token：可学习 [MASK]。
    加 spatial + spectral 位置编码（不注入丰度先验，避免预训练信息泄露）。
    """

    def __init__(
        self,
        embed_dim: int,
        spectral_patch_size: int,
        patch_size: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        token_in_dim = patch_size * patch_size * spectral_patch_size
        self.patch_proj = nn.Linear(token_in_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.pos_enc = SinePositionalEncoding(embed_dim)

    def forward(
        self,
        token_raw: torch.Tensor,
        is_masked: torch.Tensor,
        pe_spatial: torch.Tensor,
        pe_spectral: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            token_raw:   (B, T, P*P*s_p)
            is_masked:   (B, T) bool
            pe_spatial:  (B, T, 2)
            pe_spectral: (B, T)
        Returns:
            x: (B, T, D)
        """
        feat = self.patch_proj(token_raw)
        mask = is_masked.unsqueeze(-1)
        feat = torch.where(mask, self.mask_token.expand_as(feat), feat)
        x = feat + self.pos_enc(pe_spatial, pe_spectral)
        return x
