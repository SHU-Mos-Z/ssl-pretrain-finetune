"""正弦位置编码（1D 光谱 / 2D 空间）。"""

import math

import torch
import torch.nn as nn


def _sine_embed(values: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Args:
        values: (..., C)  C=1 为 1D，C=2 为 2D
        dim: 输出最后一维大小，须为偶数
    Returns:
        (..., dim)
    """
    assert dim % 2 == 0, f"sine embed dim must be even, got {dim}"
    *batch, c_in = values.shape
    n_freq = dim // (2 * c_in)
    freq = torch.arange(n_freq, device=values.device, dtype=values.dtype)
    freq = 1.0 / (10000 ** (freq / max(n_freq, 1)))

    # (..., c_in, n_freq)
    angles = values.unsqueeze(-1) * freq
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (..., c_in, 2*n_freq)
    return emb.reshape(*batch, dim)


def sine_embed_1d(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Args:
        x: (...,) 或 (..., 1)，归一化波长/谱段位置
    Returns:
        (..., dim)
    """
    if x.dim() == 0 or x.shape[-1] != 1:
        x = x.unsqueeze(-1)
    return _sine_embed(x, dim)


def sine_embed_2d(xy: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Args:
        xy: (..., 2)，归一化 (x/W, y/H)
    Returns:
        (..., dim)
    """
    assert xy.shape[-1] == 2, f"expected last dim 2, got {xy.shape[-1]}"
    return _sine_embed(xy, dim)


class SinePositionalEncoding(nn.Module):
    """批量 Token 位置编码：spatial + spectral 相加。"""

    def __init__(self, embed_dim: int):
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim

    def forward(
        self,
        pe_spatial: torch.Tensor,
        pe_spectral: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pe_spatial:  (B, T, 2)
            pe_spectral: (B, T) 或 (B, T, 1)
        Returns:
            (B, T, D)
        """
        pe_s = sine_embed_2d(pe_spatial, self.embed_dim)
        pe_l = sine_embed_1d(pe_spectral, self.embed_dim)
        return pe_s + pe_l
