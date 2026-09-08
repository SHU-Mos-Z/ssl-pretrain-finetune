"""固定端元 E* 下的像素级 OD / 强度重建（无可学习参数）。"""

import torch


def reconstruct_od(c: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """
    线性解混：OD = C · E*。

    Args:
        c: (B, K, H, W) 像素级丰度
        e: (K, S) 或 (B, K, S)，内部 detach
    Returns:
        od_hat: (B, S, H, W)
    """
    e = e.detach()
    if e.dim() == 2:
        return torch.einsum("bkhw,ks->bshw", c, e)
    return torch.einsum("bkhw,bks->bshw", c, e)


def reconstruct_intensity(od: torch.Tensor, od_max: float = 3.0) -> torch.Tensor:
    """
    比尔–朗伯逆变换：I = exp(-OD)。

    Args:
        od: (B, S, H, W)
    Returns:
        i_hat: (B, S, H, W)
    """
    return torch.exp(-od.clamp(min=0.0, max=od_max))
