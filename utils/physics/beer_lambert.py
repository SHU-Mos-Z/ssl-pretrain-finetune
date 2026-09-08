"""
比尔–朗伯定律的物理变换工具函数。

数据约定：
  本项目的 .npy 文件中，每个像素值已在全数据集层面归一化到 [0, 1]，
  即存储值为透射率 T = I / I_global_max。
  因此在归一化域内，参考入射光强 I_0 ≡ 1.0，OD 转换退化为：
      OD = -log(T + eps)
  对应的强度逆映射为：
      T_hat = exp(-OD_hat)
"""

from __future__ import annotations

import numpy as np
import torch

_OD_MIN = 0.0
_OD_MAX = 3.0


def intensity_to_od(
    I: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """将归一化透射率转换为 OD（任意 shape）。"""
    return -torch.log(I.clamp(min=eps))


def od_to_intensity(OD: torch.Tensor) -> torch.Tensor:
    """OD → 归一化透射率。"""
    return torch.exp(-OD)


def clamp_od(
    OD: torch.Tensor,
    lo: float = _OD_MIN,
    hi: float = _OD_MAX,
) -> torch.Tensor:
    """将 OD 截断到物理合理范围。"""
    return OD.clamp(min=lo, max=hi)


def intensity_to_od_np(I: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """NumPy 版：透射率 → OD。"""
    return -np.log(np.clip(I, eps, None))


def od_to_intensity_np(od: np.ndarray) -> np.ndarray:
    """NumPy 版：OD → 透射率。"""
    return np.exp(-od)
