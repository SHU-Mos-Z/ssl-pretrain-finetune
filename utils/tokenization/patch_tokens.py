"""规则空间 Patch × 谱段组 Token 的 OD 均值特征。"""

from __future__ import annotations

import numpy as np

from utils.tokenization.spectral_groups import SpectralGroups


def grid_sizes(h: int, w: int, patch_size: int) -> tuple[int, int]:
    if h % patch_size or w % patch_size:
        raise ValueError(f"H={h}, W={w} must be divisible by P={patch_size}")
    return h // patch_size, w // patch_size


def flatten_index(u: int, v: int, j: int, w_p: int, n_sp: int) -> int:
    """t = u * (W_p * n_sp) + v * n_sp + j，与 models.seq_to_grid 一致。"""
    return u * (w_p * n_sp) + v * n_sp + j


def unflatten_index(t: int, w_p: int, n_sp: int) -> tuple[int, int, int]:
    u = t // (w_p * n_sp)
    rem = t % (w_p * n_sp)
    v = rem // n_sp
    j = rem % n_sp
    return u, v, j


def compute_patch_token_raw(
    od: np.ndarray,
    patch_size: int,
    spectral_groups: SpectralGroups,
) -> np.ndarray:
    """
    计算 token_raw (T, P*P*s_p)：每个 Token 为 Patch 内对应谱段组的
    空间-谱段体素展平（(s_p, P, P) → 转置为 (P, P, s_p) → 展平），
    与标准 ViT Patch Embedding 保持一致。

    Args:
        od: (S, H, W)
    Returns:
        token_raw: (T, P*P*s_p)
    """
    s, h, w = od.shape
    if s != spectral_groups.num_bands:
        raise ValueError(f"OD bands {s} != spectral groups {spectral_groups.num_bands}")

    h_p, w_p = grid_sizes(h, w, patch_size)
    n_sp = spectral_groups.num_groups
    sp = spectral_groups.patch_size
    token_dim = patch_size * patch_size * sp
    t = h_p * w_p * n_sp
    token_raw = np.zeros((t, token_dim), dtype=np.float32)

    for u in range(h_p):
        y0, y1 = u * patch_size, (u + 1) * patch_size
        for v in range(w_p):
            x0, x1 = v * patch_size, (v + 1) * patch_size
            patch_od = od[:, y0:y1, x0:x1]          # (S, P, P)
            for j, band_idx in enumerate(spectral_groups.group_indices):
                t_idx = flatten_index(u, v, j, w_p, n_sp)
                group_cube = patch_od[band_idx]      # (s_p, P, P)
                # 转置为 (P, P, s_p) 再展平 → (P*P*s_p,)
                token_raw[t_idx] = group_cube.transpose(1, 2, 0).reshape(-1)
    return token_raw
