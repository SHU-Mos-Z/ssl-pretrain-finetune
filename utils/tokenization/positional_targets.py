"""Token 位置编码与 NMF 教师丰度目标。"""

from __future__ import annotations

import numpy as np

from utils.tokenization.patch_tokens import flatten_index, grid_sizes
from utils.tokenization.spectral_groups import SpectralGroups


def patch_center_xy(u: int, v: int, patch_size: int, h: int, w: int) -> tuple[float, float]:
    """
    归一化 Patch 中心坐标 (x/W, y/H)。
    u → 行(y)，v → 列(x)，与 models 测试代码一致。
    """
    x_c = (v + 0.5) * patch_size / w
    y_c = (u + 0.5) * patch_size / h
    return x_c, y_c


def patch_center_pixel(u: int, v: int, patch_size: int) -> tuple[int, int]:
    """Patch (u,v) 中心像素坐标 (h_idx, w_idx)。u 为行索引，v 为列索引。"""
    h_idx = u * patch_size + patch_size // 2
    w_idx = v * patch_size + patch_size // 2
    return h_idx, w_idx


def compute_pe_spatial(
    h: int,
    w: int,
    patch_size: int,
    spectral_groups: SpectralGroups,
) -> np.ndarray:
    """pe_spatial (T, 2)，最后一维 (x/W, y/H)。"""
    h_p, w_p = grid_sizes(h, w, patch_size)
    n_sp = spectral_groups.num_groups
    t = h_p * w_p * n_sp
    pe = np.zeros((t, 2), dtype=np.float32)
    for u in range(h_p):
        for v in range(w_p):
            x_c, y_c = patch_center_xy(u, v, patch_size, h, w)
            for j in range(n_sp):
                t_idx = flatten_index(u, v, j, w_p, n_sp)
                pe[t_idx, 0] = x_c
                pe[t_idx, 1] = y_c
    return pe


def compute_pe_spectral(spectral_groups: SpectralGroups, h: int, w: int, patch_size: int) -> np.ndarray:
    """pe_spectral (T,) 每个 Token 的谱段组中心。"""
    h_p, w_p = grid_sizes(h, w, patch_size)
    n_sp = spectral_groups.num_groups
    centers = spectral_groups.spectral_centers()
    t = h_p * w_p * n_sp
    pe = np.zeros(t, dtype=np.float32)
    for u in range(h_p):
        for v in range(w_p):
            for j in range(n_sp):
                t_idx = flatten_index(u, v, j, w_p, n_sp)
                pe[t_idx] = centers[j]
    return pe


def compute_pe_abund_from_map(
    c_star: np.ndarray,
    patch_size: int,
    n_sp: int,
) -> np.ndarray:
    """pe_abund (T, K)：Patch 内 C* 均值（与谱段 j 无关，各 j 共享）。"""
    k, h, w = c_star.shape
    h_p, w_p = grid_sizes(h, w, patch_size)
    t = h_p * w_p * n_sp
    pe = np.zeros((t, k), dtype=np.float32)
    for u in range(h_p):
        y0, y1 = u * patch_size, (u + 1) * patch_size
        for v in range(w_p):
            x0, x1 = v * patch_size, (v + 1) * patch_size
            c_mean = c_star[:, y0:y1, x0:x1].mean(axis=(1, 2))  # (K,)
            for j in range(n_sp):
                t_idx = flatten_index(u, v, j, w_p, n_sp)
                pe[t_idx] = c_mean
    return pe


def compute_c_star_patch(c_star: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Patch 内所有 P×P 像素的丰度展平，供 DINO-style Token 级一致性损失。
    将 (K, P, P) 转置为 (P, P, K) 再展平，得到 (P*P*K,) 的教师向量。

    Returns:
        c_star_patch: (H_p, W_p, P*P*K)
    """
    k, h, w = c_star.shape
    h_p, w_p = grid_sizes(h, w, patch_size)
    flat_dim = patch_size * patch_size * k
    out = np.zeros((h_p, w_p, flat_dim), dtype=np.float32)
    for u in range(h_p):
        y0, y1 = u * patch_size, (u + 1) * patch_size
        for v in range(w_p):
            x0, x1 = v * patch_size, (v + 1) * patch_size
            patch_c = c_star[:, y0:y1, x0:x1]               # (K, P, P)
            out[u, v] = patch_c.transpose(1, 2, 0).reshape(-1)  # (P*P*K,)
    return out
