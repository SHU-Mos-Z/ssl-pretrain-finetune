"""
梯度引导掩膜工具（3D 空–谱 Token 评分 + 课程采样）。

方法文档 Step 3：Token (u,v,j) 难度分数 h_{uvj}。
"""

from __future__ import annotations

import numpy as np

GRAD_METHODS = ("mean", "vector", "3d", "pca", "di_zenzo")


def _per_band_spatial_gradients(od_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import sobel

    c = od_np.shape[0]
    gx = np.stack([sobel(od_np[b].astype(np.float64), axis=1) for b in range(c)])
    gy = np.stack([sobel(od_np[b].astype(np.float64), axis=0) for b in range(c)])
    return gx, gy


def gradient_3d_spatial_spectral(od_np: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """(S,H,W) → (H,W) 像素级空–谱联合梯度幅值。"""
    gx, gy = _per_band_spatial_gradients(od_np)
    d_lambda = np.gradient(od_np.astype(np.float64), axis=0)
    g2 = gx ** 2 + gy ** 2 + alpha * d_lambda ** 2
    return np.sqrt(g2.sum(axis=0)).astype(np.float32)


def compute_gradient_map(
    od_np: np.ndarray,
    method: str = "3d",
    spectral_alpha: float = 1.0,
    pca_components: int = 3,
) -> np.ndarray:
    """(S,H,W) → (H,W) 梯度图 dispatcher。"""
    if method == "3d":
        return gradient_3d_spatial_spectral(od_np, alpha=spectral_alpha)
    if method == "vector":
        gx, gy = _per_band_spatial_gradients(od_np)
        return np.sqrt((gx ** 2 + gy ** 2).mean(axis=0)).astype(np.float32)
    if method == "mean":
        from scipy.ndimage import sobel
        gray = od_np.mean(axis=0)
        gx = sobel(gray.astype(np.float64), axis=1)
        gy = sobel(gray.astype(np.float64), axis=0)
        return np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)
    raise ValueError(f"grad method '{method}' not implemented in slim utils; use 3d/vector/mean")


def token_3d_gradient_scores(
    od: np.ndarray,
    patch_size: int,
    group_indices: list[np.ndarray],
    alpha: float = 1.0,
) -> np.ndarray:
    """
    按方法文档 Step 3.1 计算每个 Token 的 h_{uvj}。

    Args:
        od: (S, H, W)
        group_indices: n_sp 个 (s_p,) 波段索引
    Returns:
        scores: (T,) float32
    """
    s, h, w = od.shape
    h_p, w_p = h // patch_size, w // patch_size
    n_sp = len(group_indices)
    t = h_p * w_p * n_sp
    scores = np.zeros(t, dtype=np.float32)

    gx, gy = _per_band_spatial_gradients(od)
    # 光谱差分（组内）
    for u in range(h_p):
        y0, y1 = u * patch_size, (u + 1) * patch_size
        for v in range(w_p):
            x0, x1 = v * patch_size, (v + 1) * patch_size
            for j, bands in enumerate(group_indices):
                t_idx = u * (w_p * n_sp) + v * n_sp + j
                g2 = np.zeros((patch_size, patch_size), dtype=np.float64)
                band_od = od[bands, y0:y1, x0:x1]
                for local_i, b in enumerate(bands):
                    g2 += gx[b, y0:y1, x0:x1] ** 2 + gy[b, y0:y1, x0:x1] ** 2
                    if len(bands) > 1:
                        if local_i < len(bands) - 1:
                            diff = band_od[local_i + 1] - band_od[local_i]
                            g2 += alpha * diff ** 2
                scores[t_idx] = float(np.sqrt(g2).mean())
    return scores


def curriculum_weight(epoch: int, total_epochs: int) -> float:
    """γ(t) = epoch / total_epochs。"""
    return float(epoch) / max(total_epochs, 1)


def weighted_sample_no_replace(
    scores: np.ndarray,
    n_mask: int,
    tau: float,
    gamma: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """无放回采样 n_mask 个 Token 索引。"""
    rng = rng or np.random.default_rng()
    n = len(scores)
    n_mask = min(n_mask, n)
    p_uniform = np.ones(n, dtype=np.float64) / n
    s = scores.astype(np.float64) / (tau + 1e-8)
    s -= s.max()
    exp_s = np.exp(s)
    p_grad = exp_s / (exp_s.sum() + 1e-12)
    p_final = (1.0 - gamma) * p_uniform + gamma * p_grad
    p_final /= p_final.sum()
    return rng.choice(n, size=n_mask, replace=False, p=p_final)
