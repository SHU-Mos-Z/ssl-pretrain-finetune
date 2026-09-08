"""Step 3：Token 级 3D 梯度引导掩膜。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.masking._grad_utils import (
    curriculum_weight,
    token_3d_gradient_scores,
    weighted_sample_no_replace,
)
from utils.tokenization.spectral_groups import SpectralGroups


@dataclass
class MaskConfig:
    mask_ratio: float = 0.4
    tau: float = 1.0
    spectral_alpha: float = 1.0
    use_gradient: bool = True


def sample_token_mask(
    od: np.ndarray,
    patch_size: int,
    spectral_groups: SpectralGroups,
    epoch: int,
    total_epochs: int,
    cfg: MaskConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    采样 Token 级 is_masked (T,) bool，True=被 mask。

    Args:
        od: (S, H, W)
    Returns:
        is_masked: (T,) bool
    """
    rng = rng or np.random.default_rng()
    s, h, w = od.shape
    h_p, w_p = h // patch_size, w // patch_size
    n_sp = spectral_groups.num_groups
    t = h_p * w_p * n_sp
    n_mask = int(round(cfg.mask_ratio * t))
    n_mask = max(0, min(n_mask, t))

    is_masked = np.zeros(t, dtype=bool)
    if n_mask == 0:
        return is_masked

    if cfg.use_gradient:
        scores = token_3d_gradient_scores(
            od, patch_size, spectral_groups.group_indices, alpha=cfg.spectral_alpha,
        )
        gamma = curriculum_weight(epoch, total_epochs)
        chosen = weighted_sample_no_replace(scores, n_mask, cfg.tau, gamma, rng=rng)
    else:
        chosen = rng.choice(t, size=n_mask, replace=False)

    is_masked[chosen] = True
    return is_masked
