"""Token mask → 像素–波段 m_pix (S,H,W)，0=参与重建损失。"""

from __future__ import annotations

import numpy as np

from utils.tokenization.patch_tokens import grid_sizes, unflatten_index
from utils.tokenization.spectral_groups import SpectralGroups


def expand_token_mask_to_m_pix(
    is_masked: np.ndarray,
    h: int,
    w: int,
    patch_size: int,
    spectral_groups: SpectralGroups,
) -> np.ndarray:
    """
    将 Token 级 is_masked 扩展为 m_pix (S, H, W)。

    约定（与方法文档 Step 3.3 一致）：
      m_pix[s,y,x] = 1  可见，不参与重建损失
      m_pix[s,y,x] = 0  被 mask，参与 L_recon^OD / L_recon^I

    Args:
        is_masked: (T,) bool
    Returns:
        m_pix: (S, H, W) float32，取值 {0, 1}
    """
    s = spectral_groups.num_bands
    h_p, w_p = grid_sizes(h, w, patch_size)
    n_sp = spectral_groups.num_groups
    m_pix = np.ones((s, h, w), dtype=np.float32)

    for t_idx, masked in enumerate(is_masked):
        if not masked:
            continue
        u, v, j = unflatten_index(t_idx, w_p, n_sp)
        y0, y1 = u * patch_size, (u + 1) * patch_size
        x0, x1 = v * patch_size, (v + 1) * patch_size
        bands = spectral_groups.group_indices[j]
        m_pix[np.ix_(bands, np.arange(y0, y1), np.arange(x0, x1))] = 0.0
    return m_pix


def _self_test() -> None:
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from utils.masking.token3d_masker import MaskConfig, sample_token_mask
    from utils.tokenization.spectral_groups import SpectralGroups

    print(f"\n{'=' * 72}")
    print("  m_pix_expand + token3d_masker 自检")
    print(f"{'=' * 72}")

    od = np.random.randn(60, 64, 64).astype(np.float32) ** 2
    groups = SpectralGroups(60, 6)
    cfg = MaskConfig(mask_ratio=0.25, use_gradient=True)
    is_masked = sample_token_mask(od, 16, groups, epoch=50, total_epochs=100, cfg=cfg, rng=np.random.default_rng(0))
    m_pix = expand_token_mask_to_m_pix(is_masked, 64, 64, 16, groups)

    t = is_masked.size
    n_masked = int(is_masked.sum())
    loss_voxels = int((m_pix == 0).sum())
    print(f"  T={t}, masked tokens={n_masked}")
    print(f"  is_masked shape    {is_masked.shape}")
    print(f"  m_pix shape        {m_pix.shape}")
    print(f"  loss voxels (m=0)  {loss_voxels}")
    print(f"  expected approx    {n_masked * 16 * 16 * 10} (= tokens * P^2 * s_p)")
    assert m_pix.shape == (60, 64, 64)
    assert loss_voxels == n_masked * 16 * 16 * 10
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    _self_test()
