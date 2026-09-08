"""光谱维分组 G_j 与谱段中心位置。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpectralGroups:
    """将 S 个波段均分为 n_sp 组连续谱段。"""

    num_bands: int
    num_groups: int

    def __post_init__(self) -> None:
        if self.num_bands % self.num_groups != 0:
            raise ValueError(
                f"S={self.num_bands} must be divisible by n_sp={self.num_groups}"
            )

    @property
    def patch_size(self) -> int:
        return self.num_bands // self.num_groups

    @property
    def group_indices(self) -> list[np.ndarray]:
        """每组波段索引，长度 n_sp，元素 shape (s_p,)。"""
        sp = self.patch_size
        return [
            np.arange(j * sp, (j + 1) * sp, dtype=np.int64)
            for j in range(self.num_groups)
        ]

    def spectral_centers(self) -> np.ndarray:
        """
        归一化谱段中心 λ_j^c / λ_max ∈ (0, 1]，长度 n_sp。
        无波长表时用等间距索引中心。
        """
        sp = self.patch_size
        centers = (np.arange(self.num_groups, dtype=np.float64) + 0.5) * sp
        return (centers / self.num_bands).astype(np.float32)

    def band_to_group(self) -> np.ndarray:
        """(S,) int，每个波段所属组 j。"""
        sp = self.patch_size
        return np.arange(self.num_bands, dtype=np.int64) // sp
