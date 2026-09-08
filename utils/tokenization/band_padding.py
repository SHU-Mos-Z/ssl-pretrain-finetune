"""波段数对齐：交替向外复制首/末波段，使波段数可被 spectral_patch_size 整除。

部分数据集的实际波段数 S 无法整除 spectral_patch_size（例如 S=32,
spectral_patch_size=5），导致 Step 2 的谱段组 Token 化（`TokenBuildConfig
.num_groups_for` / `SpectralGroups`）在构造阶段直接报错。这里在不重新采样、
不插值的前提下，通过复制边界波段把 S 补齐到最近的整数倍：

* 复制的波段严格来自输入自身的第 0 波段（最低波长）与最后一个波段
  （最高波长，即“最大波段”），不引入新的光谱形状；
* 当需要补齐的波段数为偶数时，两端各补一半；为奇数时低波长端多补 1
  个，交替顺序等价于先在低波长端复制一次、再在高波长端复制一次，如此
  往复，直至补满。

该 padding 需在 Step 2 Token 化（`build_tokens`）之前，对 OD/强度立方、
逐图端元矩阵 E*、波长表三者同步应用，以保持波段轴对齐。
"""

from __future__ import annotations

import numpy as np


def band_pad_amounts(num_bands: int, spectral_patch_size: int) -> tuple[int, int]:
    """返回 (front, back)：为使 num_bands 可被 spectral_patch_size 整除，
    需要在波段轴的低波长端（front，复制第 0 波段）与高波长端
    （back，复制最后一个波段）各追加多少个波段。

    两端交替追加（先低波长端，再高波长端，如此往复），故 front 与 back
    最多相差 1；无需补齐时返回 (0, 0)。
    """
    if spectral_patch_size <= 0:
        raise ValueError("spectral_patch_size must be positive")
    if num_bands <= 0:
        raise ValueError("num_bands must be positive")
    remainder = num_bands % spectral_patch_size
    if remainder == 0:
        return 0, 0
    pad_total = spectral_patch_size - remainder
    front = (pad_total + 1) // 2
    back = pad_total // 2
    return front, back


def pad_bands(array: np.ndarray, spectral_patch_size: int, axis: int = 0) -> np.ndarray:
    """沿 `axis` 复制边界波段（`mode="edge"`），使该轴长度可被
    `spectral_patch_size` 整除；恰好整除时原样返回（不复制数组）。

    Args:
        array: 任意维度数组，`axis` 维为波段维（例如 OD/强度立方的第 0
            维、端元矩阵的第 1 维、波长表的第 0 维）。
        spectral_patch_size: 谱段组大小 s_p。
        axis: 波段所在的维度。
    """
    num_bands = array.shape[axis]
    front, back = band_pad_amounts(num_bands, spectral_patch_size)
    if front == 0 and back == 0:
        return array
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (front, back)
    return np.pad(array, pad_width, mode="edge")
