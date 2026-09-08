"""
Step 0：离线正则化 NMF（同步 MUR）—— GPU 加速版（offline_nmf_optimized.py 的副本）。

对单张 OD 立方 Y ∈ R^{S×H×W} 做无掩膜分解，输出：
  C* ∈ R^{K×H×W}  （行端元惯例下的像素丰度）
  E* ∈ R^{K×S}      （行端元惯例下的端元矩阵）

内部优化使用列端元惯例：Y ≈ C @ E^T，C ∈ R^{L×K}，E ∈ R^{S×K}。

本文件基于 offline_nmf_optimized.py（已包含 #1/#2 两项无损性能优化）继续
落实第三项优化：

  #1 消除每次 MUR 迭代中 `_objective()` 内部对 `c @ e.T` 的重复计算
     （复用收敛判断已经算出的 y_hat/resid），批处理路径默认直接跳过
     只用于打印/自检的 `_objective()` 调用（`track_objective=False`）。
  #2 批处理路径默认跳过 `stats_array()` 对 C*/E* 的全量 percentile 统计
     （单线程、随空间尺寸线性增长，且从未被批处理逻辑消费），
     仅在需要自检打印时才计算（`compute_stats=True`）。
  #6（本文件新增）单图 GPU 化：`fit_regularized_nmf` 内部的 MUR 迭代循环
     用 PyTorch 张量在 GPU（CUDA）上执行，而不是 numpy/CPU-BLAS。算法每步
     的核心开销都是稠密矩阵乘（Y(L,S)@E(S,K) 等，L 为像素数、动辄百万级，
     S/K 只有十几到二十），这种"瘦高"矩阵乘正是 GPU（cuBLAS + 高显存带宽）
     远胜于 CPU 多线程 BLAS 的场景。整张图的数据只在"进/出"两端搬运一次
     （进 GPU 一次、出 GPU 一次），中间上千步迭代全部在显存里完成。

  与 offline_nmf_optimized.py 的区别只在于 `fit_regularized_nmf` 内部换成
  torch 实现（外层的读图/OD 转换/落盘/统计逻辑完全不变）；由于 GPU 上默认用
  float32（消费级卡 fp64 吞吐远低于 fp32），数值上不再是与 CPU/float64 版本
  比特级一致，而是在合理容差内一致——需要用 tests/benchmark_offline_nmf_optimizations.py
  这类脚本对比重建 MSE 加以验证，不能像 #1/#2/#5 那样直接断言逐位相同。

  出于同样的原因，`process_data_root` 的进程级并行开关（`parallel_workers`，
  对应 CPU 版本里的 #5）在 device 为 CUDA 时默认禁用：单张 GPU 已经通过硬件
  并行吃满了单图内部的矩阵乘，多个进程同时抢同一张卡通常没有收益，反而增加
  CUDA 上下文初始化开销；如果要跨多张卡并行，需要在外层分别为每个进程指定
  不同的 CUDA_VISIBLE_DEVICES，不在本文件自动处理范围内。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.physics.beer_lambert import intensity_to_od_np
from utils.sample_exclusion import load_excluded_samples


def _auto_select_cuda_device() -> torch.device:
    """
    在所有可见 CUDA 设备里选一块当前剩余显存最多的（而不是固定用 cuda:0）。

    共享多卡机器上，"未指定设备"如果直接默认 cuda:0，很容易撞上正好被别的
    用户任务占满的那一块卡，即使本程序只想申请几十 MB 也会 OOM（见
    RuntimeError: CUDA error: out of memory，即便待分配的张量本身很小——
    这通常说明目标卡剩余显存已经很紧，连 CUDA 上下文/cuBLAS 初始化的固定
    开销都放不下）。这里用 `torch.cuda.mem_get_info` 实时查询每块可见卡的
    空闲显存，自动挑最空的一块，减少人工指定 GPU_ID/CUDA_VISIBLE_DEVICES
    的负担；仍然可以通过显式传 "cuda:N" 精确指定某一块卡。
    """
    n = torch.cuda.device_count()
    if n <= 1:
        return torch.device("cuda")
    best_idx, best_free = 0, -1
    for i in range(n):
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(i)
        except Exception:
            free_bytes = -1
        if free_bytes > best_free:
            best_free, best_idx = free_bytes, i
    print(
        f"  [自动选卡] 可见 {n} 块 CUDA 设备，选择剩余显存最多的 cuda:{best_idx} "
        f"（≈{best_free / 1024**3:.2f} GiB 空闲）"
    )
    return torch.device(f"cuda:{best_idx}")


def _resolve_device(device: str | torch.device | None) -> torch.device:
    """
    解析计算设备。

    device 为 None/""（未指定）或字面 "cuda"（未带具体索引）时：
        若 CUDA 可用，自动选择当前空闲显存最多的那块卡（见
        `_auto_select_cuda_device`），而不是不假思索地默认 cuda:0；
        CUDA 不可用则回退 CPU。
    device 为 "cuda:N" / "cpu" 等具体值时：按原样解析，不做自动选择。
    """
    if device is None or device == "" or device == "cuda":
        if torch.cuda.is_available():
            return _auto_select_cuda_device()
        return torch.device("cpu")
    return torch.device(device)


_DTYPE_ALIASES: dict[str, torch.dtype] = {
    "float32": torch.float32, "fp32": torch.float32, "f32": torch.float32,
    "float64": torch.float64, "fp64": torch.float64, "f64": torch.float64,
}


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """解析计算精度：GPU（尤其消费级显卡）建议使用 float32。"""
    if isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError(f"不支持的 dtype: {dtype!r}，可选 float32/float64")
    return _DTYPE_ALIASES[key]


@dataclass
class NMFResult:
    """NMF 分解结果（行端元输出惯例）。"""

    c_star: np.ndarray          # (K, H, W)
    e_star: np.ndarray          # (K, S)
    c_flat: np.ndarray          # (L, K) 列端元内部表示
    e_col: np.ndarray           # (S, K) 列端元内部表示
    y_hat: np.ndarray           # (L, S) 重建 OD（展平）
    od_cube: np.ndarray         # (S, H, W) 输入 OD
    metrics: dict[str, float] = field(default_factory=dict)
    c_stats: dict[str, float] = field(default_factory=dict)
    e_stats: dict[str, float] = field(default_factory=dict)
    n_iter: int = 0
    final_loss: float = 0.0


def cache_dir_name(
    k: int,
    l1: float,
    l2: float,
    l3: float,
    simplex: bool = False,
    lam_e: float = 0.0,
    e_clamp_max: float = 0.0,
) -> str:
    """
    NMF 缓存目录名。

    规则：
      - 基础名：nmf_cache_K{k}_l1{l1}_l2{l2}_l3{l3}
      - lam_e > 0 时追加 _le{lam_e}（E 的 L2 正则强度）
      - e_clamp_max > 0 时追加 _ec{e_clamp_max}（E 逐元素上界）
      - simplex=True 时末尾追加 _simplex

    例：k=16, l1=5e-4, l2=2e-4, l3=1e-2, lam_e=0.05, e_clamp_max=3, simplex=True
      → nmf_cache_K16_l15e-4_l22e-4_l31e-2_le0.05_ec3_simplex
    """
    def _fmt(x: float) -> str:
        if x == 0:
            return "0"
        exp = int(np.floor(np.log10(abs(x))))
        mant = x / (10 ** exp)
        return f"{mant:g}e{exp}"

    base = f"nmf_cache_K{k}_l1{_fmt(l1)}_l2{_fmt(l2)}_l3{_fmt(l3)}"
    if lam_e > 0.0:
        base += f"_le{lam_e:g}"
    if e_clamp_max > 0.0:
        base += f"_ec{e_clamp_max:g}"
    return base + ("_simplex" if simplex else "")


def load_intensity_cube(path: str | Path) -> np.ndarray:
    """
    读取磁盘 .npy，统一为 (S, H, W) float32。

    磁盘约定（与 old PretrainDataset 一致）：(H, W, S)。
    """
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected 3D cube, got shape {arr.shape} from {path}")
    if arr.shape[0] <= 64 and arr.shape[0] < arr.shape[1]:
        return arr
    return np.transpose(arr, (2, 0, 1))


def od_cube_to_Y(od: np.ndarray) -> tuple[np.ndarray, int, int]:
    """
    (S, H, W) → Y (L, S)，L = H * W，行优先展平空间维。
    """
    s, h, w = od.shape
    y = od.transpose(1, 2, 0).reshape(h * w, s)
    return y, h, w


def Y_to_od_cube(y: np.ndarray, h: int, w: int) -> np.ndarray:
    """Y (L, S) → (S, H, W)。"""
    s = y.shape[1]
    return y.reshape(h, w, s).transpose(2, 0, 1)


def reshape_C_to_map(c_flat: np.ndarray, h: int, w: int) -> np.ndarray:
    """C (L, K) → C* (K, H, W)。"""
    k = c_flat.shape[1]
    c_hwk = c_flat.reshape(h, w, k)
    return np.transpose(c_hwk, (2, 0, 1))


def e_col_to_row(e_col: np.ndarray) -> np.ndarray:
    """E (S, K) → E* (K, S)。"""
    return e_col.T.copy()


def _reflect_spectral_neighbors(e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """E (S, K) 光谱维镜像填充邻居。"""
    e_pad = np.pad(e, ((1, 1), (0, 0)), mode="reflect")
    return e_pad[:-2, :], e_pad[2:, :]


def _column_normalize_e(
    e: np.ndarray,
    c: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """列 L2 归一化 E，并将尺度同步乘到 C 对应列。"""
    norms = np.linalg.norm(e, axis=0, ord=2)
    norms = np.maximum(norms, eps)
    e_norm = e / norms
    c_scaled = c * norms
    return e_norm, c_scaled


def _project_simplex_rows(c: np.ndarray, eps: float) -> np.ndarray:
    """
    将 C (L, K) 的每一行投影到概率单纯形：ΣK = 1，c[l,k] ≥ 0。

    MUR 已保证 c ≥ 0，因此投影退化为行 L1 归一化：
        c_proj[l] = c[l] / sum(c[l])

    物理含义：c[l,k] 表示像素 l 中端元 k 的混合比例（丰度），
    所有端元比例之和为 1（完全混合模型）。
    """
    row_sums = c.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, eps)
    return c / row_sums


def _mul_update(
    base: np.ndarray,
    num: np.ndarray,
    den: np.ndarray,
    ratio_max: float,
    eps: float,
) -> np.ndarray:
    """乘性更新并裁剪比值，防止 MUR 单步发散。"""
    ratio = num / (den + eps)
    if ratio_max > 0:
        ratio = np.clip(ratio, 1.0 / ratio_max, ratio_max)
    return base * ratio


def _init_ce_from_svd(
    y: np.ndarray,
    k: int,
    eps: float,
    use_simplex: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    非负 SVD 风格初始化，使初始重建量级接近 Y。

    use_simplex=True 时：先列 L2 归一化 E（获取合理量级），
    再对 C 做 simplex 投影（c_init 行和 =1），令初始 E 自由承接 OD 量级。
    """
    u, s, vt = np.linalg.svd(y, full_matrices=False)
    s_k = np.maximum(s[:k], eps)
    c = np.abs(u[:, :k] * np.sqrt(s_k))
    e = np.abs(vt[:k, :].T * np.sqrt(s_k))
    if use_simplex:
        # 先归一化 E（稳定量级），再投影 C 到 simplex
        e, c = _column_normalize_e(e, c, eps)
        c = _project_simplex_rows(c, eps)
    else:
        e, c = _column_normalize_e(e, c, eps)
    return c, e


def _mur_step(
    c: np.ndarray,
    e: np.ndarray,
    y: np.ndarray,
    lam1: float,
    lam2: float,
    lam3: float,
    eps: float,
    ratio_max: float,
    update_mode: str,
    use_simplex: bool = False,
    lam_e: float = 0.0,
    e_clamp_max: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    单步 MUR 更新。

    update_mode:
      - "sync"       : C/E 均使用快照 (C^{(t)}, E^{(t)}) 并行计算（方法文档默认）
      - "alternating": 先更 C，再用 C^{(t+1)} 更 E（更稳，可选）

    use_simplex=False（默认）：
        原始行为，每步对 E 做列 L2 归一化，将尺度吸收到 C（C 量级可 >>1）。

    use_simplex=True：
        每步对 C 做行 L1 归一化（simplex 投影），将 C 约束在概率单纯形
        （每像素 ΣK = 1，c[l,k] ≥ 0）；E 不做列归一化，自由吸收 OD 量级。
        此时 C ∈ [0,1] 且 ΣK=1，与模型 softmax 输出语义一致。

    lam_e（>0 时激活，仅 simplex 模式下有意义）：
        对 E 加 L2 正则 λ_E·‖E‖²，等价于在 E MUR 分母加 λ_E·I。
        防止坍塌端元（C[:,k]≈0 导致 ctc[k,k]≈0）使 E[:,k] 趋于无穷。
        均衡点 E ≈ 数据信号 / (4λ2 + λ_E)，建议取 0.01~0.1。

    e_clamp_max（>0 时激活）：
        每步更新后将 E 逐元素裁剪到 [0, e_clamp_max]，作为物理上界兜底。
        OD 均值约为 0.25~1.17，单端元 OD 不应超过 od_max（建议与之相同）。
    """
    if update_mode not in ("sync", "alternating"):
        raise ValueError(f"update_mode must be sync or alternating, got {update_mode}")

    k = c.shape[1]
    ete = e.T @ e
    c_new = _mul_update(c, y @ e, c @ ete + lam1 / 2.0, ratio_max, eps)
    c_new = np.maximum(c_new, 0.0)

    if use_simplex:
        # simplex 模式：将 C 行投影到概率单纯形
        c_new = _project_simplex_rows(c_new, eps)

    c_for_e = c if update_mode == "sync" else c_new
    a = e.T @ e
    a_off_diag = a - np.diag(np.diag(a))
    e_left, e_right = _reflect_spectral_neighbors(e)
    ctc = c_for_e.T @ c_for_e
    num_e = y.T @ c_for_e + 2.0 * lam2 * (e_left + e_right)
    # lam_e > 0：在分母矩阵加 λ_E·I，防止坍塌端元的 ctc[k,k]→0 导致 E[:,k] 爆炸
    den_reg = ctc + 4.0 * lam2 * np.eye(k) + 4.0 * lam3 * a_off_diag
    if lam_e > 0.0:
        den_reg = den_reg + lam_e * np.eye(k)
    den_e = e @ den_reg
    e_hat = _mul_update(e, num_e, den_e, ratio_max, eps)
    e_hat = np.maximum(e_hat, 0.0)

    # e_clamp_max > 0：物理上界裁剪（OD 端元值不应超过 od_max）
    if e_clamp_max > 0.0:
        e_hat = np.minimum(e_hat, e_clamp_max)

    if use_simplex:
        # simplex 模式：E 自由演化，不做列归一化
        # E 吸收 OD 量级（确保 c_simplex @ E^T ≈ OD 成立）
        e_new = e_hat
    else:
        # 原始模式：列 L2 归一化 E，尺度转移到 C
        e_new, c_new = _column_normalize_e(e_hat, c_new, eps)

    return c_new, e_new


def _objective(
    y: np.ndarray,
    c: np.ndarray,
    e: np.ndarray,
    lam1: float,
    lam2: float,
    lam3: float,
    resid: np.ndarray | None = None,
) -> float:
    """完整正则化目标（Frobenius 重建 + L1 + 谱平滑 + 多样性）。

    优化 #1：调用方通常在收敛判断里已经算出
    ``resid = y - reconstruct_Y(c, e)``（形状 (L, S) 的大矩阵乘结果）。
    这里允许直接传入复用，避免再执行一次同样代价的 `c @ e.T`，
    数值上与原始实现完全一致（只是不重复计算而已）。
    """
    if resid is None:
        resid = y - c @ e.T
    recon = float(np.sum(resid ** 2))
    l1 = lam1 * float(np.sum(c))
    if e.shape[0] > 1:
        diff = e[1:, :] - e[:-1, :]
        smooth = lam2 * float(np.sum(diff ** 2))
    else:
        smooth = 0.0
    a = e.T @ e
    div = lam3 * float(np.sum(a ** 2) - np.trace(a ** 2))
    return recon + l1 + smooth + div


def reconstruct_Y(c: np.ndarray, e: np.ndarray) -> np.ndarray:
    """重建 Y_hat = C @ E^T，形状 (L, S)。"""
    return c @ e.T


# ──────────────────────────────────────────────────────────────────────────
# 优化 #6：MUR 迭代循环的 torch/GPU 实现。
#
# 与上面的 numpy 版本逐行对应，只是把运算对象换成 torch.Tensor，从而可以
# 跑在 CUDA 设备上。SVD 初始化（_init_ce_from_svd，S 很小，成本可忽略）
# 仍然用 numpy/CPU 完成，只有真正的迭代主循环搬到 GPU。
# ──────────────────────────────────────────────────────────────────────────


def _reflect_spectral_neighbors_torch(e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """E (S, K) 光谱维镜像填充邻居（等价于 np.pad(e, ((1,1),(0,0)), mode='reflect')）。"""
    if e.shape[0] < 2:
        return e, e
    e_pad = torch.cat([e[1:2], e, e[-2:-1]], dim=0)
    return e_pad[:-2], e_pad[2:]


def _column_normalize_e_torch(
    e: torch.Tensor, c: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """列 L2 归一化 E，并将尺度同步乘到 C 对应列（torch 版）。"""
    norms = torch.linalg.vector_norm(e, ord=2, dim=0)
    norms = torch.clamp(norms, min=eps)
    return e / norms, c * norms


def _project_simplex_rows_torch(c: torch.Tensor, eps: float) -> torch.Tensor:
    """将 C (L, K) 的每一行投影到概率单纯形（torch 版，见 numpy 版注释）。"""
    row_sums = c.sum(dim=1, keepdim=True)
    row_sums = torch.clamp(row_sums, min=eps)
    return c / row_sums


def _mul_update_torch(
    base: torch.Tensor, num: torch.Tensor, den: torch.Tensor, ratio_max: float, eps: float
) -> torch.Tensor:
    """乘性更新并裁剪比值（torch 版）。"""
    ratio = num / (den + eps)
    if ratio_max > 0:
        ratio = torch.clamp(ratio, min=1.0 / ratio_max, max=ratio_max)
    return base * ratio


def _mur_step_torch(
    c: torch.Tensor,
    e: torch.Tensor,
    y: torch.Tensor,
    lam1: float,
    lam2: float,
    lam3: float,
    eps: float,
    ratio_max: float,
    update_mode: str,
    use_simplex: bool = False,
    lam_e: float = 0.0,
    e_clamp_max: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """单步 MUR 更新（torch/GPU 版），公式与 `_mur_step`（numpy 版）完全一致。"""
    if update_mode not in ("sync", "alternating"):
        raise ValueError(f"update_mode must be sync or alternating, got {update_mode}")

    k = c.shape[1]
    ete = e.T @ e
    c_new = _mul_update_torch(c, y @ e, c @ ete + lam1 / 2.0, ratio_max, eps)
    c_new = torch.clamp(c_new, min=0.0)

    if use_simplex:
        c_new = _project_simplex_rows_torch(c_new, eps)

    c_for_e = c if update_mode == "sync" else c_new
    a = e.T @ e
    a_off_diag = a - torch.diag(torch.diagonal(a))
    e_left, e_right = _reflect_spectral_neighbors_torch(e)
    ctc = c_for_e.T @ c_for_e
    num_e = y.T @ c_for_e + 2.0 * lam2 * (e_left + e_right)
    eye_k = torch.eye(k, device=e.device, dtype=e.dtype)
    den_reg = ctc + 4.0 * lam2 * eye_k + 4.0 * lam3 * a_off_diag
    if lam_e > 0.0:
        den_reg = den_reg + lam_e * eye_k
    den_e = e @ den_reg
    e_hat = _mul_update_torch(e, num_e, den_e, ratio_max, eps)
    e_hat = torch.clamp(e_hat, min=0.0)

    if e_clamp_max > 0.0:
        e_hat = torch.clamp(e_hat, max=e_clamp_max)

    if use_simplex:
        e_new = e_hat
    else:
        e_new, c_new = _column_normalize_e_torch(e_hat, c_new, eps)

    return c_new, e_new


def _objective_torch(
    y: torch.Tensor,
    c: torch.Tensor,
    e: torch.Tensor,
    lam1: float,
    lam2: float,
    lam3: float,
    resid: torch.Tensor | None = None,
) -> float:
    """完整正则化目标（torch 版，仅用于 track_objective=True 时的诊断打印）。"""
    if resid is None:
        resid = y - c @ e.T
    recon = float(resid.pow(2).sum().item())
    l1 = lam1 * float(c.sum().item())
    if e.shape[0] > 1:
        diff = e[1:, :] - e[:-1, :]
        smooth = lam2 * float(diff.pow(2).sum().item())
    else:
        smooth = 0.0
    a = e.T @ e
    div = lam3 * float((a.pow(2).sum() - torch.diagonal(a).pow(2).sum()).item())
    return recon + l1 + smooth + div


def stats_array(x: np.ndarray) -> dict[str, float]:
    """数组数值范围统计。"""
    x = np.asarray(x, dtype=np.float64)
    return {
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "p01": float(np.percentile(x, 1)),
        "p50": float(np.percentile(x, 50)),
        "p99": float(np.percentile(x, 99)),
    }


def compute_reconstruction_metrics(
    y: np.ndarray,
    y_hat: np.ndarray,
) -> dict[str, float]:
    """比较原始 OD 立方（展平）与重建 OD 的偏差。"""
    y = y.astype(np.float64)
    y_hat = y_hat.astype(np.float64)
    resid = y - y_hat
    mse = float(np.mean(resid ** 2))
    rmse = float(np.sqrt(mse))
    y_norm = float(np.linalg.norm(y))
    rel_fro = float(np.linalg.norm(resid) / max(y_norm, 1e-12))
    y_range = float(y.max() - y.min())
    nrmse = rmse / max(y_range, 1e-12)
    return {
        "mse": mse,
        "rmse": rmse,
        "nrmse": nrmse,
        "mae": float(np.mean(np.abs(resid))),
        "max_abs": float(np.max(np.abs(resid))),
        "relative_frobenius": rel_fro,
        "r2": float(1.0 - np.sum(resid ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)),
    }


def fit_regularized_nmf(
    y: np.ndarray,
    k: int = 2,
    lam1: float = 1e-3,
    lam2: float = 1e-4,
    lam3: float = 1e-2,
    max_iter: int = 500,
    delta: float = 1e-6,
    eps: float = 1e-8,
    ratio_max: float = 10.0,
    update_mode: str = "alternating",
    seed: int | None = 42,
    verbose: bool = False,
    use_simplex: bool = False,
    lam_e: float = 0.0,
    e_clamp_max: float = 0.0,
    track_objective: bool = False,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype = torch.float32,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    对 Y (L, S) 做同步正则化 NMF。

    use_simplex=True 时，每步对 C 做 simplex 投影（每像素丰度和 =1），
    使输出 C 与模型 softmax 预测的语义一致。

    lam_e: E 的 L2 正则强度（建议 simplex 模式下设 0.01~0.1，防止坍塌端元爆炸）。
    e_clamp_max: E 逐元素上界（建议设为 od_max=3.0，物理兜底）。

    track_objective（优化 #1）：
        是否在每步记录完整正则化目标值 `_objective(...)`（含 L1/谱平滑/多样性
        项）到 `info["losses"]`。该值只用于自检/调试打印，收敛判据始终使用
        `recon`（`info["recon_trace"]`），与本参数无关。默认关闭（False），
        批处理路径不再重复执行一次与 `y_hat` 完全相同的大矩阵乘
        `c @ e.T`；需要打印完整 loss 曲线时（如 run_self_test）显式开启。

    device / dtype（优化 #6，GPU 加速）：
        实际的 MUR 迭代主循环用 torch 张量在 `device` 上执行（默认自动选择
        CUDA，不可用时回退 CPU）；SVD 初始化仍在 CPU/numpy 上完成（S 很小，
        成本可忽略）。数据只在循环开始前搬进 device 一次、循环结束后搬回
        CPU 一次，中间上千步迭代都留在显存里。`dtype` 默认 float32——消费级
        显卡的 fp64 吞吐远低于 fp32，且算法本身的 eps/ratio_max 数值稳定化
        机制在 float32 下同样适用。

    Returns:
        c: (L, K)  numpy 数组
        e: (S, K)  numpy 数组
        info: 迭代信息（loss 曲线、迭代次数等）
    """
    if y.ndim != 2:
        raise ValueError(f"Y must be 2D (L, S), got {y.shape}")
    l, s = y.shape
    if k <= 0 or k > min(l, s):
        raise ValueError(f"invalid K={k} for Y shape {y.shape}")

    torch_device = _resolve_device(device)
    torch_dtype = _resolve_dtype(dtype)

    # SVD 初始化在 CPU/numpy 上完成（S 很小，成本可忽略；用 float64 求解更稳）。
    y64 = y.astype(np.float64)
    try:
        c0, e0 = _init_ce_from_svd(y64, k, eps, use_simplex=use_simplex)
    except np.linalg.LinAlgError:
        rng = np.random.default_rng(seed)
        c0 = np.abs(rng.standard_normal((l, k)))
        e0 = np.abs(rng.standard_normal((s, k)))
        if use_simplex:
            c0 = _project_simplex_rows(c0, eps)
        else:
            e0, c0 = _column_normalize_e(e0, c0, eps)

    # 数据搬进 device 一次；后续所有迭代都在 GPU（或 CPU，取决于 device）上进行。
    y_t = torch.from_numpy(y64).to(device=torch_device, dtype=torch_dtype)
    c = torch.from_numpy(c0).to(device=torch_device, dtype=torch_dtype)
    e = torch.from_numpy(e0).to(device=torch_device, dtype=torch_dtype)

    losses: list[float] = []
    recon_trace: list[float] = []
    prev_recon: float | None = None
    n_iter = 0
    best_c, best_e = c.clone(), e.clone()
    best_recon = float("inf")
    # 优化 #1（附加）：无论走哪个分支退出循环，退出时选用的 (c, e) 对应的
    # 重建残差平方和都已经在循环体内算过一次（就是 recon 或 best_recon），
    # 因此这里全程跟踪 final_recon_value，避免最后再对最终 (c, e) 重新做一次
    # 完整的 reconstruct_Y（大矩阵乘），数值与原实现完全一致。
    final_recon_value: float | None = None

    for t in range(max_iter):
        c, e = _mur_step_torch(
            c, e, y_t, lam1, lam2, lam3, eps, ratio_max, update_mode,
            use_simplex=use_simplex, lam_e=lam_e, e_clamp_max=e_clamp_max,
        )
        y_hat = c @ e.T
        resid = y_t - y_hat
        recon = float(resid.pow(2).sum().item())
        if track_objective:
            # 优化 #1：复用上面刚算好的 resid，不再重复执行一次
            # 同样代价的 c @ e.T（数值与原实现完全一致）。
            losses.append(_objective_torch(y_t, c, e, lam1, lam2, lam3, resid=resid))
        recon_trace.append(recon)

        if recon < best_recon:
            best_recon = recon
            best_c, best_e = c.clone(), e.clone()

        if prev_recon is not None:
            rel_change = abs(prev_recon - recon) / max(prev_recon, eps)
            if verbose and (t % 50 == 0 or t == max_iter - 1):
                print(f"  iter {t:4d}  recon={recon:.6e}  rel_change={rel_change:.3e}")
            if recon > prev_recon * 1.05:
                if verbose:
                    print(f"  iter {t:4d}  recon 上升，回滚至最优解并停止")
                c, e = best_c, best_e
                final_recon_value = best_recon
                n_iter = t
                break
            if rel_change < delta:
                n_iter = t + 1
                final_recon_value = recon
                break
        prev_recon = recon
        n_iter = t + 1
    else:
        c, e = best_c, best_e
        final_recon_value = best_recon

    if final_recon_value is None or not np.isfinite(final_recon_value):
        # 极端边界情况兜底（如 max_iter<=0 导致循环体从未执行，
        # best_recon 仍是初始的 +inf），此时才退化为原始实现的显式重新计算。
        final_recon_value = float((y_t - c @ e.T).pow(2).sum().item())

    info = {
        "n_iter": n_iter,
        "final_recon": float(final_recon_value),
        "final_objective": losses[-1] if losses else 0.0,
        "losses": losses,
        "recon_trace": recon_trace,
        "update_mode": update_mode,
        "use_simplex": use_simplex,
        "lam_e": lam_e,
        "e_clamp_max": e_clamp_max,
        "device": str(torch_device),
        "dtype": str(torch_dtype),
    }
    # 数据搬回 CPU 一次；下游（reshape/落盘/统计）保持原有 numpy 流程不变。
    c_np = c.detach().cpu().numpy()
    e_np = e.detach().cpu().numpy()
    return c_np, e_np, info


def run_offline_nmf_on_cube(
    od: np.ndarray,
    k: int = 2,
    lam1: float = 1e-3,
    lam2: float = 1e-4,
    lam3: float = 1e-2,
    max_iter: int = 500,
    delta: float = 1e-6,
    eps: float = 1e-8,
    ratio_max: float = 10.0,
    update_mode: str = "alternating",
    seed: int | None = 42,
    verbose: bool = False,
    use_simplex: bool = False,
    lam_e: float = 0.0,
    e_clamp_max: float = 0.0,
    track_objective: bool = False,
    compute_stats: bool = True,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype = torch.float32,
) -> NMFResult:
    """
    对 OD 立方 (S, H, W) 运行完整离线 NMF 流水线。

    use_simplex=True 时，输出 C* 满足概率单纯形约束（每像素 ΣK = 1）。
    lam_e / e_clamp_max 见 fit_regularized_nmf。

    track_objective：见 fit_regularized_nmf（优化 #1，批处理默认关闭）。
    compute_stats（优化 #2）：是否计算 c_stats/e_stats（对 C*/E* 做
        min/max/mean/std/percentile 统计）。这些统计只在自检打印
        （run_self_test / _print_stats_table）里被使用，process_data_root
        批处理路径从不读取，却要对整幅 C*（形状 (K,H,W)，大图可达千万级
        元素）做三次 np.percentile（单线程排序）+ 一次 float64 拷贝。
        默认开启以保持与原实现行为一致；批处理路径显式传 False 跳过。
    device / dtype（优化 #6）：见 fit_regularized_nmf。
    """
    y, h, w = od_cube_to_Y(od)
    c, e, info = fit_regularized_nmf(
        y, k=k, lam1=lam1, lam2=lam2, lam3=lam3,
        max_iter=max_iter, delta=delta, eps=eps,
        ratio_max=ratio_max, update_mode=update_mode,
        seed=seed, verbose=verbose, use_simplex=use_simplex,
        lam_e=lam_e, e_clamp_max=e_clamp_max, track_objective=track_objective,
        device=device, dtype=dtype,
    )
    y_hat = reconstruct_Y(c, e)
    c_star = reshape_C_to_map(c, h, w)
    e_star = e_col_to_row(e)
    metrics = compute_reconstruction_metrics(y, y_hat)
    empty_stats: dict[str, float] = {}
    return NMFResult(
        c_star=c_star,
        e_star=e_star,
        c_flat=c,
        e_col=e,
        y_hat=y_hat,
        od_cube=od.astype(np.float32),
        metrics=metrics,
        c_stats=stats_array(c_star) if compute_stats else empty_stats,
        e_stats=stats_array(e_star) if compute_stats else empty_stats,
        n_iter=int(info["n_iter"]),
        final_loss=float(info["final_recon"]),
    )


def save_nmf_result(result: NMFResult, out_dir: str | Path, stem: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{stem}_C.npy", result.c_star.astype(np.float32))
    np.save(out_dir / f"{stem}_E.npy", result.e_star.astype(np.float32))


def _print_stats_table(title: str, stats: dict[str, float]) -> None:
    print(f"  {title}")
    for key in ("min", "p01", "p50", "p99", "max", "mean", "std"):
        print(f"    {key:<6}: {stats[key]:.6e}")


def run_self_test(
    image_path: str | None = None,
    k: int = 2,
    lam1: float = 1e-3,
    lam2: float = 1e-4,
    lam3: float = 1e-2,
    max_iter: int = 500,
    ratio_max: float = 10.0,
    update_mode: str = "alternating",
    seed: int = 42,
    synthetic: bool = False,
    use_simplex: bool = False,
    device: str | torch.device | None = None,
    dtype: str | torch.dtype = torch.float32,
) -> NMFResult:
    """加载（或合成）数据，运行 NMF，打印重建偏差与 C/E 数值范围。"""
    print(f"\n{'=' * 78}")
    print("  offline_nmf — MUR 分解自检")
    print(
        f"  K={k}, λ1={lam1}, λ2={lam2}, λ3={lam3}, max_iter={max_iter}, "
        f"update_mode={update_mode}, ratio_max={ratio_max}, seed={seed}, "
        f"simplex={'on' if use_simplex else 'off'}"
    )
    print(f"{'=' * 78}")

    if synthetic or image_path is None:
        print("\n  [数据源] 合成 OD 立方（已知 C_true, E_true）")
        rng = np.random.default_rng(seed)
        s, h, w = 40, 64, 64
        e_true = np.abs(rng.standard_normal((s, k)))
        e_true /= np.linalg.norm(e_true, axis=0, keepdims=True) + 1e-8
        c_true = np.abs(rng.standard_normal((h * w, k)))
        y = c_true @ e_true.T
        y += 0.01 * rng.standard_normal(y.shape)
        od = Y_to_od_cube(y, h, w).astype(np.float32)
        print(f"  合成 shape: OD {od.shape}")
    else:
        path = Path(image_path)
        print(f"\n  [数据源] {path}")
        intensity = load_intensity_cube(path)
        od = intensity_to_od_np(intensity).astype(np.float32)
        print(f"  intensity {intensity.shape}  range [{intensity.min():.4f}, {intensity.max():.4f}]")
        print(f"  OD        {od.shape}  range [{od.min():.4f}, {od.max():.4f}]")

    resolved_device = _resolve_device(device)
    print(f"  [设备] device={resolved_device}  dtype={_resolve_dtype(dtype)}")
    result = run_offline_nmf_on_cube(
        od, k=k, lam1=lam1, lam2=lam2, lam3=lam3,
        max_iter=max_iter, ratio_max=ratio_max, update_mode=update_mode,
        seed=seed, verbose=True, use_simplex=use_simplex,
        track_objective=True, compute_stats=True,
        device=device, dtype=dtype,
    )

    if use_simplex:
        c_sum = result.c_star.sum(axis=0)  # (H, W)
        print(
            f"\n  [simplex 检查] C_sum 统计（每像素 ΣK，应≈1）: "
            f"min={c_sum.min():.6f}  max={c_sum.max():.6f}  "
            f"mean={c_sum.mean():.6f}  std={c_sum.std():.6e}"
        )

    print(f"\n  [迭代] n_iter={result.n_iter}, final_recon_F2={result.final_loss:.6e}")

    print("\n  [重建偏差] Y vs C@E^T（OD 域，展平像素）")
    for key, val in result.metrics.items():
        print(f"    {key:<22}: {val:.6e}")

    od_hat_cube = Y_to_od_cube(result.y_hat, *result.od_cube.shape[1:])
    cube_mse = float(np.mean((result.od_cube - od_hat_cube) ** 2))
    cube_rmse = float(np.sqrt(cube_mse))
    print(f"\n  [立方体验证] (S,H,W) MSE={cube_mse:.6e}, RMSE={cube_rmse:.6e}")

    print(f"\n  [C* 数值范围] shape={result.c_star.shape}  (K, H, W)")
    _print_stats_table("全局", result.c_stats)
    for ki in range(result.c_star.shape[0]):
        _print_stats_table(f"端元 k={ki}", stats_array(result.c_star[ki]))

    print(f"\n  [E* 数值范围] shape={result.e_star.shape}  (K, S)")
    _print_stats_table("全局", result.e_stats)
    for ki in range(result.e_star.shape[0]):
        _print_stats_table(f"端元 k={ki}", stats_array(result.e_star[ki]))
        col_norm = float(np.linalg.norm(result.e_col[:, ki]))
        print(f"    col_L2_norm (内部 E[:,k]): {col_norm:.6e}")

    print(f"\n{'=' * 78}")
    print("  自检完成。")
    print(f"{'=' * 78}\n")
    return result


def check_result_health(
    stem: str,
    result: "NMFResult",
    e_warn_max: float = 50.0,
    c_sum_tol: float = 0.05,
    use_simplex: bool = False,
) -> list[str]:
    """
    检查单张图 NMF 分解结果的健康状况，返回警告字符串列表。

    检查项目：
      1. E 最大值是否异常（超过 e_warn_max）→ 端元爆炸预警
      2. E 是否存在 NaN/Inf
      3. C 是否存在 NaN/Inf
      4. simplex 模式下，C 行和是否偏离 1（|sum-1| > c_sum_tol）
      5. simplex 模式下，坍塌端元数量（整列 C[:,k].max() < 1e-3）
    """
    warns: list[str] = []
    c = result.c_flat      # (L, K)
    e = result.e_col       # (S, K)

    if not np.isfinite(e).all():
        warns.append(f"[E NaN/Inf] {stem}: E 含有 NaN 或 Inf！")
    elif e.max() > e_warn_max:
        k_max = int(np.argmax(e.max(axis=0)))
        warns.append(
            f"[E EXPLODE] {stem}: E.max={e.max():.2f} > {e_warn_max}  "
            f"(端元 k={k_max}, E[:,k].max={e[:, k_max].max():.2f})"
        )

    if not np.isfinite(c).all():
        warns.append(f"[C NaN/Inf] {stem}: C 含有 NaN 或 Inf！")

    if use_simplex:
        row_sums = c.sum(axis=1)
        bad_rows = int(np.sum(np.abs(row_sums - 1.0) > c_sum_tol))
        if bad_rows > 0:
            warns.append(
                f"[C SUM] {stem}: {bad_rows}/{c.shape[0]} 像素行和偏离 1 "
                f"(偏差>{c_sum_tol})，mean_sum={row_sums.mean():.4f}"
            )
        collapsed = [k for k in range(c.shape[1]) if c[:, k].max() < 1e-3]
        if collapsed:
            warns.append(
                f"[COLLAPSE] {stem}: 坍塌端元 k={collapsed}，"
                f"E.max={e[:, collapsed].max():.3f}"
            )

    return warns


NMF_RECONSTRUCTION_MSE_FILENAME = "nmf_reconstruction_mse.json"


def image_relpath_for_mse_index(
    data_root: Path,
    stem: str,
    *,
    project_root: Path | None = None,
) -> str:
    """Return a stable project-relative path for ``data_root/images/{stem}.npy``."""
    project_root = (project_root or _ROOT).resolve()
    data_root = Path(data_root)
    resolved_root = data_root.resolve() if data_root.is_absolute() else (Path.cwd() / data_root).resolve()
    try:
        rel_data_root = resolved_root.relative_to(project_root)
    except ValueError:
        rel_data_root = data_root
    return (rel_data_root / "images" / f"{stem}.npy").as_posix()


def save_nmf_reconstruction_mse_index(
    mse_index: dict[str, float],
    out_dir: str | Path,
    save_name: str = NMF_RECONSTRUCTION_MSE_FILENAME,
) -> Path:
    """Persist ``{image_relpath: mse}`` under the NMF cache directory."""
    if not mse_index:
        raise ValueError("mse_index is empty")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / save_name
    with save_path.open("w", encoding="utf-8") as handle:
        json.dump(mse_index, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    return save_path


def build_nmf_mse_index_from_log(
    log_path: str | Path,
    *,
    project_root: Path | None = None,
    save_name: str = NMF_RECONSTRUCTION_MSE_FILENAME,
    save_to_cache_dir: bool = True,
) -> dict[str, float]:
    """Parse an offline NMF log and build ``{image_relpath: mse}``.

    When ``save_to_cache_dir`` is True, also writes JSON into the cache directory
    parsed from the first line of the log file.
    """
    log_path = Path(log_path).expanduser().resolve()
    if not log_path.is_file():
        raise FileNotFoundError(f"log file not found: {log_path}")

    project_root = (project_root or Path.cwd()).resolve()
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"empty log file: {log_path}")

    header_pat = re.compile(r"→\s*(?P<cache_dir>\S+)")
    header_match = header_pat.search(lines[0])
    if header_match is None:
        raise ValueError(f"cannot parse cache directory from first line: {lines[0]!r}")

    cache_dir_rel = Path(header_match.group("cache_dir"))
    cache_dir = (
        cache_dir_rel
        if cache_dir_rel.is_absolute()
        else (project_root / cache_dir_rel).resolve()
    )
    data_root_rel = cache_dir_rel.parent

    item_pat = re.compile(
        r"^\s*\[\d+/\d+\]\s+(?P<stem>\S+)\s+MSE=(?P<mse>[\d.eE+-]+)"
    )
    mse_index: dict[str, float] = {}
    for line in lines[1:]:
        match = item_pat.match(line)
        if match is None:
            continue
        stem = match.group("stem")
        mse = float(match.group("mse"))
        rel_path = (data_root_rel / "images" / f"{stem}.npy").as_posix()
        mse_index[rel_path] = mse

    if not mse_index:
        raise RuntimeError(f"no MSE entries parsed from log: {log_path}")

    if save_to_cache_dir:
        save_nmf_reconstruction_mse_index(mse_index, cache_dir, save_name=save_name)
    return mse_index


def build_nmf_mse_index_from_cache(
    cache_dir: str | Path,
    *,
    project_root: Path | None = None,
    od_max: float = 3.0,
    save_name: str = NMF_RECONSTRUCTION_MSE_FILENAME,
    save_to_cache_dir: bool = True,
    included_stems: set[str] | None = None,
) -> dict[str, float]:
    """Rebuild ``{image_relpath: mse}`` from saved ``*_C.npy`` / ``*_E.npy`` caches.

    When ``included_stems`` is provided, stale cache pairs for samples outside
    the current eligible set are ignored and cannot re-enter the MSE index.
    """
    project_root = (project_root or _ROOT).resolve()
    cache_dir = Path(cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = (project_root / cache_dir).resolve()
    else:
        cache_dir = cache_dir.resolve()
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"NMF cache directory not found: {cache_dir}")

    data_root = cache_dir.parent
    images_dir = data_root / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"missing images directory: {images_dir}")

    c_paths = sorted(cache_dir.glob("*_C.npy"))
    entries: list[tuple[Path, str, Path, Path]] = []
    missing: list[str] = []
    ignored_by_filter: list[str] = []
    for c_path in c_paths:
        stem = c_path.name[: -len("_C.npy")]
        if included_stems is not None and stem not in included_stems:
            ignored_by_filter.append(stem)
            continue
        e_path = cache_dir / f"{stem}_E.npy"
        image_path = images_dir / f"{stem}.npy"
        if not e_path.is_file():
            missing.append(f"{stem}: missing E cache")
            continue
        if not image_path.is_file():
            missing.append(f"{stem}: missing image")
            continue
        entries.append((c_path, stem, e_path, image_path))

    n_total = len(entries)
    try:
        cache_label = cache_dir.relative_to(project_root).as_posix()
    except ValueError:
        cache_label = cache_dir.as_posix()
    print(f"  开始构建 MSE 索引：{n_total} 项 → {cache_label}/{save_name}")
    if missing:
        print(f"  [warn] 跳过 {len(missing)} 个不完整缓存项")
    if ignored_by_filter:
        print(
            f"  [filter] 忽略 {len(ignored_by_filter)} 个不在当前有效样本集中的旧缓存"
        )

    mse_index: dict[str, float] = {}
    for i, (c_path, stem, e_path, image_path) in enumerate(entries, start=1):
        c_star = np.load(c_path).astype(np.float32)
        e_star = np.load(e_path).astype(np.float32)
        if c_star.ndim != 3 or e_star.ndim != 2:
            raise ValueError(f"unexpected cache shapes for {stem}: C={c_star.shape}, E={e_star.shape}")
        if c_star.shape[0] != e_star.shape[0]:
            raise ValueError(f"K mismatch for {stem}: C={c_star.shape}, E={e_star.shape}")

        intensity = load_intensity_cube(image_path)
        od = intensity_to_od_np(intensity).astype(np.float32)
        if od_max is not None:
            od = np.clip(od, 0.0, od_max)
        y, h, w = od_cube_to_Y(od)

        k, ch, cw = c_star.shape
        if (ch, cw) != (h, w):
            raise ValueError(
                f"spatial shape mismatch for {stem}: C={c_star.shape}, OD={(od.shape[1], od.shape[2])}"
            )

        c_flat = c_star.transpose(1, 2, 0).reshape(h * w, k)
        e_col = e_star.T
        y_hat = reconstruct_Y(c_flat, e_col)
        mse = float(compute_reconstruction_metrics(y, y_hat)["mse"])
        rel_path = image_relpath_for_mse_index(data_root, stem, project_root=project_root)
        mse_index[rel_path] = mse
        print(f"  [{i}/{n_total}] {stem}  MSE={mse:.6f}")

    if not mse_index:
        detail = "; ".join(missing[:5]) if missing else "no *_C.npy files"
        raise RuntimeError(f"no valid NMF cache pairs found under {cache_dir}: {detail}")

    if save_to_cache_dir:
        save_path = save_nmf_reconstruction_mse_index(mse_index, cache_dir, save_name=save_name)
        try:
            display_path = save_path.relative_to(project_root).as_posix()
        except ValueError:
            display_path = save_path.as_posix()
        print(
            f"  rebuilt MSE index: {len(mse_index)} items -> "
            f"{display_path}"
        )
    return mse_index


def _init_worker_blas_threads(blas_threads: int | None) -> None:
    """
    ProcessPoolExecutor 子进程初始化器（优化 #5）。

    在子进程第一次触发 NumPy/BLAS 计算之前，把该进程内的 BLAS 线程数
    限制到较小的值，避免"多进程同时处理多张图"与"单进程内 BLAS 自己
    尝试用满全部核心做瘦矩阵乘"两种并行策略互相打架、过度订阅 CPU。
    ProcessPoolExecutor 需要用 spawn 启动方式（全新解释器）配合，才能
    保证这里设置的环境变量在 BLAS 线程池真正初始化之前生效。
    """
    if not blas_threads or blas_threads <= 0:
        return
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(blas_threads)
    try:
        from threadpoolctl import threadpool_limits  # 部分 BLAS 后端需要运行期接口兜底

        threadpool_limits(limits=blas_threads)
    except Exception:
        pass


def _process_one_image(task: dict[str, Any]) -> dict[str, Any]:
    """
    单张图的完整离线 NMF 处理：读图→OD 转换→NMF 分解→落盘→健康检查。

    串行主循环与优化 #5（进程级并行）共用这一份逻辑，保证两条路径下
    每张图的计算过程完全一致；固定使用 track_objective=False,
    compute_stats=False（优化 #1 / #2，批处理路径不需要这两项诊断量）。
    """
    path = Path(task["path"])
    stem = path.stem
    intensity = load_intensity_cube(path)
    od = intensity_to_od_np(intensity).astype(np.float32)
    if task["clamp_od_max"] is not None:
        od = np.clip(od, 0.0, task["clamp_od_max"])

    def _run() -> NMFResult:
        return run_offline_nmf_on_cube(
            od,
            k=task["k"], lam1=task["lam1"], lam2=task["lam2"], lam3=task["lam3"],
            max_iter=task["max_iter"], verbose=task["verbose"], use_simplex=task["use_simplex"],
            lam_e=task["lam_e"], e_clamp_max=task["e_clamp_max"],
            track_objective=False, compute_stats=False,
            device=task.get("device"), dtype=task.get("dtype", torch.float32),
        )

    try:
        result = _run()
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        # 批处理时间跨度长，即使启动时选了空闲显存最多的卡，也可能被其他用户
        # 后续启动的任务挤占；先清一次 CUDA 缓存重试一次，仍失败再给出明确
        # 提示（而不是让用户去猜一段裸的 CUDA RuntimeError）。
        print(
            f"  ⚠  [CUDA OOM] {stem}: 显存不足，尝试 torch.cuda.empty_cache() 后重试一次…"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            result = _run()
        except RuntimeError as exc2:
            if "out of memory" not in str(exc2).lower():
                raise
            raise RuntimeError(
                f"[CUDA OOM] {stem}: 重试后仍然显存不足（device="
                f"{task.get('device')}）。当前机器为共享多卡环境，建议：\n"
                f"  1) 用 `nvidia-smi` 查看各卡剩余显存，通过 GPU_ID=<空闲卡编号> "
                f"或 DEVICE=cuda:<空闲卡编号> 显式指定；\n"
                f"  2) 或适当调小 --k / --max-iter，减少单张图的显存占用。"
            ) from exc2
    save_nmf_result(result, task["out_dir"], stem)

    c_flat = result.c_flat
    e_col = result.e_col
    c_sum_min = c_sum_max = None
    if task["use_simplex"]:
        row_sums = c_flat.sum(axis=1)
        c_sum_min, c_sum_max = float(row_sums.min()), float(row_sums.max())

    warns = check_result_health(
        stem, result, e_warn_max=task["e_warn_max"], use_simplex=task["use_simplex"]
    )
    return {
        "stem": stem,
        "mse": float(result.metrics["mse"]),
        "n_iter": int(result.n_iter),
        "e_mean": float(e_col.mean()),
        "e_max": float(e_col.max()),
        "c_mean": float(c_flat.mean()),
        "c_std": float(c_flat.std()),
        "c_sum_min": c_sum_min,
        "c_sum_max": c_sum_max,
        "warns": warns,
    }


def _format_progress_line(n_done: int, n_total: int, r: dict[str, Any]) -> str:
    c_sum_str = ""
    if r.get("c_sum_min") is not None:
        c_sum_str = f"  C_sum∈[{r['c_sum_min']:.3f},{r['c_sum_max']:.3f}]"
    return (
        f"  [{n_done}/{n_total}] {r['stem']}"
        f"  MSE={r['mse']:.6f}"
        f"  n_iter={r['n_iter']}"
        f"  E[mean={r['e_mean']:.3f} max={r['e_max']:.3f}]"
        f"  C[mean={r['c_mean']:.4f} std={r['c_std']:.4f}]"
        + c_sum_str
    )


def process_data_root(
    data_root: str | Path,
    out_dir: str | Path | None = None,
    k: int = 2,
    lam1: float = 1e-3,
    lam2: float = 1e-4,
    lam3: float = 1e-2,
    max_iter: int = 500,
    clamp_od_max: float | None = 3.0,
    verbose: bool = False,
    use_simplex: bool = False,
    lam_e: float = 0.0,
    e_clamp_max: float = 0.0,
    e_warn_max: float = 50.0,
    exclude_samples_file: str | Path | None = None,
    class_name: str | None = None,
    dry_run: bool = False,
    parallel_workers: int = 1,
    blas_threads_per_worker: int | None = None,
    device: str | None = None,
    dtype: str = "float32",
) -> Path:
    """
    批量处理 data_root/images/*.npy，写入 NMF 缓存。

    use_simplex=True 时缓存目录名自动附加 `_simplex` 后缀。
    lam_e / e_clamp_max / e_warn_max 见 check_result_health 与 _mur_step。
    exclude_samples_file 按精确 (class_name, stem) 在任何缓存检查和 NMF
    计算前排除样本；class_name 默认取 data_root 目录名。

    parallel_workers（对应 CPU 版本的优化 #5，默认 1 = 原始串行行为）：
        大于 1 时，对 todo_files 里互相独立的图像用进程池并行处理，
        每张图内部的 MUR 计算过程与串行路径完全一致，只是把"图间独立"
        这一天然并行性交给多进程去利用，不改变任何数值结果。

        device 解析为 CUDA 设备时分两种情况：
          - device 是自动挑选（未显式指定具体索引，见 device 参数说明）
            **且** 当前进程可见的 CUDA 设备数 > 1（例如外层先
            `export CUDA_VISIBLE_DEVICES=6,7` 限定了几张卡再运行本脚本）：
            这时会保留 parallel_workers（不强制降为 1），把 todo_files
            按提交顺序轮询（round-robin）分配到 cuda:0 ~ cuda:{n-1}
            这几块**可见范围内**的卡上并行处理——真正的多卡并行，但绝不会
            超出 CUDA_VISIBLE_DEVICES 限定的范围去动机器上的其他卡。
          - device 是显式指定的单块卡（如 "cuda:3"），或当前只有 1 块
            可见 CUDA 设备：会自动把 parallel_workers 强制降为 1 并打印
            警告——单张卡的算力已经在单图内部的矩阵乘里被吃满，多个进程
            抢同一张卡通常没有收益，还会引入重复的 CUDA 上下文初始化开销。
    blas_threads_per_worker：
        parallel_workers > 1 时，建议同时限制每个 worker 进程的 BLAS
        线程数（如 4~8），避免单张图内部的瘦矩阵乘仍然尝试用满所有核
        心，与多进程并行互相抢占；为 None/<=0 时不做限制。仅影响 CPU 侧
        （读图/OD 转换等）线程数，与 GPU 迭代本身无关。
    device / dtype（优化 #6）：
        device=None（或字面 "cuda"）时自动选择：CUDA 可用时，只会在当前
        进程可见的 CUDA 设备（受 CUDA_VISIBLE_DEVICES 限制，未设置时才是
        机器上的全部卡）里挑选；不可用则回退 CPU。也就是说：若想把本次
        所有分解任务限定在某一张或某几张卡上、完全不去动机器上的其他卡，
        在启动前设置好 CUDA_VISIBLE_DEVICES（脚本里对应 GPU_ID，可传逗号
        分隔的多个卡号，如 "6,7"）即可，本函数内部的自动选卡/多卡并行都
        只会在这个限定范围内进行。若传 "cuda:N" 这种带具体索引的值，则
        精确使用该索引（索引以当前可见范围重新编号，从 0 开始），不做
        自动选择也不参与多卡轮询。dtype 见 fit_regularized_nmf。
    """
    device_is_auto = device is None or device == "" or device == "cuda"
    n_visible_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    multi_gpu_round_robin = bool(
        device_is_auto and n_visible_cuda > 1 and parallel_workers and parallel_workers > 1
    )

    if multi_gpu_round_robin:
        torch_device = torch.device("cuda:0")  # 仅作打印占位；每张图实际分配见下方 tasks 构造
        print(
            f"  [多卡并行] 当前进程可见 {n_visible_cuda} 块 CUDA 设备"
            f"（受 CUDA_VISIBLE_DEVICES 限定范围）；parallel_workers={parallel_workers}，"
            f"图像将按提交顺序轮询分配到 cuda:0 ~ cuda:{n_visible_cuda - 1}"
        )
    else:
        torch_device = _resolve_device(device)
        if torch_device.type == "cuda" and parallel_workers and parallel_workers > 1:
            reason = (
                "当前进程可见范围内只有 1 块 CUDA 设备"
                if n_visible_cuda <= 1
                else f"显式指定了固定设备 {torch_device}（未启用多卡轮询）"
            )
            print(
                f"  [警告] {reason}，忽略 parallel_workers={parallel_workers} 并强制设为 1"
                "（多个进程抢同一张卡通常没有收益，见函数文档说明）。"
            )
            parallel_workers = 1
    torch_dtype = _resolve_dtype(dtype)
    data_root = Path(data_root)
    images_dir = data_root / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"缺少 images 目录: {images_dir}")
    out_dir = (
        Path(out_dir) if out_dir
        else data_root / cache_dir_name(
            k, lam1, lam2, lam3,
            simplex=use_simplex, lam_e=lam_e, e_clamp_max=e_clamp_max,
        )
    )
    all_image_files = sorted(images_dir.glob("*.npy"))
    if not all_image_files:
        raise RuntimeError(f"无 .npy 文件: {images_dir}")

    effective_class_name = class_name or data_root.name
    excluded_stems: set[str] = set()
    if exclude_samples_file:
        excluded_identities = load_excluded_samples(exclude_samples_file)
        excluded_stems = {
            stem
            for record_class_name, stem in excluded_identities
            if record_class_name == effective_class_name
        }
        image_stems = {path.stem for path in all_image_files}
        unmatched = excluded_stems - image_stems
        if unmatched:
            preview = ", ".join(sorted(unmatched)[:20])
            raise ValueError(
                f"排除 JSON 中属于类别 {effective_class_name!r} 的 "
                f"{len(unmatched)} 条记录无法在 {images_dir} 中匹配；"
                f"前若干项: {preview}"
            )

    all_files = [path for path in all_image_files if path.stem not in excluded_stems]
    if not all_files:
        raise RuntimeError(
            f"类别 {effective_class_name!r} 在人工排除后没有剩余图像: {images_dir}"
        )
    included_stems = {path.stem for path in all_files}

    print(
        f"样本过滤：类别={effective_class_name}  原始={len(all_image_files)}  "
        f"人工排除={len(excluded_stems)}  有效={len(all_files)}"
    )
    if exclude_samples_file:
        print(f"  排除清单：{exclude_samples_file}")
    if dry_run:
        print("  [dry-run] 过滤校验通过，不创建缓存、不执行 NMF。")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 断点续跑：检查已有有效缓存 ─────────────────────────────────────────────
    # 有效缓存 = {stem}_C.npy 与 {stem}_E.npy 均存在且文件大小 > 0
    skipped: list[str] = []
    todo_files: list[Path] = []
    for path in all_files:
        stem = path.stem
        c_file = out_dir / f"{stem}_C.npy"
        e_file = out_dir / f"{stem}_E.npy"
        if c_file.is_file() and c_file.stat().st_size > 0 and \
           e_file.is_file() and e_file.stat().st_size > 0:
            skipped.append(stem)
        else:
            todo_files.append(path)

    flag_str = f"simplex={'on' if use_simplex else 'off'}  lam_e={lam_e}  e_clamp_max={e_clamp_max}"
    print(f"扫描 {len(all_files)} 张图 → {out_dir}  [{flag_str}]")
    if skipped:
        print(f"  ↩  已跳过 {len(skipped)} 张（缓存完整）：{skipped[0]} … {skipped[-1]}")
    if not todo_files:
        print("  ✓ 所有图均已分解，无需重新运行。")
        mse_index = build_nmf_mse_index_from_cache(
            out_dir,
            od_max=clamp_od_max,
            save_to_cache_dir=True,
            included_stems=included_stems,
        )
        print(f"  MSE 索引已保存：{out_dir / NMF_RECONSTRUCTION_MSE_FILENAME} ({len(mse_index)} 项)")
        return out_dir

    print(f"  → 待处理 {len(todo_files)} 张图")
    if multi_gpu_round_robin:
        print(f"  [设备] 多卡轮询 cuda:0 ~ cuda:{n_visible_cuda - 1}  dtype={torch_dtype}")
    else:
        print(f"  [设备] device={torch_device}  dtype={torch_dtype}")
    use_parallel = parallel_workers and parallel_workers > 1 and len(todo_files) > 1
    if use_parallel:
        print(
            f"  [并行] parallel_workers={parallel_workers}  "
            f"blas_threads_per_worker={blas_threads_per_worker or '<不限制>'}"
        )

    total_warns: list[str] = []
    n_total = len(all_files)
    n_done = len(skipped)

    tasks = [
        {
            "path": str(path),
            "out_dir": str(out_dir),
            "k": k, "lam1": lam1, "lam2": lam2, "lam3": lam3,
            "max_iter": max_iter, "verbose": verbose, "use_simplex": use_simplex,
            "lam_e": lam_e, "e_clamp_max": e_clamp_max, "e_warn_max": e_warn_max,
            "clamp_od_max": clamp_od_max,
            # 多卡轮询模式下，每张图按提交顺序分配到 cuda:{i % n_visible_cuda}；
            # 否则所有图共用同一个（可能是自动选出的最空闲那块）固定设备。
            "device": (
                torch.device(f"cuda:{i % n_visible_cuda}")
                if multi_gpu_round_robin
                else torch_device
            ),
            "dtype": torch_dtype,
        }
        for i, path in enumerate(todo_files)
    ]

    if use_parallel:
        # 优化 #5：图与图之间彼此独立，用进程池并行处理，
        # 每张图内部计算逻辑（_process_one_image）与串行路径完全相同。
        import multiprocessing as _mp

        ctx = _mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=parallel_workers,
            mp_context=ctx,
            initializer=_init_worker_blas_threads,
            initargs=(blas_threads_per_worker,),
        ) as executor:
            futures = [executor.submit(_process_one_image, task) for task in tasks]
            for future in as_completed(futures):
                r = future.result()
                n_done += 1
                print(_format_progress_line(n_done, n_total, r))
                for w in r["warns"]:
                    print(f"  ⚠  {w}")
                    total_warns.append(w)
    else:
        for task in tasks:
            r = _process_one_image(task)
            n_done += 1
            print(_format_progress_line(n_done, n_total, r))
            for w in r["warns"]:
                print(f"  ⚠  {w}")
                total_warns.append(w)

    mse_index = build_nmf_mse_index_from_cache(
        out_dir,
        od_max=clamp_od_max,
        save_to_cache_dir=True,
        included_stems=included_stems,
    )
    mse_index_path = out_dir / NMF_RECONSTRUCTION_MSE_FILENAME

    # --- 全局汇总 ---
    print(f"\n{'=' * 70}")
    print(
        f"批量 NMF 完毕：处理 {len(todo_files)} 张（跳过 {len(skipped)} 张），"
        f"{len(total_warns)} 条警告"
    )
    print(f"  MSE 索引已保存：{mse_index_path} ({len(mse_index)} 项)")
    if total_warns:
        print("  异常摘要：")
        for w in total_warns:
            print(f"    {w}")
    print(f"{'=' * 70}\n")
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description="离线正则化 NMF（Step 0）")
    p.add_argument("--data-root", type=str, default=None, help="含 images/ 的数据根目录（批量模式）")
    p.add_argument("--image", type=str, default=None, help="单张 .npy 路径（H,W,S 或 S,H,W）")
    p.add_argument("--synthetic", action="store_true", help="使用合成数据测试")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--l1", type=float, default=1e-3)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--l3", type=float, default=1e-2)
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--update-mode", choices=("sync", "alternating"), default="alternating")
    p.add_argument("--ratio-max", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default=None, help="若指定则保存 C*/E*")
    p.add_argument(
        "--exclude-json",
        type=str,
        default=None,
        help=(
            "批量模式可选：顶层为 record list 的分类样本排除 JSON；"
            "每条至少包含 class_name 和 stem"
        ),
    )
    p.add_argument(
        "--class-name",
        type=str,
        default=None,
        help=(
            "批量模式可选：当前 data-root 对应的分类类别名；"
            "指定排除 JSON 时默认取 data-root 的目录名"
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="批量模式下只校验并统计人工排除，不创建缓存或执行 NMF",
    )
    p.add_argument(
        "--use-simplex", action="store_true",
        help="对 C 施加 simplex 约束（每像素 ΣK=1），输出目录自动附加 _simplex 后缀",
    )
    p.add_argument(
        "--lam-e", type=float, default=0.05,
        help="E 的 L2 正则强度（simplex 模式防坍塌端元爆炸，建议 0.01~0.1，默认 0.05）",
    )
    p.add_argument(
        "--e-clamp-max", type=float, default=3.0,
        help="E 逐元素上界（物理兜底，默认与 od_max 一致=3.0；<=0 表示不启用）",
    )
    p.add_argument(
        "--e-warn-max", type=float, default=50.0,
        help="E.max 超过此值时打印异常警告（默认 50.0）",
    )
    p.add_argument(
        "--parallel-workers", type=int, default=1,
        help=(
            "批量模式下用进程池并行处理的图数（默认 1=原始串行行为）。"
            "device 为 CPU 时对应优化 #5；device 自动选卡且当前可见 CUDA 设备"
            "数 > 1 时（如先设了 CUDA_VISIBLE_DEVICES=6,7 限定几张卡），"
            "会启用多卡轮询并行，把图像分配到这几块可见的卡上；device 是固定"
            "单卡或只有 1 块可见 CUDA 设备时会被自动强制降为 1。"
        ),
    )
    p.add_argument(
        "--blas-threads-per-worker", type=int, default=0,
        help=(
            "配合 --parallel-workers>1 使用：限制每个 worker 进程的 BLAS 线程数"
            "（建议 4~8），避免单张图内部矩阵乘仍占满所有核心与多进程并行打架。"
            "<=0 表示不限制。"
        ),
    )
    p.add_argument(
        "--device", type=str, default=None,
        help=(
            "优化 #6：MUR 迭代运行的 torch 设备，如 cuda:0 / cpu。"
            "默认（或传字面 cuda）自动选择：CUDA 可用时只在当前进程可见的 CUDA "
            "设备（受 CUDA_VISIBLE_DEVICES 限制）里挑剩余显存最多的一块，"
            "不会碰机器上其他不在可见范围内的卡；不可用则回退 CPU。"
            "若要把所有分解任务限定在某一张或几张卡上，请在启动前设置 "
            "CUDA_VISIBLE_DEVICES（对应 run_offline_nmf_cuda.sh 里的 GPU_ID，"
            "可传逗号分隔的多个卡号，如 6,7）。"
        ),
    )
    p.add_argument(
        "--dtype", type=str, default="float32", choices=("float32", "float64"),
        help="优化 #6：GPU 迭代使用的浮点精度，默认 float32（消费级显卡 fp64 吞吐低）。",
    )
    args = p.parse_args()

    if args.data_root:
        process_data_root(
            args.data_root,
            out_dir=args.out_dir,
            k=args.k,
            lam1=args.l1,
            lam2=args.l2,
            lam3=args.l3,
            max_iter=args.max_iter,
            verbose=True,
            use_simplex=args.use_simplex,
            lam_e=args.lam_e,
            e_clamp_max=args.e_clamp_max if args.e_clamp_max > 0 else 0.0,
            e_warn_max=args.e_warn_max,
            exclude_samples_file=args.exclude_json,
            class_name=args.class_name,
            dry_run=args.dry_run,
            parallel_workers=args.parallel_workers,
            blas_threads_per_worker=(
                args.blas_threads_per_worker if args.blas_threads_per_worker > 0 else None
            ),
            device=args.device,
            dtype=args.dtype,
        )
        return

    if args.exclude_json or args.class_name or args.dry_run:
        p.error("--exclude-json/--class-name/--dry-run 仅可用于 --data-root 批量模式")

    result = run_self_test(
        image_path=args.image,
        k=args.k,
        lam1=args.l1,
        lam2=args.l2,
        lam3=args.l3,
        max_iter=args.max_iter,
        ratio_max=args.ratio_max,
        update_mode=args.update_mode,
        seed=args.seed,
        synthetic=args.synthetic or args.image is None,
        use_simplex=args.use_simplex,
        device=args.device,
        dtype=args.dtype,
    )

    if args.out_dir is not None:
        stem = Path(args.image).stem if args.image else "synthetic"
        save_nmf_result(result, args.out_dir, stem)
        print(f"  已保存至 {args.out_dir}/{stem}_C.npy, {stem}_E.npy")


if __name__ == "__main__":
    main()
