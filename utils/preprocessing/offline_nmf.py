"""
Step 0：离线正则化 NMF（同步 MUR）。

对单张 OD 立方 Y ∈ R^{S×H×W} 做无掩膜分解，输出：
  C* ∈ R^{K×H×W}  （行端元惯例下的像素丰度）
  E* ∈ R^{K×S}      （行端元惯例下的端元矩阵）

内部优化使用列端元惯例：Y ≈ C @ E^T，C ∈ R^{L×K}，E ∈ R^{S×K}。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.physics.beer_lambert import intensity_to_od_np
from utils.sample_exclusion import load_excluded_samples


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
) -> float:
    """完整正则化目标（Frobenius 重建 + L1 + 谱平滑 + 多样性）。"""
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
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    对 Y (L, S) 做同步正则化 NMF。

    use_simplex=True 时，每步对 C 做 simplex 投影（每像素丰度和 =1），
    使输出 C 与模型 softmax 预测的语义一致。

    lam_e: E 的 L2 正则强度（建议 simplex 模式下设 0.01~0.1，防止坍塌端元爆炸）。
    e_clamp_max: E 逐元素上界（建议设为 od_max=3.0，物理兜底）。

    Returns:
        c: (L, K)
        e: (S, K)
        info: 迭代信息（loss 曲线、迭代次数等）
    """
    if y.ndim != 2:
        raise ValueError(f"Y must be 2D (L, S), got {y.shape}")
    l, s = y.shape
    if k <= 0 or k > min(l, s):
        raise ValueError(f"invalid K={k} for Y shape {y.shape}")

    y = y.astype(np.float64)
    try:
        c, e = _init_ce_from_svd(y, k, eps, use_simplex=use_simplex)
    except np.linalg.LinAlgError:
        rng = np.random.default_rng(seed)
        c = np.abs(rng.standard_normal((l, k)))
        e = np.abs(rng.standard_normal((s, k)))
        if use_simplex:
            c = _project_simplex_rows(c, eps)
        else:
            e, c = _column_normalize_e(e, c, eps)

    losses: list[float] = []
    recon_trace: list[float] = []
    prev_recon: float | None = None
    n_iter = 0
    best_c, best_e = c.copy(), e.copy()
    best_recon = float("inf")

    for t in range(max_iter):
        c, e = _mur_step(
            c, e, y, lam1, lam2, lam3, eps, ratio_max, update_mode,
            use_simplex=use_simplex, lam_e=lam_e, e_clamp_max=e_clamp_max,
        )
        y_hat = reconstruct_Y(c, e)
        recon = float(np.sum((y - y_hat) ** 2))
        losses.append(_objective(y, c, e, lam1, lam2, lam3))
        recon_trace.append(recon)

        if recon < best_recon:
            best_recon = recon
            best_c, best_e = c.copy(), e.copy()

        if prev_recon is not None:
            rel_change = abs(prev_recon - recon) / max(prev_recon, eps)
            if verbose and (t % 50 == 0 or t == max_iter - 1):
                print(f"  iter {t:4d}  recon={recon:.6e}  rel_change={rel_change:.3e}")
            if recon > prev_recon * 1.05:
                if verbose:
                    print(f"  iter {t:4d}  recon 上升，回滚至最优解并停止")
                c, e = best_c, best_e
                n_iter = t
                break
            if rel_change < delta:
                n_iter = t + 1
                break
        prev_recon = recon
        n_iter = t + 1
    else:
        c, e = best_c, best_e

    info = {
        "n_iter": n_iter,
        "final_recon": float(np.sum((y - reconstruct_Y(c, e)) ** 2)),
        "final_objective": losses[-1] if losses else 0.0,
        "losses": losses,
        "recon_trace": recon_trace,
        "update_mode": update_mode,
        "use_simplex": use_simplex,
        "lam_e": lam_e,
        "e_clamp_max": e_clamp_max,
    }
    return c, e, info


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
) -> NMFResult:
    """
    对 OD 立方 (S, H, W) 运行完整离线 NMF 流水线。

    use_simplex=True 时，输出 C* 满足概率单纯形约束（每像素 ΣK = 1）。
    lam_e / e_clamp_max 见 fit_regularized_nmf。
    """
    y, h, w = od_cube_to_Y(od)
    c, e, info = fit_regularized_nmf(
        y, k=k, lam1=lam1, lam2=lam2, lam3=lam3,
        max_iter=max_iter, delta=delta, eps=eps,
        ratio_max=ratio_max, update_mode=update_mode,
        seed=seed, verbose=verbose, use_simplex=use_simplex,
        lam_e=lam_e, e_clamp_max=e_clamp_max,
    )
    y_hat = reconstruct_Y(c, e)
    c_star = reshape_C_to_map(c, h, w)
    e_star = e_col_to_row(e)
    metrics = compute_reconstruction_metrics(y, y_hat)
    return NMFResult(
        c_star=c_star,
        e_star=e_star,
        c_flat=c,
        e_col=e,
        y_hat=y_hat,
        od_cube=od.astype(np.float32),
        metrics=metrics,
        c_stats=stats_array(c_star),
        e_stats=stats_array(e_star),
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

    result = run_offline_nmf_on_cube(
        od, k=k, lam1=lam1, lam2=lam2, lam3=lam3,
        max_iter=max_iter, ratio_max=ratio_max, update_mode=update_mode,
        seed=seed, verbose=True, use_simplex=use_simplex,
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
) -> Path:
    """
    批量处理 data_root/images/*.npy，写入 NMF 缓存。

    use_simplex=True 时缓存目录名自动附加 `_simplex` 后缀。
    lam_e / e_clamp_max / e_warn_max 见 check_result_health 与 _mur_step。
    exclude_samples_file 按精确 (class_name, stem) 在任何缓存检查和 NMF
    计算前排除样本；class_name 默认取 data_root 目录名。
    """
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

    total_warns: list[str] = []
    n_total = len(all_files)
    n_done = len(skipped)
    for path in todo_files:
        n_done += 1
        stem = path.stem
        intensity = load_intensity_cube(path)
        od = intensity_to_od_np(intensity).astype(np.float32)
        if clamp_od_max is not None:
            od = np.clip(od, 0.0, clamp_od_max)
        result = run_offline_nmf_on_cube(
            od, k=k, lam1=lam1, lam2=lam2, lam3=lam3,
            max_iter=max_iter, verbose=verbose, use_simplex=use_simplex,
            lam_e=lam_e, e_clamp_max=e_clamp_max,
        )
        save_nmf_result(result, out_dir, stem)

        # --- 打印进度行 ---
        c_flat = result.c_flat
        e_col = result.e_col
        c_sum_str = ""
        if use_simplex:
            row_sums = c_flat.sum(axis=1)
            c_sum_str = f"  C_sum∈[{row_sums.min():.3f},{row_sums.max():.3f}]"
        progress_line = (
            f"  [{n_done}/{n_total}] {stem}"
            f"  MSE={result.metrics['mse']:.6f}"
            f"  n_iter={result.n_iter}"
            f"  E[mean={e_col.mean():.3f} max={e_col.max():.3f}]"
            f"  C[mean={c_flat.mean():.4f} std={c_flat.std():.4f}]"
            + c_sum_str
        )
        print(progress_line)

        # --- 每张图健康检查 ---
        warns = check_result_health(
            stem, result, e_warn_max=e_warn_max, use_simplex=use_simplex
        )
        for w in warns:
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
    )

    if args.out_dir is not None:
        stem = Path(args.image).stem if args.image else "synthetic"
        save_nmf_result(result, args.out_dir, stem)
        print(f"  已保存至 {args.out_dir}/{stem}_C.npy, {stem}_E.npy")


if __name__ == "__main__":
    main()
