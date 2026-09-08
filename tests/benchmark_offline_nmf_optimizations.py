#!/usr/bin/env python3
"""
离线 NMF 性能优化对比脚本。

对同一批合成高光谱数据，分别用三种方式跑一遍
``utils/preprocessing`` 下的批量离线 NMF 分解，并对比：
  1. **耗时**（墙钟时间）
  2. **最终重建 MSE**（每张图 ``nmf_reconstruction_mse.json`` 里的值）

四种场景：
  - ``baseline``           —— ``utils/preprocessing/offline_nmf.py``
                               （原始未优化实现）
  - ``optimized_serial``   —— ``utils/preprocessing/offline_nmf_optimized.py``，
                               落实 #1（消除 `_objective`/`final_recon` 里
                               对 `c @ e.T` 的重复大矩阵乘）与 #2（批处理路径
                               跳过只在自检模式下使用的 `stats_array` 全量
                               percentile 统计）两项优化，单进程串行运行
                               （``parallel_workers=1``，与 baseline 处理顺序一致）
  - ``optimized_parallel`` —— 同一份优化实现，额外开启 #5（进程级并行，
                               ``parallel_workers>1``），把"图与图互相独立"
                               这一点交给多进程去利用多核 CPU
  - ``gpu``                —— ``utils/preprocessing/offline_nmf_cuda.py``，
                               在 #1+#2 基础上把 MUR 迭代主循环换成 torch 张量
                               运算，跑在 --device 指定的 GPU 上（#6）；无可用
                               CUDA 设备或显式传 --skip-gpu 时自动跳过该场景

#1/#2 不改变任何一步 MUR 更新的数值，所以 baseline 与 optimized_serial 的
逐图 MSE 应完全一致（或仅有浮点求和顺序带来的极小误差）；optimized_parallel
只是把同样的单图计算调度到不同进程/不同 BLAS 线程数下执行，MSE 也应与前两者
一致，只有耗时会随并行度下降。gpu 场景默认用 float32 在 GPU 上迭代，数值上
不再逐位相同，但重建 MSE 应与 CPU 场景在合理容差内一致（详见报告下方说明）。

用法示例：
    # 默认：4 张 400x500x20 合成图，max_iter=60，跑起来大约 1~3 分钟
    python tests/benchmark_offline_nmf_optimizations.py

    # 复现真实"大空间尺寸"场景（更接近 1300x1800 的检测数据，会明显更慢）：
    python tests/benchmark_offline_nmf_optimizations.py \\
        --num-images 8 --height 1300 --width 1800 --bands 20 \\
        --max-iter 1000 --parallel-workers 8 --blas-threads-per-worker 8 \\
        --device cuda:7
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def build_synthetic_dataset(
    images_dir: Path,
    num_images: int,
    height: int,
    width: int,
    bands: int,
    seed: int,
) -> None:
    """
    生成 ``num_images`` 张磁盘约定 (H, W, S) 的合成透射率立方体。

    每张图由少量真实端元（k_true=4）线性混合 + 少量高斯噪声合成，
    使重建问题落在有意义的范围内（不是纯随机噪声，也不是零残差的退化解）。
    数值范围模拟项目里 [0, 1] 归一化透射率的约定。
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    k_true = 4
    l = height * width
    for i in range(num_images):
        c_true = rng.dirichlet(np.ones(k_true), size=l).astype(np.float32)  # (L, k_true)，行和=1
        e_true = rng.uniform(0.1, 2.0, size=(bands, k_true)).astype(np.float32)
        od = (c_true @ e_true.T).reshape(height, width, bands)
        od = od + rng.normal(0.0, 0.02, size=od.shape).astype(np.float32)
        od = np.clip(od, 0.0, 3.0)
        intensity = np.exp(-od).astype(np.float32)  # OD -> 透射率，符合 load_intensity_cube 的磁盘约定
        np.save(images_dir / f"synthetic{i:03d}.npy", intensity)


@dataclass
class ScenarioResult:
    name: str
    wall_time_s: float
    mse_index: dict[str, float]
    is_low_precision: bool = False  # True: GPU/float32 场景，MSE 只做容差比较


def _load_mse_index(cache_dir: Path) -> dict[str, float]:
    path = cache_dir / "nmf_reconstruction_mse.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_baseline(
    data_root: Path, out_dir: Path, params: dict[str, Any], *, repeats: int = 1
) -> ScenarioResult:
    """场景 1：utils/preprocessing/offline_nmf.py（未优化原始实现）。

    repeats > 1 时重复运行多次，取墙钟耗时的最小值（对共享机器上的系统级
    抖动更稳健，是常见的基准测试做法），MSE 索引取最后一次运行的结果。
    """
    from utils.preprocessing import offline_nmf as m

    best_wall = float("inf")
    mse_index: dict[str, float] = {}
    for _ in range(repeats):
        if out_dir.exists():
            shutil.rmtree(out_dir)
        t0 = time.perf_counter()
        m.process_data_root(
            data_root,
            out_dir=out_dir,
            k=params["k"], lam1=params["l1"], lam2=params["l2"], lam3=params["l3"],
            max_iter=params["max_iter"], use_simplex=params["use_simplex"],
            lam_e=params["lam_e"], e_clamp_max=params["e_clamp_max"],
            e_warn_max=params["e_warn_max"], verbose=False,
        )
        wall = time.perf_counter() - t0
        best_wall = min(best_wall, wall)
        mse_index = _load_mse_index(out_dir)
    return ScenarioResult("baseline（未优化）", best_wall, mse_index)


def run_optimized(
    data_root: Path,
    out_dir: Path,
    params: dict[str, Any],
    *,
    parallel_workers: int,
    blas_threads_per_worker: int | None,
    label: str,
    repeats: int = 1,
) -> ScenarioResult:
    """场景 2/3：utils/preprocessing/offline_nmf_optimized.py（#1+#2，可选 #5）。

    repeats 用法同 run_baseline。
    """
    from utils.preprocessing import offline_nmf_optimized as m

    best_wall = float("inf")
    mse_index: dict[str, float] = {}
    for _ in range(repeats):
        if out_dir.exists():
            shutil.rmtree(out_dir)
        t0 = time.perf_counter()
        m.process_data_root(
            data_root,
            out_dir=out_dir,
            k=params["k"], lam1=params["l1"], lam2=params["l2"], lam3=params["l3"],
            max_iter=params["max_iter"], use_simplex=params["use_simplex"],
            lam_e=params["lam_e"], e_clamp_max=params["e_clamp_max"],
            e_warn_max=params["e_warn_max"], verbose=False,
            parallel_workers=parallel_workers,
            blas_threads_per_worker=blas_threads_per_worker,
        )
        wall = time.perf_counter() - t0
        best_wall = min(best_wall, wall)
        mse_index = _load_mse_index(out_dir)
    return ScenarioResult(label, best_wall, mse_index)


def run_gpu(
    data_root: Path,
    out_dir: Path,
    params: dict[str, Any],
    *,
    device: str | None,
    dtype: str,
    label: str,
    repeats: int = 1,
) -> ScenarioResult:
    """场景 4：utils/preprocessing/offline_nmf_cuda.py（#1+#2+#6，GPU 迭代）。

    ``parallel_workers`` 固定传 1：GPU 场景下多进程并行没有意义（模块内部也会
    自动强制降级，这里显式传 1 只是避免打印冗余警告）。repeats 用法同上。
    """
    from utils.preprocessing import offline_nmf_cuda as m

    best_wall = float("inf")
    mse_index: dict[str, float] = {}
    for _ in range(repeats):
        if out_dir.exists():
            shutil.rmtree(out_dir)
        t0 = time.perf_counter()
        m.process_data_root(
            data_root,
            out_dir=out_dir,
            k=params["k"], lam1=params["l1"], lam2=params["l2"], lam3=params["l3"],
            max_iter=params["max_iter"], use_simplex=params["use_simplex"],
            lam_e=params["lam_e"], e_clamp_max=params["e_clamp_max"],
            e_warn_max=params["e_warn_max"], verbose=False,
            parallel_workers=1, device=device, dtype=dtype,
        )
        wall = time.perf_counter() - t0
        best_wall = min(best_wall, wall)
        mse_index = _load_mse_index(out_dir)
    return ScenarioResult(label, best_wall, mse_index, is_low_precision=True)


def _compare_mse(baseline: dict[str, float], other: dict[str, float]) -> tuple[float, float]:
    """返回 (平均绝对偏差, 最大绝对偏差)，按两个 MSE 索引的公共 key 比较。"""
    keys = sorted(set(baseline) & set(other))
    if not keys:
        raise RuntimeError("baseline 与对比场景的 MSE 索引没有公共 key，无法比较")
    diffs = [abs(baseline[k] - other[k]) for k in keys]
    return float(np.mean(diffs)), float(np.max(diffs))


def print_report(results: list[ScenarioResult]) -> None:
    baseline = results[0]
    width = 100
    print("\n" + "=" * width)
    print("离线 NMF 性能优化对比报告（耗时 + 重建 MSE）")
    print("=" * width)
    header = (
        f"{'场景':<34}{'耗时(s)':>10}{'加速比':>10}"
        f"{'MSE均值':>13}{'MSE最大值':>13}{'与baseline的MSE最大偏差':>22}"
    )
    print(header)
    print("-" * width)
    has_low_precision = False
    for r in results:
        speedup = baseline.wall_time_s / r.wall_time_s if r.wall_time_s > 0 else float("inf")
        mses = list(r.mse_index.values())
        mean_mse = float(np.mean(mses))
        max_mse = float(np.max(mses))
        if r is baseline:
            diff_str = "-"
        else:
            _, max_diff = _compare_mse(baseline.mse_index, r.mse_index)
            diff_str = f"{max_diff:.3e}" + (" (fp32)" if r.is_low_precision else "")
            has_low_precision = has_low_precision or r.is_low_precision
        print(
            f"{r.name:<34}{r.wall_time_s:>10.2f}{speedup:>9.2f}x"
            f"{mean_mse:>13.6f}{max_mse:>13.6f}{diff_str:>22}"
        )
    print("-" * width)
    print(
        "说明：优化 #1/#2/#5 不改变任何一步 MUR 更新的数值，CPU 场景 MSE 最大偏差\n"
        "     理论上应为 0，实际可能出现 1e-10 量级以内的极小差异（float 求和顺序 /\n"
        "     BLAS 线程数不同导致的正常浮点误差），不影响分解质量；若偏差明显更大，\n"
        "     说明优化实现有问题，需要排查。"
    )
    if has_low_precision:
        print(
            "     标注 (fp32) 的 gpu 场景使用 float32 在 GPU 上迭代，与 CPU/float64\n"
            "     基线不是逐位相同的计算路径，预期会有比上述量级更大、但仍应很小的\n"
            "     偏差（通常在 1e-3~1e-5 相对量级，取决于 max_iter/K 等参数）；若偏差\n"
            "     达到与 MSE 均值相近的数量级，说明 GPU 实现存在问题，需要排查。"
        )
    print("=" * width + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="对比离线 NMF：baseline / #1+#2优化 / #1+#2+#5(并行) 三种场景的耗时与 MSE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num-images", type=int, default=4, help="合成图像数量")
    p.add_argument("--height", type=int, default=400, help="合成图像高度 H")
    p.add_argument("--width", type=int, default=500, help="合成图像宽度 W")
    p.add_argument("--bands", type=int, default=20, help="波段数 S")
    p.add_argument("--k", type=int, default=16, help="NMF 端元数 K")
    p.add_argument("--l1", type=float, default=5e-4, help="C 的 L1 稀疏正则")
    p.add_argument("--l2", type=float, default=2e-4, help="E 的谱平滑正则")
    p.add_argument("--l3", type=float, default=1e-2, help="E 的多样性/去相关正则")
    p.add_argument("--max-iter", type=int, default=60, help="每张图最大 MUR 迭代数")
    p.add_argument("--lam-e", type=float, default=0.05, help="E 的 L2 正则强度")
    p.add_argument("--e-clamp-max", type=float, default=3.0, help="E 逐元素上界")
    p.add_argument("--e-warn-max", type=float, default=10.0, help="E.max 警告阈值")
    p.add_argument(
        "--use-simplex", dest="use_simplex", action="store_true", default=True,
        help="对 C 施加 simplex 约束（默认开启，与生产默认一致）",
    )
    p.add_argument("--no-use-simplex", dest="use_simplex", action="store_false")
    p.add_argument("--parallel-workers", type=int, default=4, help="optimized_parallel 场景的进程数")
    p.add_argument(
        "--blas-threads-per-worker", type=int, default=4,
        help="optimized_parallel 场景每个 worker 的 BLAS 线程数上限（<=0 表示不限制）",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="gpu 场景使用的 torch 设备，如 cuda:7；默认自动选择（CUDA 可用则用第一块可见 GPU）",
    )
    p.add_argument(
        "--dtype", type=str, default="float32", choices=("float32", "float64"),
        help="gpu 场景使用的浮点精度（默认 float32）",
    )
    p.add_argument(
        "--skip-gpu", action="store_true",
        help="跳过 gpu 场景（默认：无可用 CUDA 设备时会自动跳过并打印提示）",
    )
    p.add_argument("--seed", type=int, default=42, help="合成数据随机种子")
    p.add_argument(
        "--repeats", type=int, default=1,
        help="每个场景重复运行次数，取墙钟耗时最小值（应对共享机器负载抖动）",
    )
    p.add_argument(
        "--work-dir", type=str, default=None,
        help="数据 / 缓存临时目录；默认自动创建一个临时目录并在结束后清理",
    )
    p.add_argument("--keep", action="store_true", help="结束后不清理临时目录（便于排查）")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="offline_nmf_bench_"))
        cleanup = not args.keep

    data_root = work_dir
    images_dir = data_root / "images"
    print(
        f"[准备] 合成数据集：{args.num_images} 张 "
        f"{args.height}x{args.width}x{args.bands} → {images_dir}"
    )
    build_synthetic_dataset(
        images_dir, args.num_images, args.height, args.width, args.bands, args.seed
    )

    params: dict[str, Any] = dict(
        k=args.k, l1=args.l1, l2=args.l2, l3=args.l3, max_iter=args.max_iter,
        use_simplex=args.use_simplex, lam_e=args.lam_e, e_clamp_max=args.e_clamp_max,
        e_warn_max=args.e_warn_max,
    )
    print(f"[参数] {params}  repeats={args.repeats}")
    try:
        load1, load5, load15 = os.getloadavg()
        print(
            f"[系统负载] loadavg(1/5/15min)={load1:.1f}/{load5:.1f}/{load15:.1f}"
            f"  cpu_count={os.cpu_count()}"
            "  ← 共享机器上负载较高时，耗时对比会有噪声，建议加大 --repeats"
        )
    except (OSError, AttributeError):
        pass

    results: list[ScenarioResult] = []

    print("\n[场景 1/3] baseline（offline_nmf.py，未优化）...")
    r_base = run_baseline(data_root, work_dir / "cache_baseline", params, repeats=args.repeats)
    print(f"  → 耗时(最小值) {r_base.wall_time_s:.2f}s")
    results.append(r_base)

    print("\n[场景 2/3] optimized_serial（offline_nmf_optimized.py，#1+#2，parallel_workers=1）...")
    r_opt_serial = run_optimized(
        data_root, work_dir / "cache_opt_serial", params,
        parallel_workers=1, blas_threads_per_worker=None,
        label="optimized_serial (#1+#2)", repeats=args.repeats,
    )
    print(f"  → 耗时(最小值) {r_opt_serial.wall_time_s:.2f}s")
    results.append(r_opt_serial)

    blas_threads = args.blas_threads_per_worker if args.blas_threads_per_worker > 0 else None
    print(
        f"\n[场景 3/3] optimized_parallel（offline_nmf_optimized.py，#1+#2+#5，"
        f"parallel_workers={args.parallel_workers}，blas_threads_per_worker={blas_threads}）..."
    )
    r_opt_parallel = run_optimized(
        data_root, work_dir / "cache_opt_parallel", params,
        parallel_workers=args.parallel_workers, blas_threads_per_worker=blas_threads,
        label=f"optimized_parallel (#1+#2+#5, w={args.parallel_workers})", repeats=args.repeats,
    )
    print(f"  → 耗时(最小值) {r_opt_parallel.wall_time_s:.2f}s")
    results.append(r_opt_parallel)

    gpu_available = False
    if not args.skip_gpu:
        try:
            import torch

            gpu_available = torch.cuda.is_available()
        except ImportError:
            gpu_available = False

    if args.skip_gpu:
        print("\n[场景 4/4] gpu：显式 --skip-gpu，跳过。")
    elif not gpu_available:
        print("\n[场景 4/4] gpu：当前环境无可用 CUDA 设备，跳过。")
    else:
        print(
            f"\n[场景 4/4] gpu（offline_nmf_cuda.py，#1+#2+#6，"
            f"device={args.device or '<auto>'}，dtype={args.dtype}）..."
        )
        r_gpu = run_gpu(
            data_root, work_dir / "cache_gpu", params,
            device=args.device, dtype=args.dtype,
            label=f"gpu (#1+#2+#6, {args.dtype})", repeats=args.repeats,
        )
        print(f"  → 耗时(最小值) {r_gpu.wall_time_s:.2f}s")
        results.append(r_gpu)

    print_report(results)

    if cleanup:
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        print(f"[保留] 临时目录未清理：{work_dir}")


if __name__ == "__main__":
    main()
