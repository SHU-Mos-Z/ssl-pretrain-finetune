#!/usr/bin/env python3
"""
读取 get_hyperparm_schedule.py 生成的 CSV，依次执行 NMF-ViT 预训练 → 分割微调，
并将测试指标（Dice/IOU/HD95）尾接写回 CSV。

用法:
    python auto_pretrain_finetune.py --schedule schedule_MDC.csv --gpus 0,1
    python auto_pretrain_finetune.py --schedule schedule_MDC.csv --dry-run
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

METRIC_COLUMNS = ("Dice", "IOU", "HD95")
PROJECT_ROOT = Path(__file__).parent.resolve()
TORCHRUN = "torchrun"


def find_free_port(lo: int = 29500, hi: int = 30000) -> int:
    import random

    candidates = list(range(lo, hi))
    random.shuffle(candidates)
    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"在端口范围 [{lo}, {hi}) 内找不到可用端口")


def run_with_tee(cmd: list[str], log_path: str, env: dict | None = None) -> int:
    with open(log_path, "a", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(PROJECT_ROOT),
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        proc.wait()
    return proc.returncode


def find_pretrain_ckpt(pretrain_dir: str) -> str | None:
    best = os.path.join(pretrain_dir, "ckpt_best.pth")
    if os.path.isfile(best):
        return best
    candidates = sorted(_glob.glob(os.path.join(pretrain_dir, "ckpt_epoch*.pth")))
    return candidates[-1] if candidates else None


_METRIC_RE = re.compile(
    r"测试集结果.*?Dice=([0-9.]+).*?IoU=([0-9.]+)(?:.*?HD95=([0-9.]+))?",
    re.IGNORECASE,
)


def parse_test_metrics(log_path: str) -> dict:
    dice = iou = hd95 = float("nan")
    if not os.path.isfile(log_path):
        return {"Dice": dice, "IOU": iou, "HD95": hd95}
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = _METRIC_RE.search(line)
            if m:
                dice = float(m.group(1))
                iou = float(m.group(2))
                hd95 = float(m.group(3)) if m.group(3) else float("nan")
    return {"Dice": dice, "IOU": iou, "HD95": hd95}


def _parse_metric_list(val) -> list[float]:
    if val is None:
        return []
    if isinstance(val, list):
        return [float(x) for x in val]
    if isinstance(val, float) and math.isnan(val):
        return []
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return []
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    try:
        return [float(s)]
    except ValueError:
        return []


def _serialize_metric_list(lst: list[float]) -> str:
    return json.dumps(lst)


def _append_metric(df: pd.DataFrame, idx: int, col: str, value: float) -> None:
    current = _parse_metric_list(df.at[idx, col])
    if not math.isnan(value):
        current.append(value)
    df.at[idx, col] = _serialize_metric_list(current)


def _has_results(row) -> bool:
    return len(_parse_metric_list(row.get("Dice", "[]"))) > 0


def _format_metric_list(row, col: str = "Dice") -> str:
    lst = _parse_metric_list(row.get(col, "[]"))
    if not lst:
        return "[]"
    return json.dumps([round(x, 4) for x in lst])


def _to_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes")


def build_pretrain_cmd(
    row: "pd.Series", pretrain_save_dir: str, num_gpus: int
) -> list[str]:
    data_roots = str(row["DATA_ROOTS"]).split(";")
    cmd = [
        TORCHRUN,
        f"--nproc_per_node={num_gpus}",
        f"--master-port={find_free_port()}",
        "train_pretrain_vit.py",
        "--root",
        *data_roots,
        "--patch-size",
        str(int(float(row["PATCH_SIZE"]))),
        "--spectral-patch-size",
        str(int(float(row["SPECTRAL_PATCH_SIZE"]))),
        "--mask-ratio",
        str(float(row["MASK_RATIO"])),
        "--sobel-tau",
        str(float(row["SOBEL_TAU"])),
        "--spectral-alpha",
        str(float(row["SPECTRAL_ALPHA"])),
        "--nmf-k",
        str(int(float(row["NMF_K"]))),
        "--nmf-l1",
        str(float(row["NMF_L1"])),
        "--nmf-l2",
        str(float(row["NMF_L2"])),
        "--nmf-l3",
        str(float(row["NMF_L3"])),
        "--epochs",
        str(int(float(row["EPOCHS"]))),
        "--batch-size",
        str(int(float(row["BATCH_SIZE_PER_GPU"]))),
        "--lr",
        str(float(row["LR"])),
        "--min-lr",
        str(float(row["MIN_LR"])),
        "--weight-decay",
        str(float(row["WEIGHT_DECAY"])),
        "--warmup-epochs",
        str(int(float(row["WARMUP_EPOCHS"]))),
        "--clip-grad",
        str(float(row["CLIP_GRAD"])),
        "--seed",
        str(int(float(row["SEED"]))),
        "--embed-dim",
        str(int(float(row["EMBED_DIM"]))),
        "--vit-depth",
        str(int(float(row["VIT_DEPTH"]))),
        "--vit-heads",
        str(int(float(row["VIT_HEADS"]))),
        "--num-endmembers",
        str(int(float(row["NUM_ENDMEMBERS"]))),
        "--aggregate-mode",
        str(row["AGGREGATE_MODE"]),
        "--abundance-act",
        str(row["ABUNDANCE_ACT"]),
        "--od-max",
        str(float(row["OD_MAX"])),
        "--lambda-od",
        str(float(row["LAMBDA_OD"])),
        "--lambda-i",
        str(float(row["LAMBDA_I"])),
        "--lambda-cons-pix",
        str(float(row["LAMBDA_CONS_PIX"])),
        "--lambda-cons-token",
        str(float(row["LAMBDA_CONS_TOKEN"])),
        "--lambda-anchor",
        str(float(row.get("LAMBDA_ANCHOR", 0.1))),
        "--workers",
        str(int(float(row["WORKERS"]))),
        "--save-interval",
        str(int(float(row["SAVE_INTERVAL"]))),
        "--save-dir",
        pretrain_save_dir,
        "--progress",
        "log",
    ]
    if _to_bool(row["USE_GRADIENT_MASKING"]):
        cmd.append("--use-gradient-masking")
    if _to_bool(row.get("USE_CONS_TOKEN", True)):
        cmd.append("--use-cons-token")
    else:
        cmd.append("--no-use-cons-token")
    if _to_bool(row.get("USE_REFINE", True)):
        cmd.append("--use-refine")
    else:
        cmd.append("--no-use-refine")
    if _to_bool(row.get("AMP", True)):
        cmd.append("--amp")
    return cmd


def build_finetune_cmd(
    row: "pd.Series",
    pretrain_ckpt: str,
    finetune_save_dir: str,
    num_gpus: int,
) -> list[str]:
    cmd = [
        TORCHRUN,
        f"--nproc_per_node={num_gpus}",
        f"--master-port={find_free_port()}",
        "train_finetune_vit.py",
        "--train-root",
        str(row["FT_TRAIN_ROOT"]),
        "--val-root",
        str(row["FT_VAL_ROOT"]),
        "--test-root",
        str(row["FT_TEST_ROOT"]),
        "--epochs",
        str(int(float(row["FT_EPOCHS"]))),
        "--batch-size",
        str(int(float(row["FT_BATCH_SIZE_PER_GPU"]))),
        "--lr",
        str(float(row["FT_LR"])),
        "--min-lr",
        str(float(row["FT_MIN_LR"])),
        "--weight-decay",
        str(float(row["FT_WEIGHT_DECAY"])),
        "--warmup-epochs",
        str(int(float(row["FT_WARMUP_EPOCHS"]))),
        "--clip-grad",
        str(float(row["FT_CLIP_GRAD"])),
        "--seed",
        str(int(float(row["FT_SEED"]))),
        "--num-classes",
        str(int(float(row["FT_NUM_CLASSES"]))),
        "--embed-dim",
        str(int(float(row["FT_EMBED_DIM"]))),
        "--vit-depth",
        str(int(float(row["FT_VIT_DEPTH"]))),
        "--vit-heads",
        str(int(float(row["FT_VIT_HEADS"]))),
        "--patch-size",
        str(int(float(row["FT_PATCH_SIZE"]))),
        "--spectral-patch-size",
        str(int(float(row["FT_SPECTRAL_PATCH_SIZE"]))),
        "--num-endmembers",
        str(int(float(row["FT_NUM_ENDMEMBERS"]))),
        "--aggregate-mode",
        str(row["FT_AGGREGATE_MODE"]),
        "--workers",
        str(int(float(row["FT_WORKERS"]))),
        "--save-interval",
        str(int(float(row["SAVE_INTERVAL"]))),
        "--n-vis",
        str(int(float(row["N_VIS"]))),
        "--save-dir",
        finetune_save_dir,
        "--progress",
        "log",
        "--pretrain-ckpt",
        pretrain_ckpt,
    ]
    if _to_bool(row["EARLY_STOP"]):
        cmd += ["--early-stop", "--patience", str(int(float(row["PATIENCE"])))]
    if _to_bool(row.get("FT_AMP", True)):
        cmd.append("--amp")
    return cmd


def _section(log_path: str, title: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n  {title}  [{ts}]\n{'='*60}\n\n")


def _session_log(session_log_path: str, exp_id: int, status: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(session_log_path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}]  exp_{exp_id:04d}  {status}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="NMF-ViT 预训练→微调自动调度")
    parser.add_argument("--schedule", "-s", required=True)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--only", type=int, nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-done", action="store_true")
    args = parser.parse_args()

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        num_gpus = len(visible.split(","))
    else:
        try:
            import torch

            num_gpus = max(torch.cuda.device_count(), 1)
        except Exception:
            num_gpus = 1

    print(
        f"[auto] GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}  nproc={num_gpus}"
    )

    schedule_path = Path(args.schedule).resolve()
    df = pd.read_csv(schedule_path)
    print(f"[auto] 调度表: {schedule_path}  共 {len(df)} 行")

    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = PROJECT_ROOT / "records" / "scheduled_pretrain_finetune" / session_ts
    session_log_path = str(base_dir / f"session_{session_ts}.log")
    base_dir.mkdir(parents=True, exist_ok=True)

    with open(session_log_path, "w", encoding="utf-8") as f:
        f.write(f"session_start : {session_ts}\nschedule      : {schedule_path}\n")

    only_set = set(args.only) if args.only else None
    completed = skipped = failed = 0

    for idx, row in df.iterrows():
        exp_id = int(row.get("exp_id", idx))
        if only_set is not None and exp_id not in only_set:
            continue
        if only_set is None and exp_id < args.start_from:
            skipped += 1
            continue
        if args.skip_done and _has_results(row):
            print(f"[auto] exp_{exp_id:04d} 已有结果，跳过")
            skipped += 1
            continue

        exp_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = base_dir / f"exp_{exp_id:04d}_{exp_ts}"
        pretrain_dir = exp_dir / "pretrain_ckpts"
        finetune_dir = exp_dir / "finetune_ckpts"
        log_path = str(exp_dir / "records.txt")
        exp_dir.mkdir(parents=True)
        pretrain_dir.mkdir()
        finetune_dir.mkdir()

        print(f"\n{'='*64}\n[auto] ▶ exp_{exp_id:04d}  {exp_dir}\n{'='*64}")
        env = {**os.environ, "OMP_NUM_THREADS": "2"}

        pretrain_cmd = build_pretrain_cmd(row, str(pretrain_dir), num_gpus)
        if args.dry_run:
            print(f"  [dry-run] 预训练:\n    {' '.join(pretrain_cmd)}")
        else:
            _section(log_path, "预训练 (Stage 1/2)")
            ret = run_with_tee(pretrain_cmd, log_path, env=env)
            if ret != 0:
                _session_log(session_log_path, exp_id, f"FAILED pretrain exit={ret}")
                failed += 1
                continue

        if args.dry_run:
            finetune_cmd = build_finetune_cmd(
                row,
                str(pretrain_dir / "ckpt_best.pth"),
                str(finetune_dir),
                num_gpus,
            )
            print(f"  [dry-run] 微调:\n    {' '.join(finetune_cmd)}")
            skipped += 1
            continue

        pretrain_ckpt = find_pretrain_ckpt(str(pretrain_dir))
        if pretrain_ckpt is None:
            _session_log(session_log_path, exp_id, "FAILED no_ckpt")
            failed += 1
            continue

        finetune_cmd = build_finetune_cmd(
            row, pretrain_ckpt, str(finetune_dir), num_gpus
        )
        _section(log_path, f"微调 (Stage 2/2)  ckpt={pretrain_ckpt}")
        ret = run_with_tee(finetune_cmd, log_path, env=env)
        if ret != 0:
            _session_log(session_log_path, exp_id, f"FAILED finetune exit={ret}")
            failed += 1
            continue

        metrics = parse_test_metrics(log_path)
        for col in METRIC_COLUMNS:
            _append_metric(df, idx, col, metrics[col])
        df.to_csv(schedule_path, index=False)
        res_str = (
            f"Dice={_format_metric_list(df.loc[idx], 'Dice')}  "
            f"IOU={_format_metric_list(df.loc[idx], 'IOU')}  "
            f"HD95={_format_metric_list(df.loc[idx], 'HD95')}"
        )
        print(f"[auto] 测试指标: {res_str}")
        _session_log(session_log_path, exp_id, f"done  {res_str}")
        completed += 1

    print(f"\n[auto] 完成={completed}  跳过={skipped}  失败={failed}")
    print(f"       调度表: {schedule_path}")


if __name__ == "__main__":
    main()
