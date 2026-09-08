#!/usr/bin/env python3
"""
生成 NMF-ViT 预训练 + 微调实验的超参数调度 CSV。

用法:
    python get_hyperparm_schedule.py
    python get_hyperparm_schedule.py --output schedules/schedule_MDC.csv --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from itertools import combinations as iter_combinations

import pandas as pd

_EMPTY_METRIC_LIST = json.dumps([])

# ===========================================================================
# 用户配置
# ===========================================================================

AVAILABLE_PRETRAIN_DATASETS = [
    "data/MDC_Resized_256_320_to_256_256_overlap_0_0_preprocessed",
]

PRETRAIN_ENUM_PARAMS: dict[str, list] = {
    "PATCH_SIZE": [16],
    "SPECTRAL_PATCH_SIZE": [10],
    "MASK_RATIO": [0.4, 0.6],
    "USE_GRADIENT_MASKING": [True, False],
    "SOBEL_TAU": [1.0],
    "SPECTRAL_ALPHA": [1.0],
    "AGGREGATE_MODE": ["mean", "attention"],
    "LAMBDA_CONS_PIX": [0.3, 0.5, 1.0],
    "LAMBDA_CONS_TOKEN": [0.3, 0.5, 1.0],
}

PRETRAIN_FIXED_PARAMS: dict = {
    "EPOCHS": 100,
    "BATCH_SIZE_PER_GPU": 4,
    "LR": 4e-4,
    "MIN_LR": 2e-6,
    "WEIGHT_DECAY": 0.05,
    "WARMUP_EPOCHS": 10,
    "CLIP_GRAD": 1.0,
    "SEED": 42,
    "EMBED_DIM": 256,
    "VIT_DEPTH": 6,
    "VIT_HEADS": 8,
    "NUM_ENDMEMBERS": 2,
    "ABUNDANCE_ACT": "softmax",
    "USE_REFINE": True,
    "OD_MAX": 3.0,
    "LAMBDA_OD": 1.0,
    "LAMBDA_I": 1.0,
    "USE_CONS_TOKEN": True,
    "LAMBDA_ANCHOR": 0.1,
    "NMF_K": 2,
    "NMF_L1": 1e-3,
    "NMF_L2": 1e-4,
    "NMF_L3": 1e-2,
    "WORKERS": 4,
    "SAVE_INTERVAL": 10,
    "NUM_GPUS": 2,
    "AMP": True,
}

FINETUNE_FIXED_PARAMS: dict = {
    "FT_TRAIN_ROOT": "data/MDC_Resized_256_320_to_256_256_overlap_0_0_preprocessed_train",
    "FT_VAL_ROOT": "data/MDC_Resized_256_320_to_256_256_overlap_0_0_preprocessed_val",
    "FT_TEST_ROOT": "data/MDC_Resized_256_320_to_256_256_overlap_0_0_preprocessed_test",
    "FT_EPOCHS": 100,
    "FT_BATCH_SIZE_PER_GPU": 4,
    "FT_LR": 5e-4,
    "FT_MIN_LR": 1e-6,
    "FT_WEIGHT_DECAY": 1e-4,
    "FT_WARMUP_EPOCHS": 5,
    "FT_CLIP_GRAD": 1.0,
    "FT_SEED": 42,
    "FT_NUM_CLASSES": 2,
    "FT_EMBED_DIM": 256,
    "FT_VIT_DEPTH": 6,
    "FT_VIT_HEADS": 8,
    "FT_PATCH_SIZE": 16,
    "FT_SPECTRAL_PATCH_SIZE": 10,
    "FT_NUM_ENDMEMBERS": 2,
    "FT_AGGREGATE_MODE": "mean",
    "EARLY_STOP": False,
    "PATIENCE": 20,
    "FT_WORKERS": 4,
    "FT_NUM_GPUS": 2,
    "FT_AMP": True,
    "N_VIS": 8,
}


def is_param_needed(param_name: str, fixed: dict) -> bool:
    use_grad = bool(fixed.get("USE_GRADIENT_MASKING", True))
    if param_name in ("SOBEL_TAU", "SPECTRAL_ALPHA"):
        return use_grad
    return True


def _enumerate_recursive(param_names, param_values, current, rows, base_row):
    if not param_names:
        rows.append({**base_row, **current})
        return
    name = param_names[0]
    rest = param_names[1:]
    values = (
        param_values[name]
        if is_param_needed(name, current)
        else [param_values[name][0]]
    )
    for v in values:
        _enumerate_recursive(rest, param_values, {**current, name: v}, rows, base_row)


def build_schedule(
    available_datasets, enum_params, pretrain_fixed, finetune_fixed
) -> pd.DataFrame:
    data_roots_options = []
    for k in range(1, len(available_datasets) + 1):
        for combo in iter_combinations(available_datasets, k):
            data_roots_options.append(";".join(combo))

    base_fixed = {**pretrain_fixed, **finetune_fixed}
    param_names = list(enum_params.keys())
    rows: list[dict] = []
    for dr in data_roots_options:
        _enumerate_recursive(
            param_names, enum_params, {}, rows, {"DATA_ROOTS": dr, **base_fixed}
        )

    df = pd.DataFrame(rows)
    col_order = (
        ["DATA_ROOTS"]
        + param_names
        + list(pretrain_fixed.keys())
        + list(finetune_fixed.keys())
    )
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order].reset_index(drop=True)
    df.insert(0, "exp_id", range(len(df)))
    for m in ("Dice", "IOU", "HD95"):
        df[m] = _EMPTY_METRIC_LIST
    return df


def _infer_schedule_filename(finetune_fixed: dict) -> str:
    base = os.path.basename(finetune_fixed.get("FT_TRAIN_ROOT", "").rstrip("/"))
    m = re.match(r"^([A-Z]+)", base)
    return f"schedule_{m.group(1) if m else 'UNKNOWN'}.csv"


def _print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 64)
    print(f"  总实验数: {len(df)}")
    print("=" * 64)


def main() -> None:
    default_output = _infer_schedule_filename(FINETUNE_FIXED_PARAMS)
    parser = argparse.ArgumentParser(description="生成 NMF-ViT 超参数调度 CSV")
    parser.add_argument("--output", "-o", default=default_output)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    df = build_schedule(
        AVAILABLE_PRETRAIN_DATASETS,
        PRETRAIN_ENUM_PARAMS,
        PRETRAIN_FIXED_PARAMS,
        FINETUNE_FIXED_PARAMS,
    )
    _print_summary(df)
    if args.dry_run:
        print(f"[dry-run] 未保存，共 {len(df)} 组")
        return
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[完成] 已保存至 {args.output}")


if __name__ == "__main__":
    main()
