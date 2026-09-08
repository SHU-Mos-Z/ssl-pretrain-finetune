#!/usr/bin/env python3
from __future__ import annotations

import argparse, os, re, subprocess
from pathlib import Path
import pandas as pd


def run(script, env, dry):
    print(f"RUN {script}")
    if dry:
        return 0
    return subprocess.run(
        ["bash", script],
        env={**os.environ, **{k: str(v) for k, v in env.items()}},
        check=False,
    ).returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-s", "--schedule", required=True)
    p.add_argument("--only", type=int, nargs="*")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    df = pd.read_csv(a.schedule, keep_default_na=False)
    for i, row in df.iterrows():
        eid = int(row["exp_id"])
        if a.only and eid not in a.only:
            continue
        base = Path("records/conditioned_sweeps") / f"exp_{eid:04d}"
        pre = base / "pretrain"
        fine = base / "finetune"
        pre_env = {
            k: row[k]
            for k in (
                "DATA_ROOTS_STR",
                "SPECTRAL_MASK_RATIO",
                "SPATIAL_MASK_RATIO",
                "LAMBDA_C",
                "LAMBDA_TOKEN",
                "LAMBDA_FEATURE",
                "EPOCHS",
                "BATCH_SIZE_PER_GPU",
                "LR",
                "SEED",
            )
        }
        pre_env["SAVE_DIR"] = pre
        if run("scripts/run_pretrain_conditioned.sh", pre_env, a.dry_run):
            df.at[i, "status"] = "pretrain_failed"
            df.to_csv(a.schedule, index=False)
            continue
        ckpt = pre / "ckpt_last.pth"
        ft_env = {k: row[k] for k in ("TRAIN_ROOT", "VAL_ROOT", "TEST_ROOT")}
        ft_env.update(
            PRETRAIN_CKPT=ckpt,
            SAVE_DIR=fine,
            EPOCHS=row["FT_EPOCHS"],
            LR=row["FT_LR"],
            SEED=row["SEED"],
        )
        if run("scripts/run_finetune_conditioned.sh", ft_env, a.dry_run):
            df.at[i, "status"] = "finetune_failed"
        else:
            df.at[i, "status"] = "done"
            log = fine / "records.txt"
            if log.is_file():
                matches = re.findall(
                    r"Test Dice=([\d.]+) IoU=([\d.]+) HD95=([\d.]+)",
                    log.read_text(errors="ignore"),
                )
                if matches:
                    df.at[i, "Dice"], df.at[i, "IOU"], df.at[i, "HD95"] = matches[-1]
        if not a.dry_run:
            df.to_csv(a.schedule, index=False)


if __name__ == "__main__":
    main()
