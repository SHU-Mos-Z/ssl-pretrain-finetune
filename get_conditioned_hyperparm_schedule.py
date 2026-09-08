#!/usr/bin/env python3
from __future__ import annotations

import argparse,itertools
from pathlib import Path
import pandas as pd


def build_schedule():
    rows=[]
    roots='data/MDC_Resized_256_320_to_256_256_overlap_0_0_preprocessed;data/GPCC_Resized_512_640_to_256_256_overlap_0_0_preprocessed'
    for exp_id,(spectral,spatial,lc,lt,lf) in enumerate(itertools.product(
        [0.2,0.3],[0.1,0.2],[0.1,0.2],[0.5,1.0],[0.0,0.1])):
        rows.append(dict(exp_id=exp_id,DATA_ROOTS_STR=roots,SPECTRAL_MASK_RATIO=spectral,
          SPATIAL_MASK_RATIO=spatial,LAMBDA_C=lc,LAMBDA_TOKEN=lt,LAMBDA_FEATURE=lf,
          EPOCHS=200,BATCH_SIZE_PER_GPU=2,LR=4e-4,SEED=42,
          TRAIN_ROOT='data/MDC_Resized_256_320_to_256_256_overlap_0_0_preprocessed_finetune_train',
          VAL_ROOT='data/MDC_Resized_256_320_to_256_256_overlap_0_0_preprocessed_finetune_val',
          TEST_ROOT='data/MDC_Resized_256_320_to_256_256_overlap_0_0_preprocessed_finetune_test',
          FT_EPOCHS=100,FT_LR=5e-4,status='pending',Dice='',IOU='',HD95=''))
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('-o','--output',default='schedules/conditioned_schedule.csv');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    df=build_schedule();print(f'experiments={len(df)}')
    if not a.dry_run:
        Path(a.output).parent.mkdir(parents=True,exist_ok=True);df.to_csv(a.output,index=False);print(a.output)


if __name__=='__main__':main()
