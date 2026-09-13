# GPCC 分割性能改进实验

固定数据为：

- `data/GPCC_Resized_512_640_to_256_256_overlap_0_0_preprocessed_finetune_train_p070_20260728`
- `data/GPCC_Resized_512_640_to_256_256_overlap_0_0_preprocessed_finetune_val_p015_20260728`
- `data/GPCC_Resized_512_640_to_256_256_overlap_0_0_preprocessed_finetune_test_p015_20260728`

三个 split 分别包含 277、60、60 对 `256×256×40` 图像与二分类掩膜，
均采用直接推理。当前数据没有 `wavelengths.npy`，所以脚本默认使用归一化波段
索引；如取得真实的40波段波长表，可通过 `WAVELENGTH_FILE` 指定。

| 脚本 | 实验 |
|---|---|
| `00_g_00_default.sh` | G-00：无增强、H0、CE+Dice、统一学习率基线 |
| `01_g_a1_dihedral.sh` | G-A1：Dihedral增强 |
| `02_g_a2_dihedral_affine.sh` | G-A2：Dihedral + 轻度仿射 |
| `03_g_a3_dihedral_perspective.sh` | G-A3：Dihedral + 轻度透视 |
| `04_g_h1_residual.sh` | G-H1：残差细化分割头 |
| `05_g_h2_aspp.sh` | G-H2：ASPP分割头 |
| `06_g_h3_multiscale_aux.sh` | G-H3：decoder多尺度融合与辅助监督 |
| `07_g_l1_weighted_ce_dice.sh` | G-L1：逆平方根像素频率加权CE+Dice |
| `08_g_l2_focal_dice.sh` | G-L2：Focal+Dice |
| `09_g_l3_weighted_boundary.sh` | G-L3：加权CE+Dice+Boundary Dice |
| `10_g_o1_discriminative_lr.sh` | G-O1：backbone使用0.1倍学习率 |
| `11_g_o2_staged_unfreeze.sh` | G-O2：前20轮保持backbone权重不更新 |
| `12_g_final_seed42.sh` | 最终配置，seed 42 |
| `13_g_final_seed43.sh` | 最终配置，seed 43 |
| `14_g_final_seed44.sh` | 最终配置，seed 44 |

H、L、O阶段支持通过 `BEST_AUGMENTATION_POLICY`、
`SCREEN_SEGMENTATION_HEAD`、`SCREEN_AUX_LOSS_WEIGHT`、
`SCREEN_SEGMENTATION_LOSS`、`SCREEN_CLASS_WEIGHT_MODE` 和
`SCREEN_BOUNDARY_LOSS_WEIGHT` 固定完整的前置筛选结果。例如选择 H3 时必须同时
令 `SCREEN_AUX_LOSS_WEIGHT=0.4`；选择加权边界损失时应同时设置
`SCREEN_CLASS_WEIGHT_MODE=inverse_sqrt` 和 `SCREEN_BOUNDARY_LOSS_WEIGHT=0.2`。
三个最终脚本通过同一组 `FINAL_*` 参数确保除随机种子外的设置完全一致。

依次执行全部实验：

```bash
bash scripts/test_seg_gpcc_0913/15_run_00_to_14_sequentially.sh
```

可以统一覆盖显卡、batch size、学习率、预训练权重和三个 split：

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 BATCH_SIZE_PER_GPU=4 LR=4e-4 \
PRETRAIN_CKPT=/path/to/ckpt_last.pth \
TRAIN_ROOT=/path/to/train VAL_ROOT=/path/to/val TEST_ROOT=/path/to/test \
bash scripts/test_seg_gpcc_0913/15_run_00_to_14_sequentially.sh
```

这套脚本固定使用现有 split，以便与本项目既有 GPCC 结果进行直接比较。
