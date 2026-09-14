# PUAD 分割性能改进实验

数据固定为新的 background-aware PUAD 数据：

- `/home/zsq/processed_data/DFS3R-main/data/LUAD_PUAD_official224_bgaware_fg3138_bg1569_fullsceneval/train`
- `/home/zsq/processed_data/DFS3R-main/data/LUAD_PUAD_official224_bgaware_fg3138_bg1569_fullsceneval/val`
- `/home/zsq/processed_data/DFS3R-main/data/LUAD_PUAD_official224_bgaware_fg3138_bg1569_fullsceneval/val_scenes`
- `/home/zsq/processed_data/DFS3R-main/data/LUAD_PUAD_official224_bgaware_fg3138_bg1569_fullsceneval/test`

训练和普通验证输入为 `224×224×40` patch；`val_scenes` 与测试为完整场景，
通过 `224` 窗口、`112` stride 和 Gaussian 融合完成滑窗推理。默认每10轮
执行一次完整场景验证，并以其主 Dice 选择 checkpoint；每轮 patch 验证继续
用于低成本训练监控。默认仍使用 `tok16-sp5` 和
`records/pretrain_conditioned/20260817_005610/ckpt_last.pth`。

训练集由 3138 个前景中心 patch 和 1569 个背景中心 patch 构成；验证 patch
由 522 个前景中心 patch 和 261 个背景中心 patch 构成。该实验矩阵保留0906
的增强、分割头、损失和优化器消融设置，以便隔离数据预处理变化带来的影响。

| 脚本 | 实验 |
|---|---|
| `00_p_00_default.sh` | 严格历史基线：无增强、H0、CE+Dice、统一LR |
| `01_p_a1_dihedral.sh` | Dihedral增强 |
| `02_p_a2_dihedral_affine.sh` | Dihedral + 轻度仿射 |
| `03_p_a3_dihedral_perspective.sh` | Dihedral + 轻度透视 |
| `04_p_h1_residual.sh` | H1残差细化头 |
| `05_p_h2_aspp.sh` | H2 ASPP头 |
| `06_p_h3_multiscale_aux.sh` | H3 decoder多尺度融合与辅助监督 |
| `07_p_l1_weighted_ce_dice.sh` | 逆平方根像素频率加权CE+Dice |
| `08_p_l2_focal_dice.sh` | Focal+Dice |
| `09_p_l3_weighted_boundary.sh` | 加权CE+Dice+Boundary Dice |
| `10_p_o1_discriminative_lr.sh` | Backbone使用0.1倍学习率 |
| `11_p_o2_staged_unfreeze.sh` | 前20轮保持Backbone权重不更新 |
| `12_p_e1_scene_endmembers.sh` | 训练/验证/测试统一场景级端元 |
| `13_p_final_seed42.sh` | 默认推荐组合，seed 42 |
| `14_p_final_seed43.sh` | 同一最终组合，seed 43 |
| `15_p_final_seed44.sh` | 同一最终组合，seed 44 |

H1及后续脚本中的 `BEST_AUGMENTATION_POLICY`、`SCREEN_SEGMENTATION_HEAD`、
`SCREEN_SEGMENTATION_LOSS` 用于在前一阶段得出结论后覆盖默认筛选配置。
三个Final脚本必须使用完全相同的 `FINAL_*` 设置。

顺序脚本会统一导出 `LR` 给全部单项实验。当前默认总 batch 为
`1 GPU × 8/GPU = 8`，默认使用 `LR=4e-4`。可以在命令行覆盖全部顺序实验：

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 BATCH_SIZE_PER_GPU=4 LR=4e-4 \
  bash scripts/test_seg_puad_0914/16_run_00_to_15_sequentially.sh
```

直接运行任一 `00`～`15` 脚本时，如外部没有传入 `LR`，该脚本仍使用原有的
`4e-4` 默认值。

P-E1运行前先执行：

```bash
DATASET_ROOT=/home/zsq/processed_data/DFS3R-main/data/LUAD_PUAD_official224_bgaware_fg3138_bg1569_fullsceneval \
  bash scripts/run_offline_nmf_puad_scene_endmembers.sh
```

顺序执行脚本在场景端元缓存不存在时会跳过P-E1，不会错误回退到patch端元。

滑窗推理消融不需要重新训练。使用与checkpoint结构相匹配的实验脚本，并设置：

```bash
EVAL_ONLY_CHECKPOINT=/path/to/ckpt.pth \
TEST_WINDOW_STRIDE=56 \
TEST_WINDOW_BLEND=gaussian \
SAVE_DIR=records/test_seg_puad_0914/inference_stride56 \
bash scripts/test_seg_puad_0914/13_p_final_seed42.sh
```

分别令 `stride/blend` 为 `112/gaussian`、`56/gaussian` 和
`112/uniform` 即可得到I-0、I-1、I-2。Eval-only会跳过优化器和训练循环。
