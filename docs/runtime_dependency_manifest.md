# DFS3R 最小运行依赖清单

本清单的边界是：保留当前 `scripts/` 中预训练、分割微调、分类微调、检测微调、数据划分和离线 NMF Bash 调用链运行所需的第一方代码；不将数据、权重和实验产物收入 Git。

## Git 中必须保留

- `scripts/`：Bash 启动器、顺序实验入口和小型辅助程序。
- `models/`：预训练、分类、分割和检测模型实现。
- `utils/`：Dataset、数据增强、NMF、损失、指标、监控和调度实现。
- 根目录 `train_*.py`：所有预训练与微调入口。
- `evaluate_conditioned_detection.py`：检测独立评估入口。
- `split_pretrain_finetune.py`：预训练/微调数据划分入口。
- `auto_pretrain_finetune.py` 与 `auto_conditioned_pretrain_finetune.py`：现有编排入口。
- `tests/`：与运行链同步的回归测试。
- `configs/`：可复现实验必需的小型非敏感配置。
- `.gitignore`、`environment.yml`、`.env.example`、`pytest.ini` 和 `README.md`。

当前静态盘点包含 115 个 Bash 文件，其中 96 个直接启动微调程序，5 个是顺序执行多个子 Bash 的批量入口，4 个用于数据划分。

## 已从 `data/` 迁出的运行依赖

- `scripts/preprocessing/puad_prepare_scene_endmembers.py`：原为 `data/original/puad_prepare_scene_endmembers.py`。文件内部实现未变，仅 Bash 调用路径变化。
- `configs/sample_exclusions/plgc/*.json`：原为 `data/original/plgc_cls_hard_case_reports/*.json`。JSON 内容未变，仅 PLGC 实验脚本的配置路径变化。

## Git 之外的运行输入

- `data/`：原始和预处理数据、split、NMF cache、波长与标注。
- `records/`：预训练 checkpoint、微调 checkpoint、日志、曲线和指标。
- `PRETRAIN_CKPT`：必须在目标服务器上另行提供，并通过环境变量或修改 Bash 默认值指定。
- `TRAIN_ROOT` / `VAL_ROOT` / `TEST_ROOT`：必须指向同一份手动选定的 split 三元组。

`sam2/`、`sam3/`、`MedSAM/` 和 `MIDOGpp/` 是独立管理的第三方仓库，不是当前训练 Bash 的必需依赖。`scripts/` 中与 SAM3 数据生成相关的可选工具若单独使用，需在本地另行准备 SAM3 仓库、权重和原始数据。

## 维护规则

1. 新的第一方可执行代码不应放入 `data/` 或 `records/`。
2. 小型实验配置放入 `configs/`，数据生成程序放入 `scripts/preprocessing/`。
3. 新增 Bash 后必须在全新 clone 中检查其 Python、子 Bash、配置文件和模块导入依赖。
4. 禁止提交账号密码、token、SSH 私钥和包含患者信息的数据。
