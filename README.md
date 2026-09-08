# DFS3R

本仓库仅保存 DFS3R 预训练、分类微调、分割微调、检测微调和数据划分所需的第一方代码与小型配置。数据、NMF cache、预训练权重和实验记录不进入 Git。

## 1. 创建环境

当前实验环境使用 Python 3.10、PyTorch 2.4.1 和 CUDA 12.1 wheel：

```bash
conda env create -f environment.yml
conda activate zsq_accl_mine
```

`environment.yml` 固定了当前有效环境的主要依赖版本。目标服务器的 NVIDIA 驱动需要支持 CUDA 12.1 运行时。

## 2. 准备数据和权重

Git clone 不会生成 `data/` 和 `records/`内容。可以将数据复制到项目目录，也可将共享存储链接到当前项目：

```bash
ln -s /mnt/datasets/dfs3r data
ln -s /mnt/experiments/dfs3r records
```

如果不使用项目内的相对路径，复制环境变量模板：

```bash
cp .env.example .env.local
# 编辑 .env.local 中的绝对路径
set -a
source .env.local
set +a
```

`.env.local` 不会被 Git 跟踪。Bash 启动器会优先使用外部传入的 `CUDA_VISIBLE_DEVICES`、`NUM_GPUS`、`BATCH_SIZE_PER_GPU`、`LR`、`PRETRAIN_CKPT`、`TRAIN_ROOT`、`VAL_ROOT` 和 `TEST_ROOT`。

## 3. 运行实验

单项实验示例：

```bash
bash scripts/run_finetune_conditioned_cls.sh
```

顺序执行实验矩阵示例：

```bash
bash scripts/test_seg_puad_0906/16_run_00_to_15_sequentially.sh
```

也可只对当次命令覆盖设置：

```bash
CUDA_VISIBLE_DEVICES="4,5" \
NUM_GPUS=2 \
BATCH_SIZE_PER_GPU=4 \
LR=4e-4 \
PRETRAIN_CKPT="/mnt/checkpoints/ckpt_last.pth" \
TRAIN_ROOT="/mnt/data/train" \
VAL_ROOT="/mnt/data/val" \
TEST_ROOT="/mnt/data/test" \
bash scripts/test_seg_puad_0906/16_run_00_to_15_sequentially.sh
```

每次微调必须显式选定同一划分产生的 train/val/test 三元组，不依赖自动搜索“最新”目录。

## 4. 基本检查

```bash
find scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
python -m compileall -q models utils scripts *.py
pytest -q
```

更完整的文件边界和外部输入说明见 [docs/runtime_dependency_manifest.md](docs/runtime_dependency_manifest.md)。

## 5. Git 同步流程

主开发机：

```bash
git add <changed-files>
git commit -m "describe the change"
git push origin main
```

计算服务器首次使用 `git clone`，后续使用：

```bash
git pull --ff-only
```

提交前使用 `git status --ignored` 和 `git check-ignore -v <path>` 检查数据、权重和本地配置是否已被排除。
