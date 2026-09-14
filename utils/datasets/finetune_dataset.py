"""微调 Dataset：OD + Token 字段 + 分割标注。"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from utils.physics.beer_lambert import intensity_to_od
from utils.preprocessing.offline_nmf import load_intensity_cube
from utils.tokenization.token_builder import TokenBuildConfig, build_tokens


class FinetuneDataset(Dataset):
    """
    返回 FinetuneModel 所需 token batch 字段 + seg label。

    目录：<root>/images/*.npy, <root>/masks/*.npy
    微调阶段默认不做 mask（is_masked 全 False）。

    include_od: 为 True 时在 batch 中额外返回 'od' (S,H,W) 张量，
                供 CINET 等需要完整 OD 立方的模型使用。
    """

    def __init__(
        self,
        data_root: str,
        patch_size: int = 16,
        spectral_patch_size: int = 10,
        clamp_od_max: float | None = 3.0,
        use_dummy_abund: bool = True,
        include_od: bool = False,
    ):
        self.images_dir = os.path.join(data_root, "images")
        self.masks_dir = os.path.join(data_root, "masks")
        self.token_cfg = TokenBuildConfig(patch_size, spectral_patch_size)
        self.clamp_od_max = clamp_od_max
        self.use_dummy_abund = use_dummy_abund
        self.include_od = include_od

        img_stems = {
            os.path.splitext(f)[0]
            for f in os.listdir(self.images_dir)
            if f.endswith(".npy")
        }
        mask_stems = {
            os.path.splitext(f)[0]
            for f in os.listdir(self.masks_dir)
            if f.endswith(".npy")
        }
        paired = sorted(img_stems & mask_stems)
        if not paired:
            raise RuntimeError(f"images/masks 无法配对: {data_root}")
        self.filenames = [s + ".npy" for s in paired]

    def __len__(self) -> int:
        return len(self.filenames)

    def class_pixel_counts(self, num_classes: int) -> torch.Tensor:
        """Count training-mask pixels for deterministic loss weighting."""
        counts = np.zeros(int(num_classes), dtype=np.int64)
        for fname in self.filenames:
            seg = np.load(os.path.join(self.masks_dir, fname))
            if seg.ndim == 3:
                seg = seg.squeeze(-1)
            valid = (seg >= 0) & (seg < num_classes)
            if np.any(valid):
                counts += np.bincount(
                    seg[valid].astype(np.int64, copy=False), minlength=num_classes
                )[:num_classes]
        return torch.from_numpy(counts)

    def __getitem__(self, idx: int) -> dict:
        fname = self.filenames[idx]
        intensity = load_intensity_cube(os.path.join(self.images_dir, fname))
        od_np = intensity
        # load_intensity -> (S,H,W); convert via torch for consistency
        i_t = torch.from_numpy(intensity)
        od = intensity_to_od(i_t)
        if self.clamp_od_max is not None:
            od = od.clamp(0.0, self.clamp_od_max)
        od_np = od.numpy().astype(np.float32)

        # 微调无 NMF 缓存时用均匀 dummy C* 仅生成 pe_abund（若 use_abund_pe=False 可忽略）
        k = 2
        c_dummy = np.ones((k, od_np.shape[1], od_np.shape[2]), dtype=np.float32) / k
        tokens = build_tokens(od_np, c_dummy, self.token_cfg)
        t = tokens.num_tokens
        is_masked = torch.zeros(t, dtype=torch.bool)

        seg = np.load(os.path.join(self.masks_dir, fname))
        if seg.ndim == 3:
            seg = seg.squeeze(-1)

        sample = {
            "token_raw": torch.from_numpy(tokens.token_raw),
            "is_masked": is_masked,
            "pe_spatial": torch.from_numpy(tokens.pe_spatial),
            "pe_spectral": torch.from_numpy(tokens.pe_spectral),
            "seg": torch.from_numpy(seg.astype(np.int64)),
            "H": tokens.h,
            "W": tokens.w,
        }
        if self.include_od:
            sample["od"] = torch.from_numpy(od_np)
        return sample


def finetune_collate(batch: list[dict]) -> dict:
    h, w = batch[0]["H"], batch[0]["W"]
    for s in batch[1:]:
        if s["H"] != h or s["W"] != w:
            raise ValueError("batch 内 H/W 须一致")
    out = {
        "token_raw": torch.stack([s["token_raw"] for s in batch]),
        "is_masked": torch.stack([s["is_masked"] for s in batch]),
        "pe_spatial": torch.stack([s["pe_spatial"] for s in batch]),
        "pe_spectral": torch.stack([s["pe_spectral"] for s in batch]),
        "seg": torch.stack([s["seg"] for s in batch]),
        "H": h,
        "W": w,
    }
    if "od" in batch[0]:
        out["od"] = torch.stack([s["od"] for s in batch])
    return out


def build_finetune_loaders(
    train_root: str,
    val_root: str,
    test_root: str | None = None,
    batch_size: int = 4,
    num_workers: int = 4,
    patch_size: int = 16,
    spectral_patch_size: int = 10,
    distributed: bool = False,
    include_od: bool = False,
):
    train_ds = FinetuneDataset(train_root, patch_size, spectral_patch_size, include_od=include_od)
    val_ds = FinetuneDataset(val_root, patch_size, spectral_patch_size, include_od=include_od)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=finetune_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=finetune_collate,
    )
    test_loader = None
    if test_root is not None:
        test_ds = FinetuneDataset(test_root, patch_size, spectral_patch_size, include_od=include_od)
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=finetune_collate,
        )
    return train_loader, val_loader, test_loader, train_sampler
