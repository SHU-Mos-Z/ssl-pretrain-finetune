"""NMF 预训练 DataLoader 构建。"""

from __future__ import annotations

import os

from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from utils.datasets.nmf_pretrain_collate import nmf_pretrain_collate
from utils.datasets.nmf_pretrain_dataset import NMFPretrainDataset, set_dataset_epoch
from utils.masking.token3d_masker import MaskConfig
from utils.datasets.conditioned_pretrain_collate import conditioned_pretrain_collate
from utils.datasets.conditioned_pretrain_dataset import (
    ConditionedPretrainDataset,
    set_conditioned_dataset_epoch,
)
from utils.datasets.homogeneous_batch_sampler import HomogeneousDistributedBatchSampler
from utils.datasets.conditioned_finetune_dataset import (
    ConditionedSlidingWindowSceneDataset,
    ConditionedSlidingWindowTestDataset,
    build_conditioned_finetune_loaders,
)
from utils.datasets.conditioned_detection_dataset import (
    ConditionedDetectionDataset,
    DistributedEvalSampler,
    build_conditioned_detection_loaders,
    conditioned_detection_collate,
)
from utils.datasets.detection_view_geometry import DetectionView, DetectionViewConfig


def build_pretrain_loader(
    data_roots: list[str],
    batch_size: int = 4,
    num_workers: int = 4,
    patch_size: int = 16,
    spectral_patch_size: int | None = None,
    mask_ratio: float = 0.4,
    use_gradient_masking: bool = True,
    sobel_tau: float = 1.0,
    spectral_alpha: float = 1.0,
    nmf_k: int = 2,
    nmf_l1: float = 1e-3,
    nmf_l2: float = 1e-4,
    nmf_l3: float = 1e-2,
    nmf_simplex: bool = False,
    nmf_lam_e: float = 0.0,
    nmf_e_clamp_max: float = 0.0,
    total_epochs: int = 200,
    distributed: bool = False,
    seed: int = 0,
):
    mask_cfg = MaskConfig(
        mask_ratio=mask_ratio,
        tau=sobel_tau,
        spectral_alpha=spectral_alpha,
        use_gradient=use_gradient_masking,
    )
    s_p = spectral_patch_size if spectral_patch_size is not None else 10
    datasets = [
        NMFPretrainDataset(
            root,
            nmf_k=nmf_k,
            nmf_l1=nmf_l1,
            nmf_l2=nmf_l2,
            nmf_l3=nmf_l3,
            nmf_simplex=nmf_simplex,
            nmf_lam_e=nmf_lam_e,
            nmf_e_clamp_max=nmf_e_clamp_max,
            patch_size=patch_size,
            spectral_patch_size=s_p,
            mask_cfg=mask_cfg,
            total_epochs=total_epochs,
            seed=seed + i,
        )
        for i, root in enumerate(data_roots)
        if os.path.isdir(root)
    ]
    if not datasets:
        raise RuntimeError(f"无有效 data_roots: {data_roots}")
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=nmf_pretrain_collate,
    )
    return loader, sampler


def build_conditioned_pretrain_loader(
    data_roots: list[str],
    batch_size: int = 4,
    num_workers: int = 4,
    distributed: bool = False,
    **dataset_kwargs,
):
    datasets = [
        ConditionedPretrainDataset(
            root,
            seed=dataset_kwargs.get("seed", 42) + index,
            **{k: v for k, v in dataset_kwargs.items() if k != "seed"},
        )
        for index, root in enumerate(data_roots)
        if os.path.isdir(root)
    ]
    if not datasets:
        raise RuntimeError(f"no valid data roots: {data_roots}")
    dataset = ConcatDataset(datasets)
    batch_sampler = HomogeneousDistributedBatchSampler(
        dataset, batch_size, shuffle=True, seed=dataset_kwargs.get("seed", 42)
    )
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=conditioned_pretrain_collate,
    )
    return loader, batch_sampler


__all__ = [
    "build_pretrain_loader",
    "set_dataset_epoch",
    "NMFPretrainDataset",
    "FinetuneDataset",
    "build_finetune_loaders",
    "ConditionedPretrainDataset",
    "build_conditioned_pretrain_loader",
    "set_conditioned_dataset_epoch",
    "build_conditioned_finetune_loaders",
    "ConditionedSlidingWindowSceneDataset",
    "ConditionedSlidingWindowTestDataset",
    "ConditionedDetectionDataset",
    "DistributedEvalSampler",
    "build_conditioned_detection_loaders",
    "conditioned_detection_collate",
    "DetectionView",
    "DetectionViewConfig",
]

from utils.datasets.finetune_dataset import (
    FinetuneDataset,
    build_finetune_loaders,
)  # noqa: E402
