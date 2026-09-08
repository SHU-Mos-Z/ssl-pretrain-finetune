"""NMF 预训练 Dataset：images + nmf_cache → Token batch 字段。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.masking.m_pix_expand import expand_token_mask_to_m_pix
from utils.masking.token3d_masker import MaskConfig, sample_token_mask
from utils.physics.beer_lambert import intensity_to_od_np, od_to_intensity_np
from utils.preprocessing.offline_nmf import cache_dir_name, load_intensity_cube
from utils.tokenization.spectral_groups import SpectralGroups
from utils.tokenization.token_builder import TokenBuildConfig, build_tokens


class NMFPretrainDataset(Dataset):
    """
    目录约定：
      <root>/images/*.npy
      <root>/<nmf_cache_dir>/{stem}_C.npy, {stem}_E.npy
    """

    def __init__(
        self,
        data_root: str,
        nmf_k: int = 2,
        nmf_l1: float = 1e-3,
        nmf_l2: float = 1e-4,
        nmf_l3: float = 1e-2,
        nmf_simplex: bool = False,
        nmf_lam_e: float = 0.0,
        nmf_e_clamp_max: float = 0.0,
        nmf_cache_dir: str | None = None,
        patch_size: int = 16,
        spectral_patch_size: int = 10,
        mask_cfg: MaskConfig | None = None,
        clamp_od_max: float | None = 3.0,
        epoch: int = 0,
        total_epochs: int = 200,
        seed: int = 0,
    ):
        self.data_root = data_root
        self.images_dir = os.path.join(data_root, "images")
        # nmf_cache_dir 优先；若未指定则根据全部 NMF 超参自动拼装目录名
        self.nmf_dir = nmf_cache_dir or os.path.join(
            data_root,
            cache_dir_name(
                nmf_k, nmf_l1, nmf_l2, nmf_l3,
                simplex=nmf_simplex,
                lam_e=nmf_lam_e,
                e_clamp_max=nmf_e_clamp_max,
            ),
        )
        self.token_cfg = TokenBuildConfig(patch_size, spectral_patch_size)
        self.mask_cfg = mask_cfg or MaskConfig()
        self.clamp_od_max = clamp_od_max
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.rng = np.random.default_rng(seed)

        if not os.path.isdir(self.images_dir):
            raise RuntimeError(f"images 目录不存在: {self.images_dir}")
        if not os.path.isdir(self.nmf_dir):
            raise RuntimeError(f"NMF 缓存不存在: {self.nmf_dir}")

        self.filenames = sorted(
            f for f in os.listdir(self.images_dir) if f.endswith(".npy")
        )
        valid = []
        for f in self.filenames:
            stem = Path(f).stem
            if os.path.isfile(os.path.join(self.nmf_dir, f"{stem}_C.npy")) and os.path.isfile(
                os.path.join(self.nmf_dir, f"{stem}_E.npy")
            ):
                valid.append(f)
        if not valid:
            raise RuntimeError(f"无配对 NMF 缓存: {self.nmf_dir}")
        self.filenames = valid

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        fname = self.filenames[idx]
        stem = Path(fname).stem

        intensity = load_intensity_cube(os.path.join(self.images_dir, fname))
        od = intensity_to_od_np(intensity).astype(np.float32)
        if self.clamp_od_max is not None:
            od = np.clip(od, 0.0, self.clamp_od_max)

        c_star = np.load(os.path.join(self.nmf_dir, f"{stem}_C.npy")).astype(np.float32)
        e_star = np.load(os.path.join(self.nmf_dir, f"{stem}_E.npy")).astype(np.float32)

        tokens = build_tokens(od, c_star, self.token_cfg)
        groups = SpectralGroups(tokens.s, tokens.n_sp)
        is_masked = sample_token_mask(
            od, self.token_cfg.patch_size, groups,
            self.epoch, self.total_epochs, self.mask_cfg,
            rng=self.rng,
        )
        m_pix = expand_token_mask_to_m_pix(
            is_masked, tokens.h, tokens.w, self.token_cfg.patch_size, groups,
        )
        i_gt = od_to_intensity_np(od).astype(np.float32)

        return {
            "token_raw": torch.from_numpy(tokens.token_raw),
            "is_masked": torch.from_numpy(is_masked),
            "pe_spatial": torch.from_numpy(tokens.pe_spatial),
            "pe_spectral": torch.from_numpy(tokens.pe_spectral),
            "pe_abund": torch.from_numpy(tokens.pe_abund),
            "od": torch.from_numpy(od),
            "intensity": torch.from_numpy(i_gt),
            "c_star": torch.from_numpy(c_star),
            "c_star_patch": torch.from_numpy(tokens.c_star_patch),
            "e_star": torch.from_numpy(e_star),
            "m_pix": torch.from_numpy(m_pix),
            "H": tokens.h,
            "W": tokens.w,
            "stem": stem,
        }


def set_dataset_epoch(loader, epoch: int) -> None:
    ds = loader.dataset
    if isinstance(ds, torch.utils.data.Subset):
        ds = ds.dataset
    if hasattr(ds, "datasets"):
        for d in ds.datasets:
            if isinstance(d, NMFPretrainDataset):
                d.set_epoch(epoch)
        return
    if isinstance(ds, NMFPretrainDataset):
        ds.set_epoch(epoch)


def _self_test() -> None:
    import shutil
    import sys
    import tempfile

    import torch
    from torch.utils.data import DataLoader

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from models.nmf_pretrain_model_vit import NMFPretrainModel
    from utils.datasets.nmf_pretrain_collate import nmf_pretrain_collate
    from utils.losses.nmf_pretext_loss import NMFPretextLoss
    from utils.preprocessing.offline_nmf import cache_dir_name, process_data_root

    data_candidates = [
        root / "data/MDC_Resized_256_320_to_256_256_overlap_0_0_preprocessed",
        root / "data/GPCC_Resized_512_640_to_256_256_overlap_0_0_preprocessed_val",
    ]
    data_root = next((p for p in data_candidates if (p / "images").is_dir()), None)
    if data_root is None:
        print("跳过 dataset 集成测试：未找到 data/*/images")
        return

    print(f"\n{'=' * 72}")
    print("  nmf_pretrain_dataset 集成自检")
    print(f"{'=' * 72}")

    tmp = Path(tempfile.mkdtemp(prefix="nmf_cache_test_"))
    try:
        mini_root = tmp / "mini"
        (mini_root / "images").mkdir(parents=True)
        for f in sorted((data_root / "images").glob("*.npy"))[:2]:
            shutil.copy(f, mini_root / "images" / f.name)

        k, l1, l2, l3 = 2, 1e-3, 1e-4, 1e-2
        process_data_root(mini_root, max_iter=80, verbose=False)
        cache_name = cache_dir_name(k, l1, l2, l3)

        ds = NMFPretrainDataset(
            str(mini_root),
            nmf_cache_dir=str(mini_root / cache_name),
            patch_size=16,
            spectral_patch_size=10,
            total_epochs=100,
            epoch=10,
        )
        loader = DataLoader(ds, batch_size=min(2, len(ds)), collate_fn=nmf_pretrain_collate)
        batch = next(iter(loader))

        print(f"  dataset size       {len(ds)}")
        print(f"  token_raw          {tuple(batch['token_raw'].shape)}")
        print(f"  m_pix              {tuple(batch['m_pix'].shape)}")
        print(f"  c_star_patch       {tuple(batch['c_star_patch'].shape)}")
        print(f"  e_star             {tuple(batch['e_star'].shape)}")

        model = NMFPretrainModel(vit_depth=2, num_spectral_groups=6, spectral_patch_size=10)
        criterion = NMFPretextLoss()
        model.train()
        out = model(batch)
        loss, logs = criterion(out, batch)
        loss.backward()
        print(f"  loss_total         {logs['loss_total']:.6f}")
        print("  backward           OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    _self_test()
