"""Multi-dataset samples for endmember-conditioned pretraining."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from utils.masking.hybrid_masker import HybridMaskConfig, sample_hybrid_mask
from utils.physics.beer_lambert import intensity_to_od_np, od_to_intensity_np
from utils.preprocessing.offline_nmf import cache_dir_name, load_intensity_cube
from utils.tokenization.spectral_metadata import load_wavelengths, token_spectral_positions
from utils.tokenization.token_builder import TokenBuildConfig, build_tokens


class ConditionedPretrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root: str,
        nmf_k: int = 16,
        nmf_l1: float = 5e-4,
        nmf_l2: float = 2e-4,
        nmf_l3: float = 1e-2,
        nmf_simplex: bool = True,
        nmf_lam_e: float = 0.05,
        nmf_e_clamp_max: float = 3.0,
        nmf_cache_dir: str | None = None,
        wavelength_file: str | None = None,
        allow_index_wavelengths: bool = False,
        patch_size: int = 16,
        spectral_patch_size: int = 5,
        spectral_mask_ratio: float = 0.3,
        spatial_mask_ratio: float = 0.2,
        second_view: bool = False,
        permute_endmembers: bool = True,
        od_max: float = 3.0,
        nmf_weight_temperature: float = 0.05,
        pad_to_patch: bool = True,
        seed: int = 42,
    ):
        self.root = Path(data_root)
        self.images_dir = self.root / "images"
        self.nmf_dir = Path(nmf_cache_dir) if nmf_cache_dir else self.root / cache_dir_name(
            nmf_k, nmf_l1, nmf_l2, nmf_l3, nmf_simplex, nmf_lam_e, nmf_e_clamp_max
        )
        self.token_cfg = TokenBuildConfig(patch_size, spectral_patch_size)
        self.mask_cfg = HybridMaskConfig(spectral_mask_ratio, spatial_mask_ratio)
        self.wavelength_file = wavelength_file
        self.allow_index_wavelengths = allow_index_wavelengths
        self.second_view = second_view
        self.permute_endmembers = permute_endmembers
        self.od_max = od_max
        self.nmf_weight_temperature = nmf_weight_temperature
        self.pad_to_patch = pad_to_patch
        self.seed = seed
        self.epoch = 0
        files = sorted(self.images_dir.glob("*.npy"))
        self.files = [
            p for p in files
            if (self.nmf_dir / f"{p.stem}_C.npy").is_file()
            and (self.nmf_dir / f"{p.stem}_E.npy").is_file()
        ]
        if not self.files:
            raise RuntimeError(f"no image/NMF pairs under {self.root} and {self.nmf_dir}")
        first = load_intensity_cube(self.files[0])
        self.signature = self._signature(first)
        self.wavelengths = load_wavelengths(
            self.root, first.shape[0], self.wavelength_file, self.allow_index_wavelengths
        )

    def _signature(self, cube: np.ndarray) -> tuple[int, int, int, int]:
        s, h, w = cube.shape
        p = self.token_cfg.patch_size
        if self.pad_to_patch:
            h, w = math.ceil(h / p) * p, math.ceil(w / p) * p
        return s, h, w, int(np.load(self.nmf_dir / f"{self.files[0].stem}_E.npy", mmap_mode="r").shape[0])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.files)

    def _pad(self, x: np.ndarray, h: int, w: int, value: float = 0.0) -> np.ndarray:
        dh, dw = h - x.shape[-2], w - x.shape[-1]
        if dh == 0 and dw == 0:
            return x
        return np.pad(x, ((0, 0), (0, dh), (0, dw)), constant_values=value)

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        intensity = load_intensity_cube(path)
        od = np.clip(intensity_to_od_np(intensity), 0.0, self.od_max).astype(np.float32)
        c_star = np.load(self.nmf_dir / f"{path.stem}_C.npy").astype(np.float32)
        e_star = np.load(self.nmf_dir / f"{path.stem}_E.npy").astype(np.float32)
        s, h0, w0 = od.shape
        p = self.token_cfg.patch_size
        h, w = math.ceil(h0 / p) * p, math.ceil(w0 / p) * p
        if (h, w) != (h0, w0) and not self.pad_to_patch:
            raise ValueError(f"{path} shape {(h0,w0)} is not divisible by patch_size={p}")
        valid = np.zeros((1, h, w), dtype=np.bool_)
        valid[:, :h0, :w0] = True
        od = self._pad(od, h, w)
        c_star = self._pad(c_star, h, w, 1.0 / c_star.shape[0])

        if self.wavelengths.size != s:
            raise ValueError(f"inconsistent band count in {path}: {s} vs {self.wavelengths.size}")
        wavelengths = self.wavelengths
        tokens = build_tokens(od, c_star, self.token_cfg)
        tokens.pe_spectral[:] = token_spectral_positions(
            wavelengths, tokens.h_p, tokens.w_p, self.token_cfg.spectral_patch_size
        )
        teacher_od = np.einsum("khw,ks->shw", c_star, e_star)
        residual = np.mean((teacher_od - od) ** 2, axis=0, keepdims=True)
        w_nmf = np.exp(-residual / max(self.nmf_weight_temperature, 1e-8)).astype(np.float32)
        w_nmf *= valid

        generator = torch.Generator().manual_seed(self.seed + self.epoch * len(self) + idx)
        mask_a = sample_hybrid_mask(
            1, tokens.h_p, tokens.w_p, tokens.n_sp, p, tokens.s_p,
            self.mask_cfg, generator=generator, num_bands=s,
        )
        sample = {
            "od": torch.from_numpy(od),
            "intensity": torch.from_numpy(od_to_intensity_np(od).astype(np.float32)),
            "e_star": torch.from_numpy(e_star),
            "c_star": torch.from_numpy(c_star),
            "wavelengths": torch.from_numpy(wavelengths),
            "token_raw": torch.from_numpy(tokens.token_raw),
            "token_visible": mask_a.token_visible[0],
            "voxel_visible": mask_a.voxel_visible[0] & torch.from_numpy(valid),
            "valid_voxel": torch.from_numpy(valid),
            "w_nmf": torch.from_numpy(w_nmf),
            "pe_spatial": torch.from_numpy(tokens.pe_spatial),
            "pe_spectral": torch.from_numpy(tokens.pe_spectral),
            "stem": path.stem,
            "dataset_id": self.root.name,
        }
        if self.second_view:
            mask_b = sample_hybrid_mask(
                1, tokens.h_p, tokens.w_p, tokens.n_sp, p, tokens.s_p,
                self.mask_cfg, generator=generator, num_bands=s,
            )
            sample["token_visible_b"] = mask_b.token_visible[0]
            sample["voxel_visible_b"] = mask_b.voxel_visible[0] & torch.from_numpy(valid)
        if self.permute_endmembers:
            permutation = torch.randperm(e_star.shape[0], generator=generator)
            sample["e_star"] = sample["e_star"][permutation]
            sample["c_star"] = sample["c_star"][permutation]
        return sample


def set_conditioned_dataset_epoch(loader, epoch: int) -> None:
    datasets = getattr(loader.dataset, "datasets", [loader.dataset])
    for dataset in datasets:
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
