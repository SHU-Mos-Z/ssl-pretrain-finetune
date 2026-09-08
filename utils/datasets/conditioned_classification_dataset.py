"""Dataset and loaders for conditioned patch-level HSI classification."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from utils.augmentations.hsi_spatial import (
    PERSPECTIVE_PADDING_MODES,
    random_four_point_perspective_od_and_abundance,
    should_apply_perspective,
)
from utils.physics.beer_lambert import intensity_to_od_np
from utils.preprocessing.offline_nmf import cache_dir_name, load_intensity_cube
from utils.sample_exclusion import load_excluded_samples
from utils.tokenization.band_padding import band_pad_amounts, pad_bands
from utils.tokenization.spectral_metadata import load_wavelengths, token_spectral_positions
from utils.tokenization.token_builder import TokenBuildConfig, build_tokens


CLASSIFICATION_AUGMENTATION_POLICIES = (
    "dihedral",
    "perspective",
    "dihedral_perspective",
)


@dataclass(frozen=True)
class ClassificationSample:
    class_name: str
    label: int
    stem: str
    image_path: Path
    endmember_path: Path


def load_class_map(path: str | Path) -> dict[str, int]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        mapping = {str(name): index for index, name in enumerate(raw)}
    elif isinstance(raw, dict):
        mapping = {str(name): int(index) for name, index in raw.items()}
    else:
        raise TypeError("class map must be a JSON object or an ordered JSON list")
    values = sorted(mapping.values())
    if values != list(range(len(mapping))):
        raise ValueError("class indices must be contiguous and start at zero")
    return mapping


class ConditionedClassificationDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        *,
        class_to_idx: dict[str, int] | None = None,
        class_map_file: str | None = None,
        patch_size: int = 16,
        spectral_patch_size: int = 5,
        nmf_k: int = 16,
        nmf_l1: float = 5e-4,
        nmf_l2: float = 2e-4,
        nmf_l3: float = 1e-2,
        nmf_simplex: bool = True,
        nmf_lam_e: float = 0.05,
        nmf_e_clamp_max: float = 3.0,
        wavelength_file: str | None = None,
        allow_index_wavelengths: bool = False,
        od_max: float = 3.0,
        augment: bool = False,
        augmentation_copies: int = 1,
        augmentation_seed: int = 42,
        augmentation_policy: str = "dihedral",
        perspective_probability: float = 0.5,
        perspective_scale: float = 0.05,
        perspective_padding_mode: str = "reflection",
        exclude_samples_file: str | None = None,
    ):
        self.root = Path(data_root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"classification root does not exist: {self.root}")
        if class_map_file and class_to_idx is not None:
            raise ValueError("provide class_map_file or class_to_idx, not both")
        if class_map_file:
            class_to_idx = load_class_map(class_map_file)
        if class_to_idx is None:
            names = sorted(
                path.name
                for path in self.root.iterdir()
                if path.is_dir() and (path / "images").is_dir()
            )
            class_to_idx = {name: index for index, name in enumerate(names)}
        self.class_to_idx = dict(class_to_idx)
        self.idx_to_class = {
            index: name for name, index in self.class_to_idx.items()
        }
        if len(self.class_to_idx) < 2:
            raise RuntimeError(f"at least two classes are required under {self.root}")

        self.token_config = TokenBuildConfig(patch_size, spectral_patch_size)
        self.od_max = float(od_max)
        self.augment = bool(augment)
        self.augmentation_copies = int(augmentation_copies)
        self.augmentation_seed = int(augmentation_seed)
        self.augmentation_policy = str(augmentation_policy)
        self.perspective_probability = float(perspective_probability)
        self.perspective_scale = float(perspective_scale)
        self.perspective_padding_mode = str(perspective_padding_mode)
        self.exclude_samples_file = (
            str(Path(exclude_samples_file)) if exclude_samples_file else None
        )
        excluded_identities = (
            load_excluded_samples(exclude_samples_file)
            if exclude_samples_file
            else set()
        )
        if not 1 <= self.augmentation_copies <= 8:
            raise ValueError("augmentation_copies must be in [1,8]")
        if not self.augment and self.augmentation_copies != 1:
            raise ValueError(
                "augmentation_copies must be 1 when augment=False"
            )
        if self.augmentation_policy not in CLASSIFICATION_AUGMENTATION_POLICIES:
            raise ValueError(
                "augmentation_policy must be one of "
                f"{CLASSIFICATION_AUGMENTATION_POLICIES}"
            )
        if not 0.0 <= self.perspective_probability <= 1.0:
            raise ValueError("perspective_probability must be in [0,1]")
        if not 0.0 <= self.perspective_scale < 0.5:
            raise ValueError("perspective_scale must be in [0,0.5)")
        if self.perspective_padding_mode not in PERSPECTIVE_PADDING_MODES:
            raise ValueError(
                "perspective_padding_mode must be one of "
                f"{PERSPECTIVE_PADDING_MODES}"
            )
        # persistent_workers=True 时，各 worker 持有独立的 Dataset 对象；共享
        # Tensor 使训练主进程设置的 epoch 能立即被 worker 读取。
        self._shared_epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        nmf_name = cache_dir_name(
            nmf_k,
            nmf_l1,
            nmf_l2,
            nmf_l3,
            nmf_simplex,
            nmf_lam_e,
            nmf_e_clamp_max,
        )

        samples: list[ClassificationSample] = []
        excluded_counts = {name: 0 for name in self.class_to_idx}
        matched_excluded: set[tuple[str, str]] = set()
        usable_before_exclusion = 0
        band_count: int | None = None
        spatial_shape: tuple[int, int] | None = None
        for class_name, label in sorted(
            self.class_to_idx.items(), key=lambda item: item[1]
        ):
            class_root = self.root / class_name
            images_dir = class_root / "images"
            nmf_dir = class_root / nmf_name
            if not images_dir.is_dir():
                raise FileNotFoundError(
                    f"class '{class_name}' is missing images directory: {images_dir}"
                )
            stems = sorted(path.stem for path in images_dir.glob("*.npy"))
            usable = [stem for stem in stems if (nmf_dir / f"{stem}_E.npy").is_file()]
            usable_before_exclusion += len(usable)
            retained: list[str] = []
            for stem in usable:
                identity = (class_name, stem)
                if identity in excluded_identities:
                    matched_excluded.add(identity)
                    excluded_counts[class_name] += 1
                else:
                    retained.append(stem)
            usable = retained
            if not usable:
                raise RuntimeError(
                    f"class '{class_name}' has no retained image/endmember pairs in "
                    f"{self.root} after sample exclusion"
                )
            for stem in usable:
                image_path = images_dir / f"{stem}.npy"
                if band_count is None:
                    cube = load_intensity_cube(image_path)
                    band_count = int(cube.shape[0])
                    spatial_shape = (int(cube.shape[1]), int(cube.shape[2]))
                samples.append(
                    ClassificationSample(
                        class_name,
                        label,
                        stem,
                        image_path,
                        nmf_dir / f"{stem}_E.npy",
                    )
                )

        assert band_count is not None and spatial_shape is not None
        if spatial_shape[0] % patch_size or spatial_shape[1] % patch_size:
            raise ValueError(
                f"classification image {spatial_shape} must be divisible by patch_size={patch_size}"
            )
        raw_wavelengths = load_wavelengths(
            self.root,
            band_count,
            wavelength_file,
            allow_index_wavelengths,
        )
        front, back = band_pad_amounts(band_count, spectral_patch_size)
        if front or back:
            warnings.warn(
                f"{self.root}: band count S={band_count} is not divisible by "
                f"spectral_patch_size={spectral_patch_size}; mirror-padding by "
                f"duplicating band 0 {front} time(s) at the low-wavelength edge "
                f"and the last band {back} time(s) at the high-wavelength edge "
                f"(alternating) to reach S={band_count + front + back}.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.wavelengths = pad_bands(raw_wavelengths, spectral_patch_size, axis=0)
        self.raw_band_count = band_count
        self.band_count = int(self.wavelengths.size)
        self.spatial_shape = spatial_shape
        self.samples = samples
        self.exclusion_summary = {
            "file": self.exclude_samples_file,
            "requested_unique_samples": len(excluded_identities),
            "matched_training_samples": len(matched_excluded),
            "unmatched_samples": len(excluded_identities - matched_excluded),
            "samples_before_exclusion": usable_before_exclusion,
            "samples_after_exclusion": len(samples),
            "excluded_by_class": excluded_counts,
        }

    def __len__(self) -> int:
        copies = self.augmentation_copies if self.augment else 1
        return len(self.samples) * copies

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    def class_counts(self) -> torch.Tensor:
        labels = torch.tensor([sample.label for sample in self.samples])
        counts = torch.bincount(labels, minlength=self.num_classes)
        copies = self.augmentation_copies if self.augment else 1
        return counts * copies

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to choose deterministic, distinct transforms."""
        self._shared_epoch.fill_(int(epoch))

    def _transform_ids_for(self, base_index: int) -> list[int]:
        """Return this sample's distinct transform IDs for the current epoch."""
        epoch = int(self._shared_epoch.item())
        generator = torch.Generator().manual_seed(
            self.augmentation_seed + 1_000_003 * epoch + base_index
        )
        return torch.randperm(8, generator=generator)[
            : self.augmentation_copies
        ].tolist()

    def _perspective_applied_for(self, base_index: int, copy_index: int) -> bool:
        if not self.augment or self.augmentation_policy == "dihedral":
            return False
        return should_apply_perspective(
            self.perspective_probability,
            base_seed=self.augmentation_seed,
            epoch=int(self._shared_epoch.item()),
            sample_index=base_index,
            copy_index=copy_index,
        )

    @staticmethod
    def _spatial_augment(cube: np.ndarray, transform: int) -> np.ndarray:
        if not 0 <= transform < 8:
            raise ValueError("spatial transform ID must be in [0,7]")
        rotated = np.rot90(cube, k=transform % 4, axes=(-2, -1))
        if transform >= 4:
            rotated = np.flip(rotated, axis=-1)
        return np.ascontiguousarray(rotated)

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.augment:
            base_index, copy_index = divmod(index, self.augmentation_copies)
            transform = (
                self._transform_ids_for(base_index)[copy_index]
                if self.augmentation_policy in {"dihedral", "dihedral_perspective"}
                else 0
            )
        else:
            base_index, copy_index, transform = index, 0, 0
        sample = self.samples[base_index]
        intensity = load_intensity_cube(sample.image_path)
        if intensity.shape != (self.raw_band_count, *self.spatial_shape):
            raise ValueError(
                f"inconsistent cube shape for {sample.image_path}: {intensity.shape}"
            )
        if self.augment and self.augmentation_policy in {
            "dihedral",
            "dihedral_perspective",
        }:
            intensity = self._spatial_augment(intensity, transform)
        intensity = np.ascontiguousarray(intensity.astype(np.float32, copy=False))
        intensity = pad_bands(intensity, self.token_config.spectral_patch_size, axis=0)
        od = np.clip(intensity_to_od_np(intensity), 0, self.od_max).astype(np.float32)
        perspective_applied = self._perspective_applied_for(base_index, copy_index)
        perspective_seed = -1
        if perspective_applied:
            perspective = random_four_point_perspective_od_and_abundance(
                od,
                scale=self.perspective_scale,
                base_seed=self.augmentation_seed,
                epoch=int(self._shared_epoch.item()),
                sample_index=base_index,
                copy_index=copy_index,
                od_max=self.od_max,
                padding_mode=self.perspective_padding_mode,
            )
            od = np.ascontiguousarray(
                perspective["od"].detach().cpu().numpy().astype(np.float32, copy=False)
            )
            intensity = np.ascontiguousarray(
                perspective["intensity"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            perspective_seed = int(perspective["parameters"].seed)
        endmembers = np.load(sample.endmember_path).astype(np.float32)
        if endmembers.ndim != 2 or endmembers.shape[1] != self.raw_band_count:
            raise ValueError(
                f"endmember shape {endmembers.shape} is incompatible with raw S={self.raw_band_count}"
            )
        endmembers = pad_bands(endmembers, self.token_config.spectral_patch_size, axis=1)

        _, height, width = od.shape
        dummy_abundance = np.full(
            (endmembers.shape[0], height, width),
            1.0 / endmembers.shape[0],
            dtype=np.float32,
        )
        tokens = build_tokens(od, dummy_abundance, self.token_config)
        tokens.pe_spectral[:] = token_spectral_positions(
            self.wavelengths, tokens.h_p, tokens.w_p, tokens.s_p
        )
        token_visible = np.ones(
            (tokens.h_p, tokens.w_p, tokens.n_sp), dtype=np.bool_
        )
        voxel_visible = np.ones_like(od, dtype=np.bool_)
        return {
            "od": torch.from_numpy(od),
            "intensity": torch.from_numpy(intensity),
            "e_star": torch.from_numpy(endmembers),
            "wavelengths": torch.from_numpy(self.wavelengths.copy()),
            "token_raw": torch.from_numpy(tokens.token_raw),
            "token_visible": torch.from_numpy(token_visible),
            "voxel_visible": torch.from_numpy(voxel_visible),
            "pe_spatial": torch.from_numpy(tokens.pe_spatial),
            "pe_spectral": torch.from_numpy(tokens.pe_spectral),
            "label": torch.tensor(sample.label, dtype=torch.long),
            "sample_index": torch.tensor(index, dtype=torch.long),
            "base_sample_index": torch.tensor(base_index, dtype=torch.long),
            "augmentation_copy_index": torch.tensor(copy_index, dtype=torch.long),
            "augmentation_transform_id": torch.tensor(transform, dtype=torch.long),
            "augmentation_perspective_applied": torch.tensor(
                perspective_applied, dtype=torch.bool
            ),
            "augmentation_perspective_seed": torch.tensor(
                perspective_seed, dtype=torch.long
            ),
            "stem": sample.stem,
            "class_name": sample.class_name,
        }


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation data without DistributedSampler's duplicate padding."""

    def __init__(self, dataset: Dataset, num_replicas: int, rank: int):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        return len(range(self.rank, len(self.dataset), self.num_replicas))


class EpochWeightedRandomSampler(Sampler[int]):
    """Deterministic epoch-wise weighted sampling, optionally sharded by rank."""

    def __init__(
        self,
        weights: torch.Tensor,
        num_samples: int,
        seed: int = 42,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        if weights.ndim != 1 or len(weights) == 0:
            raise ValueError("balanced-sampling weights must be a non-empty vector")
        if num_samples <= 0:
            raise ValueError("balanced-sampling num_samples must be positive")
        if num_replicas <= 0:
            raise ValueError("balanced-sampling num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError("balanced-sampling rank is out of range")
        self.weights = weights.to(dtype=torch.double, device="cpu")
        self.dataset_size = int(num_samples)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.num_samples = (self.dataset_size + self.num_replicas - 1) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=generator,
        )
        rank_indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(rank_indices) != self.num_samples:
            raise RuntimeError("balanced sampler produced an invalid rank shard")
        return iter(rank_indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


def classification_collate(samples: list[dict]) -> dict:
    if not samples:
        raise ValueError("cannot collate an empty classification batch")
    output = {
        key: torch.stack([sample[key] for sample in samples])
        for key, value in samples[0].items()
        if isinstance(value, torch.Tensor)
    }
    output["stem"] = [sample["stem"] for sample in samples]
    output["class_name"] = [sample["class_name"] for sample in samples]
    return output


def build_conditioned_classification_loaders(
    train_root: str,
    val_root: str,
    test_root: str | None,
    batch_size: int = 4,
    num_workers: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    sampling_strategy: str = "standard",
    sampling_seed: int = 42,
    train_exclude_samples_file: str | None = None,
    **dataset_kwargs,
):
    if sampling_strategy not in {"standard", "balanced"}:
        raise ValueError("sampling_strategy must be 'standard' or 'balanced'")
    train_dataset = ConditionedClassificationDataset(
        train_root,
        exclude_samples_file=train_exclude_samples_file,
        **dataset_kwargs,
    )
    shared_kwargs = dict(dataset_kwargs)
    shared_kwargs.pop("class_map_file", None)
    shared_kwargs["class_to_idx"] = train_dataset.class_to_idx
    shared_kwargs["augment"] = False
    shared_kwargs["augmentation_copies"] = 1
    val_dataset = ConditionedClassificationDataset(val_root, **shared_kwargs)
    test_dataset = (
        ConditionedClassificationDataset(test_root, **shared_kwargs)
        if test_root
        else None
    )

    if sampling_strategy == "balanced":
        copies = train_dataset.augmentation_copies if train_dataset.augment else 1
        base_labels = torch.tensor(
            [sample.label for sample in train_dataset.samples], dtype=torch.long
        )
        labels = base_labels.repeat_interleave(copies)
        if len(labels) != len(train_dataset):
            raise RuntimeError("balanced-sampling labels do not match dataset length")
        class_counts = torch.bincount(
            labels, minlength=train_dataset.num_classes
        ).clamp_min(1)
        weights = class_counts.to(torch.double).reciprocal()[labels]
        train_sampler = EpochWeightedRandomSampler(
            weights,
            len(train_dataset),
            sampling_seed,
            num_replicas=world_size if distributed else 1,
            rank=rank if distributed else 0,
        )
    else:
        train_sampler = (
            DistributedSampler(train_dataset, shuffle=True) if distributed else None
        )
    val_sampler = (
        DistributedEvalSampler(val_dataset, world_size, rank) if distributed else None
    )
    test_sampler = (
        DistributedEvalSampler(test_dataset, world_size, rank)
        if distributed and test_dataset is not None
        else None
    )
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=classification_collate,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, sampler=val_sampler, drop_last=False, **common
    )
    test_loader = (
        DataLoader(
            test_dataset,
            shuffle=False,
            sampler=test_sampler,
            drop_last=False,
            **common,
        )
        if test_dataset is not None
        else None
    )
    return train_loader, val_loader, test_loader, train_sampler, train_dataset.class_to_idx
