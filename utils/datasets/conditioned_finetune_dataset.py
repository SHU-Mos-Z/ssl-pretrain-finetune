from __future__ import annotations

import csv
import warnings
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from utils.physics.beer_lambert import intensity_to_od_np
from utils.preprocessing.offline_nmf import cache_dir_name, load_intensity_cube
from utils.tokenization.band_padding import band_pad_amounts, pad_bands
from utils.tokenization.patch_tokens import compute_patch_token_raw, grid_sizes
from utils.tokenization.positional_targets import compute_pe_spatial
from utils.tokenization.spectral_metadata import load_wavelengths, token_spectral_positions
from utils.tokenization.spectral_groups import SpectralGroups
from utils.tokenization.token_builder import TokenBuildConfig
from utils.metrics import infer_segmentation_scene_id
from utils.augmentations.segmentation_spatial import (
    SEGMENTATION_AUGMENTATION_POLICIES,
    augment_hsi_segmentation_pair,
)


MODEL_INPUT_KEYS = (
    'od', 'intensity', 'e_star', 'wavelengths', 'token_raw',
    'token_visible', 'voxel_visible', 'pe_spatial', 'pe_spectral',
)


def build_conditioned_model_inputs(
    intensity: np.ndarray,
    e_star: np.ndarray,
    wavelengths: np.ndarray,
    cfg: TokenBuildConfig,
    od_max: float,
) -> dict[str, torch.Tensor]:
    """Build the conditioned model input for one spatial image/tile.

    ``intensity`` uses ``(S,H,W)`` and ``e_star`` uses ``(K,S)`` before
    spectral edge padding.  This is shared by the ordinary patch Dataset and
    full-scene sliding-window inference so their numerical preprocessing stays
    identical.
    """
    intensity = np.asarray(intensity, dtype=np.float32)
    e_star = np.asarray(e_star, dtype=np.float32)
    wavelengths = np.asarray(wavelengths, dtype=np.float32).reshape(-1)
    raw_band_count = int(intensity.shape[0])
    if intensity.ndim != 3:
        raise ValueError(f'intensity must have shape (S,H,W), got {intensity.shape}')
    if e_star.ndim != 2 or e_star.shape[1] != raw_band_count:
        raise ValueError(
            f'endmember shape {e_star.shape} is incompatible with S={raw_band_count}'
        )
    if wavelengths.size != raw_band_count:
        raise ValueError(
            f'wavelength count {wavelengths.size} is incompatible with S={raw_band_count}'
        )

    # Copy intensity even when no spectral padding is needed: mmap slices are
    # read-only and torch.from_numpy must not expose a non-writable buffer.
    intensity = pad_bands(intensity, cfg.spectral_patch_size, axis=0).copy()
    e_star = pad_bands(e_star, cfg.spectral_patch_size, axis=1).copy()
    wavelengths = pad_bands(wavelengths, cfg.spectral_patch_size, axis=0).copy()
    od = np.clip(intensity_to_od_np(intensity), 0, od_max).astype(np.float32)
    s, h, w = od.shape
    if h % cfg.patch_size or w % cfg.patch_size:
        raise ValueError(
            f'conditioned input {(h, w)} must be divisible by patch size {cfg.patch_size}'
        )
    # Fine-tuning consumes only raw OD tokens and their positions.  The generic
    # pre-training token builder also constructs dummy abundance positions and
    # dense abundance targets, neither of which is returned here.  Build only
    # the model inputs that are actually used, with the same ordering/formulae.
    n_sp = cfg.num_groups_for(s)
    groups = SpectralGroups(s, n_sp)
    h_p, w_p = grid_sizes(h, w, cfg.patch_size)
    token_raw = compute_patch_token_raw(od, cfg.patch_size, groups)
    pe_spatial = compute_pe_spatial(h, w, cfg.patch_size, groups)
    pe_spectral = token_spectral_positions(
        wavelengths, h_p, w_p, groups.patch_size
    )
    visible = np.ones((h_p, w_p, n_sp), np.bool_)
    voxel = np.ones((s, h, w), np.bool_)
    return {
        'od': torch.from_numpy(od),
        'intensity': torch.from_numpy(intensity),
        'e_star': torch.from_numpy(e_star),
        'wavelengths': torch.from_numpy(wavelengths),
        'token_raw': torch.from_numpy(token_raw),
        'token_visible': torch.from_numpy(visible),
        'voxel_visible': torch.from_numpy(voxel),
        'pe_spatial': torch.from_numpy(pe_spatial),
        'pe_spectral': torch.from_numpy(pe_spectral),
    }


def collate_conditioned_model_inputs(
    samples: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError('cannot collate an empty conditioned tile batch')
    return {key: torch.stack([sample[key] for sample in samples]) for key in MODEL_INPUT_KEYS}


def _open_intensity_memmap(path: Path) -> tuple[np.ndarray, bool]:
    """Return a read-only cube and whether its disk layout is CHW."""
    array = np.load(path, mmap_mode='r')
    if array.ndim != 3:
        raise ValueError(f'expected 3D cube, got shape {array.shape} from {path}')
    channel_first = bool(array.shape[0] <= 64 and array.shape[0] < array.shape[1])
    return array, channel_first


class ConditionedSlidingWindowSceneDataset:
    """Lightweight full-scene source used by sliding validation/test inference.

    Images remain memory-mapped on disk.  A scene-level NMF endmember matrix is
    loaded once, then reused for every spatial tile; the full abundance map is
    deliberately not read because it is not a model input during fine-tuning.
    """

    is_sliding_window_source = True

    def __init__(self, data_root: str, patch_size=16,
                 spectral_patch_size=5, nmf_k=16, nmf_l1=5e-4, nmf_l2=2e-4,
                 nmf_l3=1e-2, nmf_simplex=True, nmf_lam_e=.05,
                 nmf_e_clamp_max=3.0, nmf_cache_dir=None,
                 wavelength_file=None, allow_index_wavelengths=False, od_max=3.0,
                 endmember_scope='patch', scene_endmember_root=None):
        self.root = Path(data_root)
        if endmember_scope not in {'patch', 'scene'}:
            raise ValueError("endmember_scope must be 'patch' or 'scene'")
        self.endmember_scope = endmember_scope
        if endmember_scope == 'scene' and scene_endmember_root:
            self.nmf_dir = Path(scene_endmember_root)
        else:
            self.nmf_dir = Path(nmf_cache_dir) if nmf_cache_dir else self.root / cache_dir_name(
                nmf_k, nmf_l1, nmf_l2, nmf_l3, nmf_simplex, nmf_lam_e, nmf_e_clamp_max
            )
        self.cfg = TokenBuildConfig(patch_size, spectral_patch_size)
        self.od_max = float(od_max)
        image_stems = {p.stem for p in (self.root / 'images').glob('*.npy')}
        mask_stems = {p.stem for p in (self.root / 'masks').glob('*.npy')}
        paired_stems = sorted(image_stems & mask_stems)
        self.stems = sorted(
            stem for stem in image_stems & mask_stems
            if (self.nmf_dir / f'{stem}_E.npy').is_file()
        )
        if self.endmember_scope == 'scene' and len(self.stems) != len(paired_stems):
            missing = [
                stem for stem in paired_stems
                if not (self.nmf_dir / f'{stem}_E.npy').is_file()
            ]
            raise FileNotFoundError(
                f'{len(missing)} full scenes have no scene-level endmember cache under '
                f'{self.nmf_dir}; first missing scenes: {missing[:10]}'
            )
        if not self.stems:
            raise RuntimeError(f'no full-scene image/mask/E pairs for {self.root}')
        first, first_chw = _open_intensity_memmap(
            self.root / 'images' / f'{self.stems[0]}.npy'
        )
        self.raw_band_count = int(first.shape[0] if first_chw else first.shape[2])
        self.wavelengths = load_wavelengths(
            self.root, self.raw_band_count, wavelength_file, allow_index_wavelengths
        )

    def __len__(self) -> int:
        return len(self.stems)

    def distributed_indices(self, max_scenes: int = 0) -> list[int]:
        """Exact non-padding DDP shard, avoiding duplicated test scenes."""
        indices = list(range(len(self)))
        if max_scenes > 0:
            indices = indices[:max_scenes]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            indices = indices[rank::world_size]
        return indices

    def load_scene(self, index: int) -> dict[str, object]:
        stem = self.stems[index]
        image_path = self.root / 'images' / f'{stem}.npy'
        cube, channel_first = _open_intensity_memmap(image_path)
        band_count = int(cube.shape[0] if channel_first else cube.shape[2])
        if band_count != self.raw_band_count:
            raise ValueError(
                f'inconsistent band count for {stem}: {band_count} vs {self.raw_band_count}'
            )
        height, width = (
            (int(cube.shape[1]), int(cube.shape[2]))
            if channel_first else (int(cube.shape[0]), int(cube.shape[1]))
        )
        mask = np.load(self.root / 'masks' / f'{stem}.npy').squeeze().astype(np.int64)
        if mask.shape != (height, width):
            raise ValueError(
                f'image/mask shape mismatch for {stem}: {(height, width)} vs {mask.shape}'
            )
        e_star = np.load(self.nmf_dir / f'{stem}_E.npy').astype(np.float32)
        if e_star.shape[1] != self.raw_band_count:
            raise ValueError(
                f'endmember shape {e_star.shape} is incompatible with S={self.raw_band_count}'
            )
        return {
            'stem': stem,
            'scene_id': infer_segmentation_scene_id(stem),
            'cube': cube,
            'channel_first': channel_first,
            'mask': mask,
            'e_star': e_star,
            'height': height,
            'width': width,
        }

    @staticmethod
    def read_tile(
        scene: dict[str, object], y: int, x: int, height: int, width: int
    ) -> np.ndarray:
        cube = scene['cube']
        if bool(scene['channel_first']):
            tile = cube[:, y:y + height, x:x + width]
        else:
            tile = cube[y:y + height, x:x + width, :].transpose(2, 0, 1)
        return np.asarray(tile, dtype=np.float32)

    def build_tile_inputs(self, intensity: np.ndarray, e_star: np.ndarray) -> dict[str, torch.Tensor]:
        return build_conditioned_model_inputs(
            intensity, e_star, self.wavelengths, self.cfg, self.od_max
        )


# Backwards-compatible public name used by existing test-only call sites.
ConditionedSlidingWindowTestDataset = ConditionedSlidingWindowSceneDataset


class ConditionedFinetuneDataset(Dataset):
    def __init__(self, data_root: str, patch_size=16,
                 spectral_patch_size=5, nmf_k=16, nmf_l1=5e-4, nmf_l2=2e-4,
                 nmf_l3=1e-2, nmf_simplex=True, nmf_lam_e=.05,
                 nmf_e_clamp_max=3.0, nmf_cache_dir=None,
                 wavelength_file=None, allow_index_wavelengths=False, od_max=3.0,
                 augment=False, augmentation_copies=1, augmentation_seed=42,
                 augmentation_policy='dihedral', augmentation_probability=1.0,
                 affine_rotation_degrees=15.0, affine_scale_delta=0.1,
                 affine_translate_fraction=0.05, perspective_scale=0.05,
                 augmentation_padding_mode='reflection',
                 endmember_scope='patch', scene_endmember_root=None):
        self.root = Path(data_root)
        self.nmf_dir = Path(nmf_cache_dir) if nmf_cache_dir else self.root / cache_dir_name(
            nmf_k,nmf_l1,nmf_l2,nmf_l3,nmf_simplex,nmf_lam_e,nmf_e_clamp_max)
        self.cfg = TokenBuildConfig(patch_size,spectral_patch_size)
        self.allow_index_wavelengths, self.od_max = allow_index_wavelengths, od_max
        self.augment = bool(augment)
        self.augmentation_copies = int(augmentation_copies)
        self.augmentation_seed = int(augmentation_seed)
        self.augmentation_policy = str(augmentation_policy)
        self.augmentation_probability = float(augmentation_probability)
        self.affine_rotation_degrees = float(affine_rotation_degrees)
        self.affine_scale_delta = float(affine_scale_delta)
        self.affine_translate_fraction = float(affine_translate_fraction)
        self.perspective_scale = float(perspective_scale)
        self.augmentation_padding_mode = str(augmentation_padding_mode)
        self.endmember_scope = str(endmember_scope)
        self.scene_endmember_root = (
            Path(scene_endmember_root) if scene_endmember_root else None
        )
        if not 1 <= self.augmentation_copies <= 8:
            raise ValueError('augmentation_copies must be in [1,8]')
        if not self.augment and self.augmentation_copies != 1:
            raise ValueError('augmentation_copies must be 1 when augment=False')
        if self.augmentation_policy not in SEGMENTATION_AUGMENTATION_POLICIES:
            raise ValueError(
                f'augmentation_policy must be one of {SEGMENTATION_AUGMENTATION_POLICIES}'
            )
        if not 0.0 <= self.augmentation_probability <= 1.0:
            raise ValueError('augmentation_probability must be in [0,1]')
        if self.endmember_scope not in {'patch', 'scene'}:
            raise ValueError("endmember_scope must be 'patch' or 'scene'")
        if self.endmember_scope == 'scene' and self.scene_endmember_root is None:
            raise ValueError('scene_endmember_root is required for scene endmembers')
        self._shared_epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self._scene_stems = self._load_scene_stem_mapping()
        image_stems={p.stem for p in (self.root/'images').glob('*.npy')}
        mask_stems={p.stem for p in (self.root/'masks').glob('*.npy')}
        paired = sorted(image_stems & mask_stems)
        self.stems = [s for s in paired if self._endmember_path(s).is_file()]
        if self.endmember_scope == 'scene' and len(self.stems) != len(paired):
            missing = [s for s in paired if not self._endmember_path(s).is_file()]
            raise FileNotFoundError(
                f'{len(missing)} samples have no scene-level endmember cache under '
                f'{self.scene_endmember_root}; first missing samples: {missing[:10]}'
            )
        if not self.stems: raise RuntimeError(f'no image/mask/E pairs for {self.root}')
        first_cube=load_intensity_cube(self.root/'images'/f'{self.stems[0]}.npy')
        self.raw_band_count=int(first_cube.shape[0])
        raw_wavelengths=load_wavelengths(
            self.root,self.raw_band_count,wavelength_file,self.allow_index_wavelengths
        )
        self.raw_wavelengths=raw_wavelengths
        front,back=band_pad_amounts(self.raw_band_count,spectral_patch_size)
        if front or back:
            warnings.warn(
                f"{self.root}: band count S={self.raw_band_count} is not divisible by "
                f"spectral_patch_size={spectral_patch_size}; mirror-padding by "
                f"duplicating band 0 {front} time(s) at the low-wavelength edge and "
                f"the last band {back} time(s) at the high-wavelength edge (alternating) "
                f"to reach S={self.raw_band_count + front + back}.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.wavelengths=pad_bands(raw_wavelengths,spectral_patch_size,axis=0)

    def _load_scene_stem_mapping(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        manifest = self.root / 'manifest.csv'
        if not manifest.is_file():
            return mapping
        with manifest.open('r', encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                stem = str(row.get('stem', '')).strip()
                source = str(row.get('source_hdr', '')).strip()
                scene = Path(source).stem if source else str(row.get('scene_id', '')).strip()
                if stem and scene:
                    mapping[stem] = scene
        return mapping

    def _scene_stem(self, stem: str) -> str:
        return self._scene_stems.get(stem, infer_segmentation_scene_id(stem))

    def _endmember_path(self, stem: str) -> Path:
        if self.endmember_scope == 'patch':
            return self.nmf_dir / f'{stem}_E.npy'
        assert self.scene_endmember_root is not None
        return self.scene_endmember_root / f'{self._scene_stem(stem)}_E.npy'

    def __len__(self):
        copies = self.augmentation_copies if self.augment else 1
        return len(self.stems) * copies

    def set_epoch(self, epoch: int) -> None:
        self._shared_epoch.fill_(int(epoch))

    def _transform_ids_for(self, base_index: int) -> list[int]:
        generator = torch.Generator(device='cpu').manual_seed(
            self.augmentation_seed
            + 1_000_003 * int(self._shared_epoch.item())
            + base_index
        )
        return torch.randperm(8, generator=generator)[:self.augmentation_copies].tolist()

    def class_pixel_counts(self, num_classes: int) -> torch.Tensor:
        counts = np.zeros(int(num_classes), dtype=np.int64)
        for stem in self.stems:
            mask = np.load(self.root/'masks'/f'{stem}.npy').squeeze()
            if mask.size:
                counts += np.bincount(mask.reshape(-1), minlength=num_classes)[:num_classes]
        return torch.from_numpy(counts)

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        if self.augment:
            base_index, copy_index = divmod(idx, self.augmentation_copies)
            transform_id = self._transform_ids_for(base_index)[copy_index]
        else:
            base_index, copy_index, transform_id = idx, 0, 0
        stem=self.stems[base_index]
        cube, channel_first = _open_intensity_memmap(
            self.root/'images'/f'{stem}.npy'
        )
        intensity = cube if channel_first else cube.transpose(2, 0, 1)
        if intensity.shape[0] != self.raw_band_count:
            raise ValueError(
                f'inconsistent band count for {stem}: {intensity.shape[0]} vs {self.raw_band_count}'
            )
        seg=np.load(self.root/'masks'/f'{stem}.npy').squeeze().astype(np.int64)
        if self.augment:
            augmented = augment_hsi_segmentation_pair(
                intensity,
                seg,
                policy=self.augmentation_policy,
                probability=self.augmentation_probability,
                base_seed=self.augmentation_seed,
                epoch=int(self._shared_epoch.item()),
                sample_index=base_index,
                copy_index=copy_index,
                transform_id=transform_id,
                affine_rotation_degrees=self.affine_rotation_degrees,
                affine_scale_delta=self.affine_scale_delta,
                affine_translate_fraction=self.affine_translate_fraction,
                perspective_scale=self.perspective_scale,
                padding_mode=self.augmentation_padding_mode,
                od_max=self.od_max,
            )
            intensity, seg = augmented.intensity, augmented.mask.astype(np.int64)
        e=np.load(self._endmember_path(stem)).astype(np.float32)
        if e.shape[1] != self.raw_band_count:
            raise ValueError(
                f'endmember shape {e.shape} is incompatible with raw S={self.raw_band_count}'
            )
        model_inputs=build_conditioned_model_inputs(
            intensity,e,self.raw_wavelengths,self.cfg,self.od_max
        )
        return {**model_inputs,'seg':torch.from_numpy(seg),
                'stem':stem,'scene_id':infer_segmentation_scene_id(stem)}


def _collate(samples):
    out={k:torch.stack([s[k] for s in samples]) for k,v in samples[0].items() if isinstance(v,torch.Tensor)}
    out['stem']=[s['stem'] for s in samples]
    out['scene_id']=[s['scene_id'] for s in samples]
    return out


class DistributedSequentialBatchSampler(Sampler[list[int]]):
    """Shard complete sequential evaluation batches across DDP ranks.

    Keeping the original batch boundaries is important because the historical
    ``batch_allclass_macro`` Dice is non-linear within each batch.  A regular
    sample-level distributed sampler would regroup samples and subtly change
    that metric.  This sampler assigns each original batch to exactly one rank,
    so evaluation avoids duplicate work without changing batch composition.
    """

    def __init__(
        self, dataset_size: int, batch_size: int, max_batches: int = 0
    ) -> None:
        if dataset_size < 0:
            raise ValueError('dataset_size must be non-negative')
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        if max_batches < 0:
            raise ValueError('max_batches must be non-negative')
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.max_batches = int(max_batches)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    @property
    def num_global_batches(self) -> int:
        count = (self.dataset_size + self.batch_size - 1) // self.batch_size
        return min(count, self.max_batches) if self.max_batches > 0 else count

    def __iter__(self):
        for batch_index in range(self.rank, self.num_global_batches, self.world_size):
            start = batch_index * self.batch_size
            stop = min(start + self.batch_size, self.dataset_size)
            yield list(range(start, stop))

    def __len__(self) -> int:
        remaining = self.num_global_batches - self.rank
        return max(0, (remaining + self.world_size - 1) // self.world_size)


def build_conditioned_finetune_loaders(train_root,val_root,test_root,
                                       batch_size=4,num_workers=4,distributed=False,
                                       test_inference_mode='direct',
                                       scene_val_root=None,
                                       persistent_workers=False,
                                       prefetch_factor=2,
                                       distributed_validation=False,
                                       validation_max_batches=0,
                                       **kwargs):
    train=ConditionedFinetuneDataset(train_root,**kwargs)
    evaluation_kwargs = dict(kwargs)
    evaluation_kwargs['augment'] = False
    evaluation_kwargs['augmentation_copies'] = 1
    val=ConditionedFinetuneDataset(val_root,**evaluation_kwargs)
    sliding_keys = {
        'patch_size', 'spectral_patch_size', 'nmf_k', 'nmf_l1', 'nmf_l2',
        'nmf_l3', 'nmf_simplex', 'nmf_lam_e', 'nmf_e_clamp_max',
        'nmf_cache_dir', 'wavelength_file', 'allow_index_wavelengths',
        'od_max', 'endmember_scope', 'scene_endmember_root',
    }
    scene_val = (
        ConditionedSlidingWindowSceneDataset(
            scene_val_root,
            **{key: value for key, value in kwargs.items() if key in sliding_keys},
        )
        if scene_val_root
        else None
    )
    if test_root and test_inference_mode == 'sliding_window':
        test=ConditionedSlidingWindowSceneDataset(
            test_root, **{key: value for key, value in kwargs.items() if key in sliding_keys}
        )
    elif test_root and test_inference_mode == 'direct':
        test=ConditionedFinetuneDataset(test_root,**evaluation_kwargs)
    elif test_root:
        raise ValueError(f'unknown test_inference_mode: {test_inference_mode!r}')
    else:
        test=None
    sampler=DistributedSampler(train,shuffle=True) if distributed else None
    if num_workers < 0:
        raise ValueError('num_workers must be non-negative')
    if prefetch_factor <= 0:
        raise ValueError('prefetch_factor must be positive')
    worker_options = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=bool(persistent_workers and num_workers > 0),
    )
    if num_workers > 0:
        worker_options['prefetch_factor'] = int(prefetch_factor)
    common=dict(batch_size=batch_size,collate_fn=_collate,**worker_options)
    train_loader=DataLoader(
        train,shuffle=sampler is None,sampler=sampler,drop_last=True,**common
    )
    if distributed and distributed_validation:
        val_loader=DataLoader(
            val,
            batch_sampler=DistributedSequentialBatchSampler(
                len(val), batch_size, validation_max_batches
            ),
            collate_fn=_collate,
            **worker_options,
        )
    else:
        val_loader=DataLoader(val,shuffle=False,**common)
    test_loader=(test if isinstance(test,ConditionedSlidingWindowSceneDataset)
                 else DataLoader(test,shuffle=False,**common) if test else None)
    if scene_val is not None:
        return train_loader,val_loader,scene_val,test_loader,sampler
    # Preserve the historical four-item public return contract unless the new
    # complete-scene validation feature was explicitly requested.
    return train_loader,val_loader,test_loader,sampler
