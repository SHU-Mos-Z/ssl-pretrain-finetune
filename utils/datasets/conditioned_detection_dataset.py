"""COCO-style detection dataset for the endmember-conditioned HSI backbone."""

from __future__ import annotations

import json
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

from utils.physics.beer_lambert import intensity_to_od_np
from utils.preprocessing.offline_nmf import cache_dir_name
from utils.tokenization.band_padding import band_pad_amounts, pad_bands
from utils.tokenization.spectral_metadata import load_wavelengths, token_spectral_positions
from utils.tokenization.token_builder import TokenBuildConfig, build_tokens
from utils.datasets.detection_view_geometry import (
    DetectionView,
    DetectionViewConfig,
    build_evaluation_views,
    project_annotations_to_view,
    sample_training_view,
)


MODEL_INPUT_KEYS = (
    "od",
    "intensity",
    "e_star",
    "wavelengths",
    "token_raw",
    "token_visible",
    "voxel_visible",
    "pe_spatial",
    "pe_spectral",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"COCO annotation must be an object: {path}")
    return payload


def _category_signature(categories: Any, path: Path) -> tuple[tuple[int, str, str], ...]:
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"COCO categories cannot be empty: {path}")
    signature = tuple(
        sorted(
            (
                int(category["id"]),
                str(category["name"]),
                str(category.get("supercategory", "")),
            )
            for category in categories
        )
    )
    if len({item[0] for item in signature}) != len(signature):
        raise ValueError(f"duplicate COCO category IDs: {path}")
    return signature


def read_coco_annotation_source(path: Path) -> dict[str, Any]:
    """Read one monolithic COCO JSON or merge one-image JSON fragments."""
    if path.is_file():
        return _read_json(path)
    if not path.is_dir():
        raise FileNotFoundError(f"COCO annotation source does not exist: {path}")

    fragment_paths = sorted(path.glob("*.json"))
    if not fragment_paths:
        raise FileNotFoundError(f"no per-patch JSON files found in: {path}")
    merged: dict[str, Any] = {
        "info": {
            "description": "In-memory merge of one-image COCO fragments",
            "annotation_layout": "one_image_coco_fragment_directory",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }
    expected_categories: tuple[tuple[int, str, str], ...] | None = None
    for fragment_path in fragment_paths:
        fragment = _read_json(fragment_path)
        images = fragment.get("images")
        annotations = fragment.get("annotations")
        if not isinstance(images, list) or len(images) != 1:
            raise ValueError(
                f"per-patch COCO JSON must contain exactly one image: {fragment_path}"
            )
        if not isinstance(annotations, list):
            raise ValueError(f"COCO annotations must be a list: {fragment_path}")
        image = images[0]
        image_id = int(image["id"])
        image_stem = Path(str(image["file_name"])).stem
        if image_stem != fragment_path.stem:
            raise ValueError(
                "per-patch JSON/file_name stem mismatch: "
                f"{fragment_path.name} vs {image_stem}"
            )
        for annotation in annotations:
            if int(annotation["image_id"]) != image_id:
                raise ValueError(
                    f"fragment annotation references another image: {fragment_path}"
                )
        categories = fragment.get("categories")
        signature = _category_signature(categories, fragment_path)
        if expected_categories is None:
            expected_categories = signature
            merged["categories"] = categories
            merged["licenses"] = fragment.get("licenses", [])
        elif signature != expected_categories:
            raise ValueError(f"inconsistent COCO categories in {fragment_path}")
        merged["images"].append(image)
        merged["annotations"].extend(annotations)
    return merged


def _boxes_xywh_to_xyxy(boxes: Sequence[Sequence[float]]) -> np.ndarray:
    if not boxes:
        return np.empty((0, 4), dtype=np.float32)
    output = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    output[:, 2] += output[:, 0]
    output[:, 3] += output[:, 1]
    return output


def _npy_cube_shape_chw(
    path: Path, declared_shape: tuple[int, int] | None = None
) -> tuple[int, int, int]:
    """Read only a NumPy header/memmap and mirror ``load_intensity_cube`` layout."""

    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 3:
        raise ValueError(f"expected 3D HSI cube, got {array.shape} from {path}")
    if declared_shape is not None:
        if tuple(array.shape[:2]) == tuple(declared_shape):
            return int(array.shape[2]), int(array.shape[0]), int(array.shape[1])
        if tuple(array.shape[1:]) == tuple(declared_shape):
            return int(array.shape[0]), int(array.shape[1]), int(array.shape[2])
        raise ValueError(
            f"neither HWC nor CHW layout matches declared H/W={declared_shape}: "
            f"{array.shape} from {path}"
        )
    if array.shape[0] <= 64 and array.shape[0] < array.shape[1]:
        return tuple(int(value) for value in array.shape)
    return int(array.shape[2]), int(array.shape[0]), int(array.shape[1])


def _load_intensity_crop(
    path: Path,
    crop_xyxy: tuple[int, int, int, int],
    source_size: tuple[int, int],
) -> np.ndarray:
    """Memory-map one source scene and materialize only the requested CHW crop."""

    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 3:
        raise ValueError(f"expected 3D cube, got {array.shape} from {path}")
    x1, y1, x2, y2 = crop_xyxy
    if tuple(array.shape[1:]) == tuple(source_size):
        crop = array[:, y1:y2, x1:x2]
    elif tuple(array.shape[:2]) == tuple(source_size):
        crop = array[y1:y2, x1:x2, :].transpose(2, 0, 1)
    else:
        raise ValueError(
            f"neither HWC nor CHW layout matches source H/W={source_size}: "
            f"{array.shape} from {path}"
        )
    return np.asarray(crop, dtype=np.float32).copy()


def _resize_chw(array: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    if tuple(array.shape[-2:]) == tuple(output_size):
        return np.ascontiguousarray(array, dtype=np.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)
    resized = F.interpolate(tensor, size=output_size, mode="bilinear", align_corners=False)
    return resized.squeeze(0).numpy().astype(np.float32, copy=False)


def _resize_mask(array: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    if tuple(array.shape) == tuple(output_size):
        return np.ascontiguousarray(array, dtype=np.uint8)
    tensor = torch.from_numpy(np.ascontiguousarray(array)).float()[None, None]
    resized = F.interpolate(tensor, size=output_size, mode="nearest")
    return (resized[0, 0].numpy() > 0.5).astype(np.uint8)


def _transform_boxes(boxes: np.ndarray, height: int, width: int, operation: str) -> np.ndarray:
    if not len(boxes) or operation == "identity":
        return boxes.copy()
    output = boxes.copy()
    x1, y1, x2, y2 = boxes.T
    if operation == "hflip":
        output[:, 0], output[:, 2] = width - x2, width - x1
    elif operation == "vflip":
        output[:, 1], output[:, 3] = height - y2, height - y1
    elif operation == "rot180":
        output[:, 0], output[:, 2] = width - x2, width - x1
        output[:, 1], output[:, 3] = height - y2, height - y1
    else:
        raise ValueError(f"unsupported spatial operation: {operation}")
    return output


def _transform_spatial(array: np.ndarray, operation: str) -> np.ndarray:
    if operation == "identity":
        return np.ascontiguousarray(array)
    if operation == "hflip":
        return np.ascontiguousarray(np.flip(array, axis=-1))
    if operation == "vflip":
        return np.ascontiguousarray(np.flip(array, axis=-2))
    if operation == "rot180":
        return np.ascontiguousarray(np.flip(array, axis=(-2, -1)))
    raise ValueError(f"unsupported spatial operation: {operation}")


class ConditionedDetectionDataset(Dataset):
    """Return ``(conditioned_model_inputs, variable_length_target)``.

    Internal labels are contiguous zero-based indices for sigmoid heads.  Original
    COCO category IDs are retained in ``target['category_ids']`` and mapped back
    during evaluation.  Custom ``ignore=1`` annotations and standard crowd
    annotations never enter ordinary GT boxes.
    """

    def __init__(
        self,
        data_root: str | Path,
        annotation_file: str | Path,
        *,
        patch_size: int = 16,
        spectral_patch_size: int = 5,
        nmf_k: int = 16,
        nmf_l1: float = 5e-4,
        nmf_l2: float = 2e-4,
        nmf_l3: float = 1e-2,
        nmf_simplex: bool = True,
        nmf_lam_e: float = 0.05,
        nmf_e_clamp_max: float = 3.0,
        nmf_cache_dir: str | Path | None = None,
        wavelength_file: str | None = None,
        allow_index_wavelengths: bool = False,
        od_max: float = 3.0,
        augment: bool = False,
        augmentation_probability: float = 0.5,
        seed: int = 42,
        require_ignore_mask: bool = True,
        view_config: DetectionViewConfig | Mapping[str, Any] | None = None,
        training: bool = False,
    ):
        self.root = Path(data_root).expanduser().resolve()
        annotation_path = Path(annotation_file).expanduser()
        if not annotation_path.is_absolute():
            annotation_path = self.root / annotation_path
        self.annotation_path = annotation_path.resolve()
        self.images_dir = self.root / "images"
        self.nmf_dir = (
            Path(nmf_cache_dir).expanduser().resolve()
            if nmf_cache_dir
            else self.root
            / cache_dir_name(
                nmf_k,
                nmf_l1,
                nmf_l2,
                nmf_l3,
                nmf_simplex,
                nmf_lam_e,
                nmf_e_clamp_max,
            )
        )
        self.token_config = TokenBuildConfig(patch_size, spectral_patch_size)
        self.od_max = float(od_max)
        self.augment = bool(augment)
        self.augmentation_probability = float(augmentation_probability)
        self.seed = int(seed)
        if view_config is None:
            self.view_config = DetectionViewConfig(seed=self.seed)
        elif isinstance(view_config, DetectionViewConfig):
            self.view_config = view_config
        else:
            self.view_config = DetectionViewConfig.from_dict(dict(view_config))
        self.view_config.validate()
        self.training = bool(training)
        # DataLoader workers keep independent Dataset objects.  A regular Python
        # integer changed by the training process therefore becomes stale when
        # persistent_workers=True.  Shared tensor storage lets every worker read
        # the epoch most recently set by the training loop without respawning.
        self._shared_epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.require_ignore_mask = bool(require_ignore_mask)
        if not 0 <= self.augmentation_probability <= 1:
            raise ValueError("augmentation_probability must be in [0,1]")
        if not self.images_dir.is_dir() or not self.annotation_path.exists():
            raise FileNotFoundError(
                "missing detection images/annotation source: "
                f"{self.images_dir}, {self.annotation_path}"
            )
        if not self.nmf_dir.is_dir():
            raise FileNotFoundError(
                f"missing offline NMF cache {self.nmf_dir}; run scripts/run_offline_nmf.sh"
            )

        self.coco = read_coco_annotation_source(self.annotation_path)
        categories = sorted(self.coco.get("categories", []), key=lambda item: int(item["id"]))
        if not categories:
            raise ValueError("COCO categories cannot be empty")
        category_ids = [int(item["id"]) for item in categories]
        if len(category_ids) != len(set(category_ids)):
            raise ValueError("duplicate COCO category IDs")
        self.category_id_to_label = {category_id: index for index, category_id in enumerate(category_ids)}
        self.label_to_category_id = {value: key for key, value in self.category_id_to_label.items()}
        self.category_names = {int(item["id"]): str(item["name"]) for item in categories}

        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        annotation_ids: set[int] = set()
        for annotation in self.coco.get("annotations", []):
            annotation_id = int(annotation["id"])
            if annotation_id in annotation_ids:
                raise ValueError(f"duplicate annotation id={annotation_id}")
            annotation_ids.add(annotation_id)
            category_id = int(annotation["category_id"])
            if category_id not in self.category_id_to_label:
                raise ValueError(f"unknown category_id={category_id}")
            annotations_by_image[int(annotation["image_id"])].append(annotation)

        image_ids: set[int] = set()
        records: list[dict[str, Any]] = []
        raw_band_count: int | None = None
        spatial_shape: tuple[int, int] | None = None
        missing: list[str] = []
        for image in self.coco.get("images", []):
            image_id = int(image["id"])
            if image_id in image_ids:
                raise ValueError(f"duplicate image id={image_id}")
            image_ids.add(image_id)
            image_path = self.root / str(image["file_name"])
            stem = image_path.stem
            endmember_path = self.nmf_dir / f"{stem}_E.npy"
            ignore_name = image.get("ignore_mask_file_name")
            ignore_path = self.root / str(ignore_name) if ignore_name else None
            required = [image_path, endmember_path]
            if self.require_ignore_mask:
                if ignore_path is None:
                    missing.append(f"image_id={image_id}: missing ignore_mask_file_name")
                else:
                    required.append(ignore_path)
            absent = [str(path) for path in required if not path.is_file()]
            if absent:
                missing.extend(absent)
                continue
            declared_shape = (int(image["height"]), int(image["width"]))
            cube_shape = _npy_cube_shape_chw(image_path, declared_shape)
            current_shape = cube_shape[1:]
            if current_shape != declared_shape:
                raise ValueError(
                    f"image_id={image_id}: HSI shape={current_shape}, COCO={declared_shape}"
                )
            if spatial_shape is None:
                spatial_shape = current_shape
            elif current_shape != spatial_shape and self.view_config.view_mode == "direct":
                raise ValueError(
                    "all direct-mode images in one ConditionedDetectionDataset must share H/W; "
                    f"got {spatial_shape} and {current_shape}"
                )
            output_shape = (
                current_shape
                if self.view_config.view_mode == "direct"
                else self.view_config.model_input_size
            )
            assert output_shape is not None
            if output_shape[0] % patch_size or output_shape[1] % patch_size:
                raise ValueError(
                    f"model input shape {output_shape} is not divisible by patch_size={patch_size}"
                )
            if raw_band_count is None:
                raw_band_count = cube_shape[0]
            elif cube_shape[0] != raw_band_count:
                raise ValueError("all images in one dataset must share the raw band count")
            records.append(
                {
                    "image": image,
                    "image_path": image_path,
                    "stem": stem,
                    "endmember_path": endmember_path,
                    "ignore_path": ignore_path,
                    "annotations": annotations_by_image.get(image_id, []),
                    "source_size": current_shape,
                }
            )
        if missing:
            preview = "\n".join(missing[:12])
            raise FileNotFoundError(f"incomplete image/NMF/ignore inputs (first 12):\n{preview}")
        if not records or raw_band_count is None or spatial_shape is None:
            raise RuntimeError(f"no usable detection records in {self.annotation_path}")
        unknown_image_ids = set(annotations_by_image) - image_ids
        if unknown_image_ids:
            raise ValueError(f"annotations reference missing image IDs: {sorted(unknown_image_ids)[:10]}")
        self.records = records
        self.raw_band_count = raw_band_count
        self.spatial_shape = (
            spatial_shape
            if self.view_config.view_mode == "direct"
            else self.view_config.model_input_size
        )
        assert self.spatial_shape is not None
        self._evaluation_views: list[tuple[int, DetectionView]] = []
        if not self.training:
            next_view_id = 0
            for source_index, record in enumerate(self.records):
                views = build_evaluation_views(
                    source_image_id=int(record["image"]["id"]),
                    source_stem=str(record["stem"]),
                    source_size=tuple(record["source_size"]),
                    config=self.view_config,
                    first_view_id=next_view_id,
                )
                self._evaluation_views.extend((source_index, view) for view in views)
                next_view_id += len(views)
        raw_wavelengths = load_wavelengths(
            self.root,
            raw_band_count,
            wavelength_file,
            allow_index_wavelengths,
        )
        front, back = band_pad_amounts(raw_band_count, spectral_patch_size)
        if front or back:
            warnings.warn(
                f"{self.root}: padding spectral bands {raw_band_count} -> "
                f"{raw_band_count + front + back}",
                RuntimeWarning,
                stacklevel=2,
            )
        self.wavelengths = pad_bands(raw_wavelengths, spectral_patch_size, axis=0)

    @property
    def num_classes(self) -> int:
        return len(self.category_id_to_label)

    @property
    def image_ids(self) -> list[int]:
        return [int(record["image"]["id"]) for record in self.records]

    @property
    def source_coco(self) -> dict[str, Any]:
        return self.coco

    @property
    def evaluation_view_manifest(self) -> list[dict[str, Any]]:
        return [view.to_dict() for _, view in self._evaluation_views]

    def set_epoch(self, epoch: int) -> None:
        self._shared_epoch.fill_(int(epoch))

    def __len__(self) -> int:
        if self.view_config.view_mode == "direct":
            return len(self.records)
        if self.training:
            return len(self.records) * self.view_config.train_views_per_source
        return len(self._evaluation_views)

    def view_for_index(self, index: int) -> tuple[dict[str, Any], DetectionView]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.training:
            if self.view_config.view_mode == "direct":
                source_index, view_slot = index, 0
            else:
                source_index = index // self.view_config.train_views_per_source
                view_slot = index % self.view_config.train_views_per_source
            record = self.records[source_index]
            view = sample_training_view(
                source_image_id=int(record["image"]["id"]),
                source_stem=str(record["stem"]),
                source_size=tuple(record["source_size"]),
                annotations=record["annotations"],
                config=self.view_config,
                epoch=int(self._shared_epoch.item()),
                source_index=source_index,
                view_slot=view_slot,
            )
            return record, view
        source_index, view = self._evaluation_views[index]
        return self.records[source_index], view

    def _operation_for(self, index: int) -> str:
        if not self.augment:
            return "identity"
        epoch = int(self._shared_epoch.item())
        generator = random.Random(self.seed + 1_000_003 * epoch + index)
        if generator.random() >= self.augmentation_probability:
            return "identity"
        return generator.choice(("hflip", "vflip", "rot180"))

    def __getitem__(self, index: int):
        record, view = self.view_for_index(index)
        image_info = record["image"]
        source_height, source_width = tuple(record["source_size"])
        intensity = _load_intensity_crop(
            record["image_path"], view.crop_xyxy, (source_height, source_width)
        )
        crop_height, crop_width = view.crop_size
        if intensity.shape != (self.raw_band_count, crop_height, crop_width):
            raise ValueError(f"inconsistent HSI shape for {record['stem']}: {intensity.shape}")
        endmembers = np.load(record["endmember_path"], allow_pickle=False).astype(np.float32)
        if endmembers.ndim != 2 or endmembers.shape[1] != self.raw_band_count:
            raise ValueError(f"invalid E* shape for {record['stem']}: {endmembers.shape}")
        if record["ignore_path"] is None:
            ignore_mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
        else:
            source_ignore = np.load(record["ignore_path"], mmap_mode="r", allow_pickle=False)
            if tuple(source_ignore.shape) != (source_height, source_width):
                raise ValueError(
                    f"invalid ignore mask shape for {record['stem']}: {source_ignore.shape}"
                )
            x1, y1, x2, y2 = view.crop_xyxy
            ignore_mask = (np.asarray(source_ignore[y1:y2, x1:x2]) > 0).astype(np.uint8)

        projected = project_annotations_to_view(
            record["annotations"], view, self.view_config
        )
        ordinary = projected["positive"]
        crowd = projected["crowd"]
        ignored = projected["ignored"]

        output_height, output_width = view.output_size
        intensity = _resize_chw(intensity, view.output_size)
        ignore_mask = _resize_mask(ignore_mask, view.output_size)

        operation = self._operation_for(index)
        intensity = _transform_spatial(intensity, operation)
        ignore_mask = _transform_spatial(ignore_mask, operation)

        def transformed(items):
            boxes = (
                _boxes_xywh_to_xyxy([])
                if not items
                else np.asarray([item["box_xyxy"] for item in items], dtype=np.float32)
            )
            return _transform_boxes(boxes, output_height, output_width, operation)

        boxes = transformed(ordinary)
        crowd_boxes = transformed(crowd)
        ignore_boxes = transformed(ignored)
        intensity = pad_bands(intensity, self.token_config.spectral_patch_size, axis=0)
        endmembers = pad_bands(endmembers, self.token_config.spectral_patch_size, axis=1)
        od = np.clip(intensity_to_od_np(intensity), 0, self.od_max).astype(np.float32)
        spectral_bands = int(od.shape[0])
        if spectral_bands != self.wavelengths.size:
            raise ValueError("padded intensity/wavelength size mismatch")
        dummy_abundance = np.full(
            (endmembers.shape[0], output_height, output_width),
            1.0 / endmembers.shape[0],
            np.float32,
        )
        tokens = build_tokens(od, dummy_abundance, self.token_config)
        tokens.pe_spectral[:] = token_spectral_positions(
            self.wavelengths,
            tokens.h_p,
            tokens.w_p,
            tokens.s_p,
        )
        model_inputs = {
            "od": torch.from_numpy(od),
            "intensity": torch.from_numpy(intensity.astype(np.float32, copy=False)),
            "e_star": torch.from_numpy(endmembers),
            "wavelengths": torch.from_numpy(self.wavelengths.copy()),
            "token_raw": torch.from_numpy(tokens.token_raw),
            "token_visible": torch.ones((tokens.h_p, tokens.w_p, tokens.n_sp), dtype=torch.bool),
            "voxel_visible": torch.ones(
                (spectral_bands, output_height, output_width), dtype=torch.bool
            ),
            "pe_spatial": torch.from_numpy(tokens.pe_spatial),
            "pe_spectral": torch.from_numpy(tokens.pe_spectral),
        }
        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.tensor(
                [self.category_id_to_label[int(item["category_id"])] for item in ordinary],
                dtype=torch.int64,
            ),
            "category_ids": torch.tensor(
                [int(item["category_id"]) for item in ordinary], dtype=torch.int64
            ),
            "iscrowd": torch.tensor(
                [int(item.get("iscrowd", 0)) for item in ordinary], dtype=torch.int64
            ),
            "area": torch.tensor(
                [
                    float((box[2] - box[0]) * (box[3] - box[1]))
                    for box in boxes
                ],
                dtype=torch.float32,
            ),
            "annotation_ids": torch.tensor(
                [int(item["id"]) for item in ordinary], dtype=torch.int64
            ),
            "source_truncated": torch.tensor(
                [bool(item.get("source_truncated", False)) for item in ordinary],
                dtype=torch.bool,
            ),
            "crop_truncated": torch.tensor(
                [bool(item.get("crop_truncated", False)) for item in ordinary],
                dtype=torch.bool,
            ),
            "crowd_boxes": torch.from_numpy(crowd_boxes),
            "ignore_boxes": torch.from_numpy(ignore_boxes),
            "ignore_mask": torch.from_numpy(ignore_mask.astype(np.bool_)),
            "image_id": torch.tensor([int(view.view_id)], dtype=torch.int64),
            "source_image_id": torch.tensor([int(image_info["id"])], dtype=torch.int64),
            "image_size": torch.tensor([output_height, output_width], dtype=torch.int64),
            "source_size": torch.tensor([source_height, source_width], dtype=torch.int64),
            "crop_xyxy": torch.tensor(view.crop_xyxy, dtype=torch.float32),
            "ownership_xyxy": torch.tensor(view.ownership_xyxy, dtype=torch.float32),
            "scale_xy": torch.tensor(view.scale_xy, dtype=torch.float32),
            "sample_index": torch.tensor(index, dtype=torch.int64),
            "stem": record["stem"],
            "augmentation": operation,
        }
        return model_inputs, target


def conditioned_detection_collate(samples):
    if not samples:
        raise ValueError("cannot collate an empty batch")
    inputs = {
        key: torch.stack([sample[0][key] for sample in samples], dim=0)
        for key in MODEL_INPUT_KEYS
    }
    targets = [sample[1] for sample in samples]
    sizes = {tuple(int(v) for v in target["image_size"].tolist()) for target in targets}
    if len(sizes) != 1:
        raise ValueError(f"one detection batch must share H/W, got {sorted(sizes)}")
    return inputs, targets


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation records without duplicate-padding the tail."""

    def __init__(self, dataset: Dataset, num_replicas: int, rank: int):
        if not 0 <= rank < num_replicas:
            raise ValueError("invalid distributed rank")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        return (len(self.dataset) - self.rank + self.num_replicas - 1) // self.num_replicas


def build_conditioned_detection_loaders(
    train_root: str,
    train_annotation: str,
    val_root: str,
    val_annotation: str,
    test_root: str | None,
    test_annotation: str | None,
    *,
    batch_size: int = 2,
    num_workers: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    augment: bool = True,
    **dataset_kwargs,
):
    train_dataset = ConditionedDetectionDataset(
        train_root,
        train_annotation,
        augment=augment,
        training=True,
        **dataset_kwargs,
    )
    val_dataset = ConditionedDetectionDataset(
        val_root,
        val_annotation,
        augment=False,
        training=False,
        **dataset_kwargs,
    )
    test_dataset = (
        ConditionedDetectionDataset(
            test_root,
            test_annotation,
            augment=False,
            training=False,
            **dataset_kwargs,
        )
        if test_root and test_annotation
        else None
    )
    mapping = train_dataset.category_id_to_label
    for name, dataset in (("validation", val_dataset), ("test", test_dataset)):
        if dataset is not None and dataset.category_id_to_label != mapping:
            raise ValueError(f"{name} category mapping differs from training mapping")
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        if distributed
        else None
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
        collate_fn=conditioned_detection_collate,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, sampler=val_sampler, **common)
    test_loader = (
        DataLoader(test_dataset, shuffle=False, sampler=test_sampler, **common)
        if test_dataset is not None
        else None
    )
    return train_loader, val_loader, test_loader, train_sampler
