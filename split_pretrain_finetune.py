#!/usr/bin/env python3
"""基于 NMF 重建误差过滤 + 比例切分，把一个已预处理的数据集拆分为：

  {data_root}_pretrain_p{ratio}_{date}
      仅 images/（+ 对应 NMF 缓存子集），供自监督预训练
  {data_root}_finetune_train_p{ratio}_{date} / _val / _test
      分割：images/ + masks/ + nmf_cache_xxx/；
      分类：{class}/images/ + {class}/nmf_cache_xxx/；
      检测：images/ + masks/ + ignore_masks/ + annotations/ +
            review_patch_visualizations/ + nmf_cache_xxx/

  目录名中的 p{ratio} 表示该目录样本数占 MSE 过滤后全集的百分比（四舍五入）；
  {date} 为 YYYYMMDD，同一次运行生成的 pretrain/finetune 目录日期一致。

  PRETRAIN_RATIO=1.0 时仅生成 pretrain 目录；PRETRAIN_RATIO=0.0 时仅生成 finetune 目录。

拆分逻辑：
  1. 分类数据集可先按 --exclude-json 中的精确 (class_name, stem) 清单排除
     已经人工确认无效的样本；排除在任何 split 生成之前生效。
  2. 根据 NMF 分解参数定位 nmf_cache_xxx/ 目录（分割数据集在 data_root 下
     一个；分类数据集在每个类别子目录下各一个，与 run_offline_nmf.sh 的
     产出结构一致）。
  3. 读取该目录下的 nmf_reconstruction_mse.json，剔除未记录重建误差或
     误差超过 --mse-threshold 的样本。
  4. 分割数据集中，微调候选还需与 masks/ 目录取交集（预训练候选不要求
     mask）；没有 mask 的样本优先划入预训练，不会被浪费。
  5. 按 --pretrain-ratio 切出预训练子集，剩余按 --finetune-val-ratio /
     --finetune-test-ratio 再切 train/val/test（分类数据集按类别分别
     切分；检测数据集按 source_stem 等来源字段分组，并近似平衡图像数、
     可训练框总数与逐类别框数）。
  6. 所有输出目录中的数据文件均为软链接，指向原始数据项，不复制。

用法示例（分割数据集）：
  ./scripts/run_split_pretrain_finetune_segmentation.sh
  # 或
  python split_pretrain_finetune.py \\
      --data-root data/MDC_..._preprocessed \\
      --kind segmentation \\
      --k 16 --l1 5e-4 --l2 2e-4 --l3 1e-2 \\
      --use-simplex --lam-e 0.05 --e-clamp-max 3.0 \\
      --mse-threshold 0.02 --pretrain-ratio 0.3 \\
      --finetune-val-ratio 0.15 --finetune-test-ratio 0.15

用法示例（分类数据集，按类别子目录组织）：
  DATA_ROOT=data/2018WBC_..._bands50 ./scripts/run_split_pretrain_finetune_classification.sh

用法示例（检测数据集）：
  DATA_ROOT=data/2018WBC_detection_... ./scripts/run_split_pretrain_finetune_detection.sh
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.preprocessing.offline_nmf import (  # noqa: E402
    NMF_RECONSTRUCTION_MSE_FILENAME,
    cache_dir_name,
)
from utils.sample_exclusion import load_excluded_samples  # noqa: E402


@dataclass
class Bucket:
    """一个"类别桶"：分割数据集只有一个（label=None）；
    分类数据集每个类别一个。
    """

    label: str | None
    root: Path
    images_dir: Path
    masks_dir: Path | None
    nmf_dir: Path
    annotations_dir: Path | None = None
    ignore_masks_dir: Path | None = None
    review_dir: Path | None = None


@dataclass
class Item:
    bucket: Bucket
    stem: str
    mse: float
    has_mask: bool
    detection_group: str | None = None
    ordinary_count: int = 0
    category_counts: dict[int, int] = field(default_factory=dict)

    @property
    def image_path(self) -> Path:
        return self.bucket.images_dir / f"{self.stem}.npy"

    @property
    def mask_path(self) -> Path:
        assert self.bucket.masks_dir is not None
        return self.bucket.masks_dir / f"{self.stem}.npy"

    @property
    def c_path(self) -> Path:
        return self.bucket.nmf_dir / f"{self.stem}_C.npy"

    @property
    def e_path(self) -> Path:
        return self.bucket.nmf_dir / f"{self.stem}_E.npy"

    @property
    def annotation_path(self) -> Path:
        assert self.bucket.annotations_dir is not None
        return self.bucket.annotations_dir / f"{self.stem}.json"

    @property
    def ignore_mask_path(self) -> Path:
        assert self.bucket.ignore_masks_dir is not None
        return self.bucket.ignore_masks_dir / f"{self.stem}.npy"

    @property
    def review_path(self) -> Path:
        assert self.bucket.review_dir is not None
        return self.bucket.review_dir / f"{self.stem}.png"

    @property
    def link_stem(self) -> str:
        """跨类别铺平到 pretrain/images 时使用的唯一文件名
        （分类数据集加类别前缀）。
        """
        if self.bucket.label:
            return f"{self.bucket.label}__{self.stem}"
        return self.stem


# ────────────────────────────────────────────────────────────────
# 桶发现 / 样本收集
# ────────────────────────────────────────────────────────────────


def discover_buckets(
    data_root: Path,
    kind: str,
    class_dirs: list[str] | None,
    nmf_dir_name: str,
) -> list[Bucket]:
    if kind == "segmentation":
        images_dir = data_root / "images"
        masks_dir = data_root / "masks"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"缺少 images 目录: {images_dir}")
        return [
            Bucket(
                None,
                data_root,
                images_dir,
                masks_dir if masks_dir.is_dir() else None,
                data_root / nmf_dir_name,
            )
        ]

    if kind == "detection":
        required_dirs = {
            "images": data_root / "images",
            "masks": data_root / "masks",
            "ignore_masks": data_root / "ignore_masks",
            "annotations": data_root / "annotations",
            "review_patch_visualizations": data_root / "review_patch_visualizations",
        }
        missing = [str(path) for path in required_dirs.values() if not path.is_dir()]
        if missing:
            raise FileNotFoundError(
                "检测数据集缺少必需目录:\n  " + "\n  ".join(missing)
            )
        wavelength_path = data_root / "wavelengths.npy"
        if not wavelength_path.is_file():
            raise FileNotFoundError(f"检测数据集缺少 wavelengths.npy: {wavelength_path}")
        return [
            Bucket(
                label=None,
                root=data_root,
                images_dir=required_dirs["images"],
                masks_dir=required_dirs["masks"],
                nmf_dir=data_root / nmf_dir_name,
                annotations_dir=required_dirs["annotations"],
                ignore_masks_dir=required_dirs["ignore_masks"],
                review_dir=required_dirs["review_patch_visualizations"],
            )
        ]

    # classification
    if class_dirs:
        names = list(class_dirs)
    else:
        names = sorted(
            p.name
            for p in data_root.iterdir()
            if p.is_dir() and (p / "images").is_dir()
        )
    if not names:
        raise FileNotFoundError(f"在 {data_root} 下未找到含 images/ 的类别子目录")
    buckets: list[Bucket] = []
    for name in names:
        root = data_root / name
        images_dir = root / "images"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"类别 '{name}' 缺少 images 目录: {images_dir}")
        buckets.append(Bucket(name, root, images_dir, None, root / nmf_dir_name))
    return buckets


def read_detection_fragment(
    path: Path,
    expected_stem: str,
    group_field: str,
) -> tuple[str, int, dict[int, int], int, set[int], tuple[tuple[int, str], ...]]:
    """Validate one-image COCO JSON and return split-balancing metadata."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"检测标注必须是 JSON object: {path}")
    images = payload.get("images")
    annotations = payload.get("annotations")
    categories = payload.get("categories")
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError(f"逐 Patch JSON 必须且只能含一个 images 记录: {path}")
    if not isinstance(annotations, list):
        raise ValueError(f"annotations 必须是 list: {path}")
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"categories 不能为空: {path}")

    image = images[0]
    image_id = int(image["id"])
    file_stem = Path(str(image["file_name"])).stem
    if file_stem != expected_stem:
        raise ValueError(
            f"JSON 文件名与 images[0].file_name 不配对: {path.name} vs {file_stem}"
        )
    group_value = image.get(group_field)
    if group_value is None or str(group_value).strip() == "":
        raise ValueError(f"images[0] 缺少检测分组字段 {group_field!r}: {path}")

    valid_category_ids = {int(category["id"]) for category in categories}
    category_signature = tuple(
        sorted((int(category["id"]), str(category["name"])) for category in categories)
    )
    if len(category_signature) != len(valid_category_ids):
        raise ValueError(f"categories 中存在重复 id: {path}")
    annotation_ids: set[int] = set()
    category_counts: dict[int, int] = {}
    ordinary_count = 0
    for annotation in annotations:
        annotation_id = int(annotation["id"])
        if annotation_id in annotation_ids:
            raise ValueError(f"同一 JSON 内 annotation id 重复: {annotation_id}, {path}")
        annotation_ids.add(annotation_id)
        if int(annotation["image_id"]) != image_id:
            raise ValueError(f"annotation 引用了其他 image_id: {path}")
        category_id = int(annotation["category_id"])
        if category_id not in valid_category_ids:
            raise ValueError(f"annotation 使用未知 category_id={category_id}: {path}")
        if not bool(annotation.get("ignore", 0)) and not bool(
            annotation.get("iscrowd", 0)
        ):
            ordinary_count += 1
            category_counts[category_id] = category_counts.get(category_id, 0) + 1
    return (
        str(group_value),
        ordinary_count,
        category_counts,
        image_id,
        annotation_ids,
        category_signature,
    )


def load_mse_stem_index(bucket: Bucket) -> dict[str, float]:
    mse_json = bucket.nmf_dir / NMF_RECONSTRUCTION_MSE_FILENAME
    if not mse_json.is_file():
        raise FileNotFoundError(
            f"缺少 MSE 索引: {mse_json}\n"
            f"  请先对该目录运行 scripts/run_offline_nmf.sh"
            f"（DATA_ROOT={bucket.root}），"
            f"确保生成 {NMF_RECONSTRUCTION_MSE_FILENAME}"
        )
    raw = json.loads(mse_json.read_text(encoding="utf-8"))
    return {Path(k).stem: float(v) for k, v in raw.items()}


def collect_items(
    bucket: Bucket,
    mse_threshold: float,
    kind: str,
    detection_group_field: str,
    excluded_identities: set[tuple[str, str]],
) -> tuple[list[Item], dict[str, int]]:
    mse_stem_index = load_mse_stem_index(bucket)
    image_stems = sorted(p.stem for p in bucket.images_dir.glob("*.npy"))
    mask_stems = None
    if bucket.masks_dir is not None:
        mask_stems = {p.stem for p in bucket.masks_dir.glob("*.npy")}
    image_stem_set = set(image_stems)
    if kind == "detection":
        assert bucket.annotations_dir is not None
        assert bucket.ignore_masks_dir is not None
        assert bucket.review_dir is not None
        companion_stems = {
            "masks": mask_stems or set(),
            "annotations": {p.stem for p in bucket.annotations_dir.glob("*.json")},
            "ignore_masks": {p.stem for p in bucket.ignore_masks_dir.glob("*.npy")},
            "review_patch_visualizations": {
                p.stem for p in bucket.review_dir.glob("*.png")
            },
        }
        mismatches: list[str] = []
        for companion_name, stems in companion_stems.items():
            missing = sorted(image_stem_set - stems)
            extra = sorted(stems - image_stem_set)
            if missing or extra:
                mismatches.append(
                    f"{companion_name}: missing={missing[:8]} (n={len(missing)}), "
                    f"extra={extra[:8]} (n={len(extra)})"
                )
        if mismatches:
            raise ValueError(
                "检测数据集逐 Patch 文件未严格一一配对:\n  "
                + "\n  ".join(mismatches)
            )

    stats = {
        "total_images": len(image_stems),
        "manually_excluded": 0,
        "no_mse_record": 0,
        "mse_over_threshold": 0,
        "missing_nmf_cache": 0,
        "no_mask": 0,
        "missing_annotation": 0,
        "missing_ignore_mask": 0,
        "missing_review_visualization": 0,
        "kept": 0,
    }
    items: list[Item] = []
    detection_image_ids: set[int] = set()
    detection_annotation_ids: set[int] = set()
    detection_category_signature: tuple[tuple[int, str], ...] | None = None
    for stem in image_stems:
        if bucket.label is not None and (bucket.label, stem) in excluded_identities:
            stats["manually_excluded"] += 1
            continue
        mse = mse_stem_index.get(stem)
        if mse is None:
            stats["no_mse_record"] += 1
            continue
        if mse > mse_threshold:
            stats["mse_over_threshold"] += 1
            continue
        c_file = bucket.nmf_dir / f"{stem}_C.npy"
        e_file = bucket.nmf_dir / f"{stem}_E.npy"
        if not c_file.is_file() or not e_file.is_file():
            stats["missing_nmf_cache"] += 1
            continue
        has_mask = True
        if mask_stems is not None:
            has_mask = stem in mask_stems
            if not has_mask:
                stats["no_mask"] += 1
        detection_group = None
        ordinary_count = 0
        category_counts: dict[int, int] = {}
        if kind == "detection":
            assert bucket.annotations_dir is not None
            assert bucket.ignore_masks_dir is not None
            assert bucket.review_dir is not None
            annotation_path = bucket.annotations_dir / f"{stem}.json"
            ignore_mask_path = bucket.ignore_masks_dir / f"{stem}.npy"
            review_path = bucket.review_dir / f"{stem}.png"
            missing_companion = False
            if not annotation_path.is_file():
                stats["missing_annotation"] += 1
                missing_companion = True
            if not ignore_mask_path.is_file():
                stats["missing_ignore_mask"] += 1
                missing_companion = True
            if not review_path.is_file():
                stats["missing_review_visualization"] += 1
                missing_companion = True
            if not has_mask or missing_companion:
                continue
            (
                detection_group,
                ordinary_count,
                category_counts,
                image_id,
                annotation_ids,
                category_signature,
            ) = read_detection_fragment(
                annotation_path,
                expected_stem=stem,
                group_field=detection_group_field,
            )
            if image_id in detection_image_ids:
                raise ValueError(f"检测 JSON 之间 image id 重复: {image_id}")
            duplicate_annotation_ids = detection_annotation_ids & annotation_ids
            if duplicate_annotation_ids:
                raise ValueError(
                    "检测 JSON 之间 annotation id 重复: "
                    f"{sorted(duplicate_annotation_ids)[:10]}"
                )
            detection_image_ids.add(image_id)
            detection_annotation_ids.update(annotation_ids)
            if detection_category_signature is None:
                detection_category_signature = category_signature
            elif category_signature != detection_category_signature:
                raise ValueError(
                    f"检测 JSON 的 categories 不一致: {annotation_path}"
                )
        items.append(
            Item(
                bucket,
                stem,
                mse,
                has_mask,
                detection_group=detection_group,
                ordinary_count=ordinary_count,
                category_counts=category_counts,
            )
        )
        stats["kept"] += 1
    return items, stats


# ────────────────────────────────────────────────────────────────
# 目录命名
# ────────────────────────────────────────────────────────────────


def format_ratio_tag(count: int, total: int) -> str:
    """返回 p{percent}，percent 为占全集（MSE 过滤后）的整数百分比。"""
    if total <= 0:
        return "p000"
    pct = round(100 * count / total)
    return f"p{min(max(pct, 0), 100):03d}"


def build_output_dir(
    data_root: Path, role: str, count: int, total: int, run_date: str,
) -> Path:
    """例如 data/MDC_..._pretrain_p030_20260717"""
    ratio_tag = format_ratio_tag(count, total)
    return Path(f"{data_root}_{role}_{ratio_tag}_{run_date}")


# ────────────────────────────────────────────────────────────────
# 切分
# ────────────────────────────────────────────────────────────────


def split_bucket_items(
    items: list[Item],
    pretrain_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Item]]:
    """
    切分策略：
      - 微调候选 = has_mask 的样本；没有 mask 的样本无法进入微调，
        优先划入预训练。
      - 先按 pretrain_ratio 确定"微调"目标数量
        = round((1 - pretrain_ratio) * 总数)，从有 mask 的样本中
        随机抽取；不足则微调用尽所有有 mask 样本（此时预训练占比
        会高于设定值）。
      - 预训练 = 总数 - 微调（含全部无 mask 样本 + 未被抽中的
        有 mask 样本）。
      - 微调集合再按 val_ratio / test_ratio 切 train/val/test。
    """
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)

    has_mask_items = [it for it in shuffled if it.has_mask]
    no_mask_items = [it for it in shuffled if not it.has_mask]

    n_total = len(shuffled)

    if pretrain_ratio >= 1.0:
        finetune_target = 0
    elif pretrain_ratio <= 0.0:
        finetune_target = len(has_mask_items)
    else:
        finetune_target = round(n_total * (1.0 - pretrain_ratio))
        finetune_target = min(finetune_target, len(has_mask_items))

    finetune_items = has_mask_items[:finetune_target]
    pretrain_items = no_mask_items + has_mask_items[finetune_target:]

    m = len(finetune_items)
    n_val = round(m * val_ratio)
    n_test = round(m * test_ratio)
    n_train = m - n_val - n_test
    return {
        "pretrain": pretrain_items,
        "finetune_train": finetune_items[:n_train],
        "finetune_val": finetune_items[n_train : n_train + n_val],
        "finetune_test": finetune_items[n_train + n_val :],
    }


def classification_group_id(stem: str, group_regex: str) -> str:
    """Extract a classification split group from a sample stem.

    A named group called ``group`` is preferred; otherwise the first capture
    group or the complete match is used. Grouping is opt-in and never affects
    segmentation splitting.
    """
    match = re.match(group_regex, stem)
    if match is None:
        raise ValueError(
            f"classification group regex {group_regex!r} does not match stem {stem!r}"
        )
    if "group" in match.groupdict():
        value = match.group("group")
    elif match.groups():
        value = match.group(1)
    else:
        value = match.group(0)
    if not value:
        raise ValueError(f"empty group id extracted from stem {stem!r}")
    return value


def split_classification_grouped_items(
    items: list[Item],
    pretrain_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    group_regex: str,
) -> dict[str, list[Item]]:
    """Split one class bucket by groups so related patches never cross splits."""
    groups: dict[str, list[Item]] = {}
    for item in items:
        group_id = classification_group_id(item.stem, group_regex)
        groups.setdefault(group_id, []).append(item)
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)

    num_groups = len(group_ids)
    num_pretrain = round(num_groups * pretrain_ratio)
    finetune_groups = num_groups - num_pretrain
    num_val = round(finetune_groups * val_ratio)
    num_test = round(finetune_groups * test_ratio)
    num_train = finetune_groups - num_val - num_test
    boundaries = (
        num_pretrain,
        num_pretrain + num_train,
        num_pretrain + num_train + num_val,
    )
    split_groups = {
        "pretrain": group_ids[: boundaries[0]],
        "finetune_train": group_ids[boundaries[0] : boundaries[1]],
        "finetune_val": group_ids[boundaries[1] : boundaries[2]],
        "finetune_test": group_ids[boundaries[2] :],
    }
    return {
        split_name: [item for group_id in ids for item in groups[group_id]]
        for split_name, ids in split_groups.items()
    }


def split_detection_grouped_items(
    items: list[Item],
    pretrain_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Item]]:
    """Group-aware detection split with approximate image/box/category balance."""
    split_ratios = {
        "pretrain": pretrain_ratio,
        "finetune_train": (1.0 - pretrain_ratio) * (1.0 - val_ratio - test_ratio),
        "finetune_val": (1.0 - pretrain_ratio) * val_ratio,
        "finetune_test": (1.0 - pretrain_ratio) * test_ratio,
    }
    active_splits = [name for name, ratio in split_ratios.items() if ratio > 0]
    groups: dict[str, list[Item]] = {}
    for item in items:
        if item.detection_group is None:
            raise ValueError(f"检测样本缺少分组信息: {item.stem}")
        groups.setdefault(item.detection_group, []).append(item)
    if len(groups) < len(active_splits):
        raise ValueError(
            f"检测数据只有 {len(groups)} 个来源 group，无法保证 "
            f"{len(active_splits)} 个非空 split；请调整划分比例"
        )

    category_ids = sorted(
        {category_id for item in items for category_id in item.category_counts}
    )

    def group_metrics(group_items: list[Item]) -> dict[str, float]:
        metrics = {
            "images": float(len(group_items)),
            "ordinary_boxes": float(sum(item.ordinary_count for item in group_items)),
        }
        for category_id in category_ids:
            metrics[f"category_{category_id}"] = float(
                sum(item.category_counts.get(category_id, 0) for item in group_items)
            )
        return metrics

    group_records = [
        (group_id, groups[group_id], group_metrics(groups[group_id]))
        for group_id in sorted(groups)
    ]
    rng = random.Random(seed)
    rng.shuffle(group_records)
    group_records.sort(
        key=lambda record: (
            record[2]["ordinary_boxes"],
            record[2]["images"],
        ),
        reverse=True,
    )
    total_metrics = group_metrics(items)
    metric_names = [
        name for name, total in total_metrics.items() if total > 0
    ]
    assigned_groups: dict[str, list[tuple[str, list[Item], dict[str, float]]]] = {
        name: [] for name in split_ratios
    }
    current_metrics = {
        name: {metric: 0.0 for metric in total_metrics} for name in split_ratios
    }

    # 先给每个启用的 split 一个 group，避免极端不平衡时出现空集合。
    for split_name, record in zip(active_splits, group_records[: len(active_splits)]):
        assigned_groups[split_name].append(record)
        for metric, value in record[2].items():
            current_metrics[split_name][metric] += value

    def global_score(candidate_split: str, record_metrics: dict[str, float]) -> float:
        score = 0.0
        for split_name in active_splits:
            ratio = split_ratios[split_name]
            for metric in metric_names:
                target = total_metrics[metric] * ratio
                value = current_metrics[split_name][metric]
                if split_name == candidate_split:
                    value += record_metrics[metric]
                score += ((value - target) / max(target, 1.0)) ** 2
        return score

    for record in group_records[len(active_splits) :]:
        candidates = active_splits[:]
        rng.shuffle(candidates)
        chosen = min(candidates, key=lambda name: global_score(name, record[2]))
        assigned_groups[chosen].append(record)
        for metric, value in record[2].items():
            current_metrics[chosen][metric] += value

    result = {
        split_name: [
            item
            for _group_id, group_items, _metrics in assigned_groups[split_name]
            for item in group_items
        ]
        for split_name in split_ratios
    }
    assigned = [item.stem for split_items in result.values() for item in split_items]
    if len(assigned) != len(items) or len(set(assigned)) != len(items):
        raise RuntimeError("检测分组划分未能做到每个样本恰好分配一次")
    return result


# ────────────────────────────────────────────────────────────────
# 落地（软链接）
# ────────────────────────────────────────────────────────────────


def make_symlink(link_path: Path, target: Path, dry_run: bool) -> None:
    if dry_run:
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(target.resolve())


def link_wavelength_file(
    data_root: Path,
    buckets: list[Bucket],
    pretrain_root: Path,
    dry_run: bool,
) -> None:
    candidates = [data_root / "wavelengths.npy"]
    candidates += [b.root / "wavelengths.npy" for b in buckets]
    for cand in candidates:
        if cand.is_file():
            make_symlink(pretrain_root / "wavelengths.npy", cand, dry_run)
            return


def materialize_pretrain(
    items: list[Item],
    pretrain_root: Path,
    nmf_dir_name: str,
    dry_run: bool,
) -> None:
    images_out = pretrain_root / "images"
    nmf_out = pretrain_root / nmf_dir_name
    for it in items:
        make_symlink(
            images_out / f"{it.link_stem}.npy",
            it.image_path,
            dry_run,
        )
        make_symlink(
            nmf_out / f"{it.link_stem}_C.npy",
            it.c_path,
            dry_run,
        )
        make_symlink(
            nmf_out / f"{it.link_stem}_E.npy",
            it.e_path,
            dry_run,
        )


def materialize_finetune(
    items: list[Item],
    split_root: Path,
    kind: str,
    nmf_dir_name: str,
    dry_run: bool,
) -> None:
    for it in items:
        if kind == "segmentation":
            make_symlink(
                split_root / "images" / f"{it.stem}.npy",
                it.image_path,
                dry_run,
            )
            make_symlink(
                split_root / "masks" / f"{it.stem}.npy",
                it.mask_path,
                dry_run,
            )
            nmf_out = split_root / nmf_dir_name
        elif kind == "detection":
            make_symlink(
                split_root / "images" / f"{it.stem}.npy",
                it.image_path,
                dry_run,
            )
            make_symlink(
                split_root / "masks" / f"{it.stem}.npy",
                it.mask_path,
                dry_run,
            )
            make_symlink(
                split_root / "ignore_masks" / f"{it.stem}.npy",
                it.ignore_mask_path,
                dry_run,
            )
            make_symlink(
                split_root / "annotations" / f"{it.stem}.json",
                it.annotation_path,
                dry_run,
            )
            make_symlink(
                split_root / "review_patch_visualizations" / f"{it.stem}.png",
                it.review_path,
                dry_run,
            )
            nmf_out = split_root / nmf_dir_name
        else:
            class_root = split_root / (it.bucket.label or "unknown")
            make_symlink(
                class_root / "images" / f"{it.stem}.npy",
                it.image_path,
                dry_run,
            )
            nmf_out = class_root / nmf_dir_name
        make_symlink(
            nmf_out / f"{it.stem}_C.npy",
            it.c_path,
            dry_run,
        )
        make_symlink(
            nmf_out / f"{it.stem}_E.npy",
            it.e_path,
            dry_run,
        )


# ────────────────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"data-root 不存在: {data_root}")

    run_date = args.run_date or date.today().strftime("%Y%m%d")
    gen_pretrain = args.pretrain_ratio > 0.0
    gen_finetune = args.pretrain_ratio < 1.0

    nmf_dir_name = cache_dir_name(
        args.k,
        args.l1,
        args.l2,
        args.l3,
        simplex=args.use_simplex,
        lam_e=args.lam_e,
        e_clamp_max=args.e_clamp_max,
    )
    print("=" * 78)
    print(f"数据集: {data_root}  kind={args.kind}")
    print(f"NMF 缓存目录名: {nmf_dir_name}")
    print(f"MSE 阈值: {args.mse_threshold}  " f"预训练占比: {args.pretrain_ratio}")
    print(f"生成 pretrain: {gen_pretrain}  生成 finetune: {gen_finetune}  日期: {run_date}")
    print(
        f"微调 val/test 占比: "
        f"{args.finetune_val_ratio}/{args.finetune_test_ratio}  "
        f"seed={args.seed}"
    )
    if args.dry_run:
        print("[dry-run] 只统计与打印，不生成任何软链接")
    print("=" * 78)

    buckets = discover_buckets(
        data_root,
        args.kind,
        args.class_dirs,
        nmf_dir_name,
    )

    excluded_identities = (
        load_excluded_samples(args.exclude_json) if args.exclude_json else set()
    )
    if excluded_identities:
        available_identities = {
            (bucket.label, image_path.stem)
            for bucket in buckets
            if bucket.label is not None
            for image_path in bucket.images_dir.glob("*.npy")
        }
        unmatched_exclusions = excluded_identities - available_identities
        if unmatched_exclusions:
            preview = ", ".join(
                f"{class_name}/{stem}"
                for class_name, stem in sorted(unmatched_exclusions)[:20]
            )
            raise ValueError(
                f"样本排除 JSON 中有 {len(unmatched_exclusions)} 条记录无法在 "
                f"data-root 中匹配；前若干项: {preview}"
            )
        print(
            f"人工排除清单: {args.exclude_json}  "
            f"精确匹配 {len(excluded_identities)} 项"
        )
    else:
        print("人工排除清单: <disabled>")

    manifest: dict[str, dict] = {"buckets": {}}
    all_splits: dict[str, list[Item]] = {
        "pretrain": [],
        "finetune_train": [],
        "finetune_val": [],
        "finetune_test": [],
    }
    total_kept = 0

    for bucket in buckets:
        label = bucket.label or "<root>"
        print(f"\n── 桶: {label}")
        print(f"   images = {bucket.images_dir}")
        print(f"   nmf    = {bucket.nmf_dir}")
        items, stats = collect_items(
            bucket,
            args.mse_threshold,
            args.kind,
            args.detection_group_field,
            excluded_identities,
        )
        print(
            f"   总图 {stats['total_images']}  |  "
            f"人工排除 {stats['manually_excluded']}  |  "
            f"无MSE记录 {stats['no_mse_record']}  |  "
            f"MSE超阈值 {stats['mse_over_threshold']}  |  "
            f"缺NMF缓存 {stats['missing_nmf_cache']}  |  "
            f"无mask {stats['no_mask']}  |  保留 {stats['kept']}"
        )
        if args.kind == "detection":
            print(
                f"   检测配套缺失: annotation={stats['missing_annotation']}  "
                f"ignore_mask={stats['missing_ignore_mask']}  "
                f"review_png={stats['missing_review_visualization']}"
            )
        if not items:
            print(f"   [warn] 桶 '{label}' 没有可用样本，跳过")
            manifest["buckets"][label] = {"stats": stats, "splits": {}}
            continue

        if args.kind == "detection":
            splits = split_detection_grouped_items(
                items,
                args.pretrain_ratio,
                args.finetune_val_ratio,
                args.finetune_test_ratio,
                args.seed,
            )
        elif args.kind == "classification" and args.classification_group_regex:
            splits = split_classification_grouped_items(
                items,
                args.pretrain_ratio,
                args.finetune_val_ratio,
                args.finetune_test_ratio,
                args.seed,
                args.classification_group_regex,
            )
        else:
            splits = split_bucket_items(
                items,
                args.pretrain_ratio,
                args.finetune_val_ratio,
                args.finetune_test_ratio,
                args.seed,
            )
        print(
            f"   → 预训练 {len(splits['pretrain'])}  |  "
            f"微调train {len(splits['finetune_train'])}  |  "
            f"微调val {len(splits['finetune_val'])}  |  "
            f"微调test {len(splits['finetune_test'])}"
        )
        bucket_manifest: dict[str, object] = {
            "stats": stats,
            "splits": {k: [it.stem for it in v] for k, v in splits.items()},
        }
        if args.kind == "detection":
            bucket_manifest["detection_split_stats"] = {
                split_name: {
                    "groups": sorted(
                        {it.detection_group for it in split_items if it.detection_group}
                    ),
                    "images": len(split_items),
                    "ordinary_boxes": sum(it.ordinary_count for it in split_items),
                    "category_counts": {
                        str(category_id): sum(
                            it.category_counts.get(category_id, 0) for it in split_items
                        )
                        for category_id in sorted(
                            {
                                category_id
                                for it in items
                                for category_id in it.category_counts
                            }
                        )
                    },
                }
                for split_name, split_items in splits.items()
            }
        manifest["buckets"][label] = bucket_manifest
        for key, lst in splits.items():
            all_splits[key].extend(lst)
        total_kept += stats["kept"]

    print("\n" + "=" * 78)
    print(f"汇总（全集 {total_kept} 项，人工排除与 MSE 过滤后）:")
    for key, lst in all_splits.items():
        tag = format_ratio_tag(len(lst), total_kept) if total_kept else "p000"
        print(f"  {key:<16}: {len(lst):5d}  ({tag})")
    print("=" * 78)

    if gen_pretrain and not all_splits["pretrain"]:
        print("[warn] 预训练子集为空，跳过 pretrain 目录生成")
        gen_pretrain = False
    if gen_finetune:
        finetune_total = (
            len(all_splits["finetune_train"])
            + len(all_splits["finetune_val"])
            + len(all_splits["finetune_test"])
        )
        if finetune_total == 0:
            print("[warn] 微调子集为空，跳过 finetune 目录生成")
            gen_finetune = False
        elif args.pretrain_ratio <= 0.0 and all_splits["pretrain"]:
            print(
                f"[warn] PRETRAIN_RATIO=0 但仍有 {len(all_splits['pretrain'])} 项"
                f"无 mask 无法进入微调，这些样本不会写入任何目录"
            )

    pretrain_root: Path | None = None
    if gen_pretrain:
        pretrain_root = build_output_dir(
            data_root, "pretrain", len(all_splits["pretrain"]), total_kept, run_date,
        )
        print(f"\n生成 {pretrain_root} ...")
        materialize_pretrain(
            all_splits["pretrain"],
            pretrain_root,
            nmf_dir_name,
            args.dry_run,
        )
        link_wavelength_file(data_root, buckets, pretrain_root, args.dry_run)

    finetune_roots: dict[str, Path] = {}
    if gen_finetune:
        finetune_roles = {
            "finetune_train": "finetune_train",
            "finetune_val": "finetune_val",
            "finetune_test": "finetune_test",
        }
        for split_key, role in finetune_roles.items():
            count = len(all_splits[split_key])
            if count == 0:
                print(f"[warn] {split_key} 为空，跳过目录生成")
                continue
            out_root = build_output_dir(data_root, role, count, total_kept, run_date)
            finetune_roots[split_key] = out_root
            print(f"生成 {out_root} ...")
            materialize_finetune(
                all_splits[split_key],
                out_root,
                args.kind,
                nmf_dir_name,
                args.dry_run,
            )
            if args.kind in ("classification", "detection"):
                link_wavelength_file(data_root, buckets, out_root, args.dry_run)

    manifest["summary"] = {k: len(v) for k, v in all_splits.items()}
    manifest["total_kept"] = total_kept
    manifest["run_date"] = run_date
    excluded_by_class: dict[str, int] = {}
    for class_name, _ in sorted(excluded_identities):
        excluded_by_class[class_name] = excluded_by_class.get(class_name, 0) + 1
    manifest["manual_exclusion"] = {
        "file": str(Path(args.exclude_json)) if args.exclude_json else None,
        "requested_unique_samples": len(excluded_identities),
        "matched_samples": len(excluded_identities),
        "excluded_by_class": excluded_by_class,
    }
    manifest["output_dirs"] = {}
    if pretrain_root is not None:
        manifest["output_dirs"]["pretrain"] = str(pretrain_root)
    manifest["output_dirs"].update(
        {k: str(v) for k, v in finetune_roots.items()}
    )
    manifest["args"] = {
        "data_root": str(data_root),
        "kind": args.kind,
        "nmf_dir_name": nmf_dir_name,
        "mse_threshold": args.mse_threshold,
        "pretrain_ratio": args.pretrain_ratio,
        "finetune_val_ratio": args.finetune_val_ratio,
        "finetune_test_ratio": args.finetune_test_ratio,
        "seed": args.seed,
        "run_date": run_date,
        "classification_group_regex": args.classification_group_regex,
        "detection_group_field": args.detection_group_field,
        "exclude_json": str(Path(args.exclude_json)) if args.exclude_json else None,
    }
    if args.manifest_out:
        manifest_path = Path(args.manifest_out)
    else:
        manifest_path = Path(f"{data_root}_split_manifest_{run_date}.json")
    if not args.dry_run:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nmanifest 已保存: {manifest_path}")
    else:
        print(f"\n[dry-run] manifest 未写入，若正式运行将保存到: {manifest_path}")

    print("\n完成。")
    if pretrain_root is not None:
        print(f"  预训练目录:      {pretrain_root}")
    finetune_labels = {
        "finetune_train": "微调训练目录",
        "finetune_val": "微调验证目录",
        "finetune_test": "微调测试目录",
    }
    for split_key, label in finetune_labels.items():
        if split_key in finetune_roots:
            print(f"  {label}:    {finetune_roots[split_key]}")
    if pretrain_root or finetune_roots:
        print("  各目录均以软链接组织所需数据与 nmf_cache_xxx/；原始文件未复制")


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "按 NMF 重建误差过滤 + 比例切分数据集为 "
            "_pretrain / _finetune_{train,val,test}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="数据集根目录（分类数据集为含各类别子目录的根）",
    )
    p.add_argument(
        "--kind",
        choices=("segmentation", "classification", "detection"),
        required=True,
    )
    p.add_argument(
        "--class-dirs",
        type=str,
        nargs="*",
        default=None,
        help=(
            "分类数据集的类别子目录名，留空则自动探测 data-root 下"
            "所有含 images/ 的直接子目录"
        ),
    )
    p.add_argument(
        "--classification-group-regex",
        type=str,
        default=None,
        help=(
            "仅分类模式生效：从文件 stem 提取患者/切片/ROI group_id 的正则；"
            "优先读取命名组 (?P<group>...)，否则读取第一个捕获组。"
            "同一 group 的全部 patch 将进入同一 split"
        ),
    )
    p.add_argument(
        "--exclude-json",
        type=str,
        default=None,
        help=(
            "仅分类模式生效：在任何 split 生成之前，按顶层 JSON list 中的 "
            "(class_name, stem) 精确排除人工确认无效的样本"
        ),
    )
    p.add_argument(
        "--detection-group-field",
        type=str,
        default="source_stem",
        help=(
            "仅检测模式生效：逐 Patch JSON 的 images[0] 中用于防数据泄漏的"
            "来源分组字段；默认 source_stem"
        ),
    )

    # NMF 缓存键（与 run_offline_nmf.sh / offline_nmf.py 参数命名保持一致）
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--l1", type=float, default=5e-4)
    p.add_argument("--l2", type=float, default=2e-4)
    p.add_argument("--l3", type=float, default=1e-2)
    p.add_argument(
        "--use-simplex",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--no-use-simplex",
        dest="use_simplex",
        action="store_false",
    )
    p.add_argument("--lam-e", type=float, default=0.05)
    p.add_argument("--e-clamp-max", type=float, default=3.0)

    p.add_argument(
        "--mse-threshold",
        type=float,
        required=True,
        help="重建 MSE 超过此值的样本剔除",
    )
    p.add_argument(
        "--pretrain-ratio",
        type=float,
        default=0.3,
        help="预训练样本占（MSE过滤后）总量的比例",
    )
    p.add_argument(
        "--finetune-val-ratio",
        type=float,
        default=0.15,
        help="微调集合中划入 val 的比例",
    )
    p.add_argument(
        "--finetune-test-ratio",
        type=float,
        default=0.15,
        help="微调集合中划入 test 的比例",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--run-date",
        type=str,
        default=None,
        help="目录名日期后缀 YYYYMMDD，默认取脚本启动当日（同一次运行内固定）",
    )
    p.add_argument(
        "--manifest-out",
        type=str,
        default=None,
        help="切分清单 json 输出路径，默认 {data_root}_split_manifest_{date}.json",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计与打印，不创建任何软链接/manifest",
    )

    args = p.parse_args()
    if not (0.0 <= args.pretrain_ratio <= 1.0):
        p.error("--pretrain-ratio 必须在 [0, 1] 之间")
    if args.run_date is not None and (
        len(args.run_date) != 8 or not args.run_date.isdigit()
    ):
        p.error("--run-date 必须为 YYYYMMDD 格式")
    if args.pretrain_ratio < 1.0 and (
        args.finetune_val_ratio + args.finetune_test_ratio >= 1.0
    ):
        p.error("--finetune-val-ratio + --finetune-test-ratio 必须小于 1")
    if args.classification_group_regex and args.kind != "classification":
        p.error("--classification-group-regex 仅可用于 --kind classification")
    if args.exclude_json and args.kind != "classification":
        p.error("--exclude-json 目前仅可用于 --kind classification")
    if not args.detection_group_field.strip():
        p.error("--detection-group-field 不能为空")
    if args.classification_group_regex:
        try:
            re.compile(args.classification_group_regex)
        except re.error as exc:
            p.error(f"--classification-group-regex 不是有效正则: {exc}")
    run(args)


if __name__ == "__main__":
    main()
