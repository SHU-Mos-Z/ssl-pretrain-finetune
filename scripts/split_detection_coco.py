#!/usr/bin/env python3
"""Create deterministic group-aware train/validation/test COCO JSON files."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--prefix", default="instances_train_val_test")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--group-field",
        default="source_stem",
        help="COCO image field kept intact across splits; use image_id for per-image splitting",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(value <= 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        parser.error("train/val/test ratios must be positive and sum to 1")
    return args


def _group_key(image: dict, group_field: str) -> str:
    if group_field == "image_id":
        return str(image["id"])
    value = image.get(group_field)
    if value is None:
        raise KeyError(f"image_id={image['id']} lacks group field {group_field!r}")
    return str(value)


def split_payload(payload: dict, ratios: tuple[float, float, float], group_field: str, seed: int):
    annotations_by_image = defaultdict(list)
    for annotation in payload.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    groups: dict[str, dict] = {}
    for image in payload.get("images", []):
        key = _group_key(image, group_field)
        record = groups.setdefault(key, {"images": [], "ordinary": 0})
        record["images"].append(image)
        record["ordinary"] += sum(
            not bool(annotation.get("ignore", 0)) and not bool(annotation.get("iscrowd", 0))
            for annotation in annotations_by_image[int(image["id"])]
        )
    if len(groups) < 3:
        raise ValueError("at least three independent groups are required")

    randomizer = random.Random(seed)
    items = list(groups.items())
    randomizer.shuffle(items)
    items.sort(
        key=lambda item: (item[1]["ordinary"], len(item[1]["images"])), reverse=True
    )
    names = ("train", "val", "test")
    target_images = [len(payload["images"]) * ratio for ratio in ratios]
    target_objects = [
        sum(record["ordinary"] for record in groups.values()) * ratio for ratio in ratios
    ]
    assigned = {name: [] for name in names}
    current_images = [0, 0, 0]
    current_objects = [0, 0, 0]
    # Seed each split once so small datasets cannot collapse entirely into train.
    for split_index, (group_name, record) in enumerate(items[:3]):
        assigned[names[split_index]].append(group_name)
        current_images[split_index] += len(record["images"])
        current_objects[split_index] += record["ordinary"]
    for group_name, record in items[3:]:
        # Squared relative deficit combines image and ordinary-instance balance.
        scores = []
        for index in range(3):
            next_images = current_images[index] + len(record["images"])
            next_objects = current_objects[index] + record["ordinary"]
            score = (next_images / max(target_images[index], 1.0)) ** 2
            score += (next_objects / max(target_objects[index], 1.0)) ** 2
            scores.append(score)
        split_index = min(range(3), key=scores.__getitem__)
        assigned[names[split_index]].append(group_name)
        current_images[split_index] += len(record["images"])
        current_objects[split_index] += record["ordinary"]

    outputs = {}
    for name in names:
        group_set = set(assigned[name])
        images = [
            image for image in payload["images"] if _group_key(image, group_field) in group_set
        ]
        image_ids = {int(image["id"]) for image in images}
        annotations = [
            annotation
            for annotation in payload.get("annotations", [])
            if int(annotation["image_id"]) in image_ids
        ]
        outputs[name] = {
            key: value
            for key, value in payload.items()
            if key not in {"images", "annotations"}
        }
        outputs[name]["images"] = images
        outputs[name]["annotations"] = annotations
    return outputs, assigned


def main() -> None:
    args = get_args()
    input_path = Path(args.input_json).expanduser().resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    outputs, groups = split_payload(
        payload,
        (args.train_ratio, args.val_ratio, args.test_ratio),
        args.group_field,
        args.seed,
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": str(input_path),
        "seed": args.seed,
        "group_field": args.group_field,
        "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "splits": {},
    }
    for name, output in outputs.items():
        path = output_dir / f"{args.prefix}_{name}.json"
        path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        ordinary = sum(
            not bool(item.get("ignore", 0)) and not bool(item.get("iscrowd", 0))
            for item in output["annotations"]
        )
        manifest["splits"][name] = {
            "file": path.name,
            "groups": groups[name],
            "num_groups": len(groups[name]),
            "num_images": len(output["images"]),
            "num_annotations": len(output["annotations"]),
            "num_ordinary_annotations": ordinary,
        }
        print(
            f"[{name}] groups={len(groups[name])} images={len(output['images'])} "
            f"annotations={len(output['annotations'])} ordinary={ordinary} -> {path}"
        )
    manifest_path = output_dir / f"{args.prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
