#!/usr/bin/env python3
"""Prepare compact downstream-task image tiles for the framework/PPT figure.

The script deliberately separates tasks already supported by the repository
from the detection task that is still represented only by mask-derived
candidate boxes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "assets" / "downstream_tasks"

TILE_WIDTH = 360
TILE_HEIGHT = 480
PAD = 12
GAP = 8

FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(str(path), size=size)


def pseudo_rgb(cube: np.ndarray) -> np.ndarray:
    """Match the pseudo-RGB rule used by dataview_seg_label.ipynb."""
    if cube.ndim != 3:
        raise ValueError(f"Expected H x W x B cube, got {cube.shape}")
    bands = cube.shape[-1]
    indices = [int(bands * 0.75), int(bands * 0.50), int(bands * 0.25)]
    rgb = cube[..., indices].astype(np.float32)
    lo, hi = np.nanpercentile(rgb, [1, 99], axis=(0, 1), keepdims=True)
    return np.clip((rgb - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def rgb_image(path: Path, size: tuple[int, int]) -> Image.Image:
    cube = np.load(path)
    array = (pseudo_rgb(cube) * 255).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB").resize(size, Image.Resampling.LANCZOS)


def representative(class_dir: Path) -> Path:
    """Choose a deterministic, visually informative non-augmented sample."""
    paths = [
        path
        for path in sorted((class_dir / "images").glob("*.npy"))
        if "_flip" not in path.stem.lower()
    ]
    if not paths:
        raise FileNotFoundError(f"No samples found in {class_dir / 'images'}")

    sample_count = min(32, len(paths))
    sample_ids = np.linspace(0, len(paths) - 1, sample_count, dtype=int)
    candidates = [paths[index] for index in sample_ids]
    contrasts = np.asarray(
        [float(np.std(pseudo_rgb(np.load(path)))) for path in candidates]
    )
    target = float(np.quantile(contrasts, 0.75))
    return candidates[int(np.argmin(np.abs(contrasts - target)))]


def rounded_label(
    canvas: Image.Image,
    xy: tuple[int, int],
    text: str,
    fill: tuple[int, int, int],
) -> None:
    draw = ImageDraw.Draw(canvas)
    label_font = font(20, bold=True)
    box = draw.textbbox((0, 0), text, font=label_font)
    width = box[2] - box[0] + 18
    height = box[3] - box[1] + 10
    x, y = xy
    draw.rounded_rectangle((x, y, x + width, y + height), radius=7, fill=fill)
    draw.text((x + 9, y + 3), text, font=label_font, fill=(255, 255, 255))


def classification_tile(
    dataset_root: Path,
    class_names: list[str],
    output_name: str,
    columns: int,
) -> tuple[Path, dict[str, str]]:
    canvas = Image.new("RGBA", (TILE_WIDTH, TILE_HEIGHT), (255, 255, 255, 0))
    rows = int(np.ceil(len(class_names) / columns))
    cell_width = (TILE_WIDTH - 2 * PAD - (columns - 1) * GAP) // columns
    cell_height = (TILE_HEIGHT - 2 * PAD - (rows - 1) * GAP) // rows
    image_height = cell_height - 34
    selections: dict[str, str] = {}

    for index, class_name in enumerate(class_names):
        row, column = divmod(index, columns)
        x = PAD + column * (cell_width + GAP)
        y = PAD + row * (cell_height + GAP)
        selected = representative(dataset_root / class_name)
        selections[class_name] = str(selected.relative_to(ROOT))
        image = rgb_image(selected, (cell_width, image_height))
        canvas.alpha_composite(image.convert("RGBA"), (x, y))

        draw = ImageDraw.Draw(canvas)
        draw.rectangle(
            (x, y + image_height, x + cell_width, y + cell_height),
            fill=(247, 248, 250, 255),
        )
        draw.text(
            (x + 7, y + image_height + 4),
            class_name,
            font=font(19, bold=True),
            fill=(35, 39, 47, 255),
        )

    output = OUT_DIR / output_name
    canvas.save(output)
    return output, selections


def load_segmentation_pair(
    dataset_root: Path, stem: str
) -> tuple[np.ndarray, np.ndarray, Path, Path]:
    image_path = dataset_root / "images" / f"{stem}.npy"
    mask_path = dataset_root / "masks" / f"{stem}.npy"
    cube = np.load(image_path)
    mask = np.squeeze(np.load(mask_path))
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got {mask.shape}")
    return pseudo_rgb(cube), mask, image_path, mask_path


def color_mask(mask: np.ndarray) -> Image.Image:
    foreground = mask > 0
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    output[foreground] = np.asarray((35, 166, 92), dtype=np.uint8)
    return Image.fromarray(output, mode="RGB")


def segmentation_tile(
    dataset_root: Path,
    stem: str,
    output_name: str,
) -> tuple[Path, dict[str, str]]:
    rgb, mask, image_path, mask_path = load_segmentation_pair(dataset_root, stem)
    canvas = Image.new("RGBA", (TILE_WIDTH, TILE_HEIGHT), (255, 255, 255, 0))
    panel_height = (TILE_HEIGHT - 2 * PAD - GAP) // 2
    image_width = TILE_WIDTH - 2 * PAD

    input_image = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")
    input_image = input_image.resize(
        (image_width, panel_height), Image.Resampling.LANCZOS
    )
    mask_image = color_mask(mask).resize(
        (image_width, panel_height), Image.Resampling.NEAREST
    )
    canvas.alpha_composite(input_image.convert("RGBA"), (PAD, PAD))
    canvas.alpha_composite(
        mask_image.convert("RGBA"), (PAD, PAD + panel_height + GAP)
    )
    rounded_label(canvas, (PAD + 7, PAD + 7), "HSI", (36, 74, 124))
    rounded_label(
        canvas,
        (PAD + 7, PAD + panel_height + GAP + 7),
        "GT Mask",
        (35, 166, 92),
    )

    output = OUT_DIR / output_name
    canvas.save(output)
    return output, {
        "image": str(image_path.relative_to(ROOT)),
        "mask": str(mask_path.relative_to(ROOT)),
    }


def candidate_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    labels, count = ndimage.label(mask > 0, structure=np.ones((3, 3), dtype=np.uint8))
    boxes: list[tuple[int, tuple[int, int, int, int]]] = []
    for component_id in range(1, count + 1):
        ys, xs = np.where(labels == component_id)
        area = len(xs)
        if area < 64:
            continue
        boxes.append((area, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))))
    boxes.sort(key=lambda item: item[0], reverse=True)
    return [box for _, box in boxes[:12]]


def detection_tile(
    dataset_root: Path,
    stem: str,
    output_name: str,
) -> tuple[Path, dict[str, str | int]]:
    rgb, mask, image_path, mask_path = load_segmentation_pair(dataset_root, stem)
    boxes = candidate_boxes(mask)
    source = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")
    source = source.resize((TILE_WIDTH - 2 * PAD, TILE_WIDTH - 2 * PAD))

    scale_x = source.width / mask.shape[1]
    scale_y = source.height / mask.shape[0]
    draw = ImageDraw.Draw(source)
    colors = [(255, 195, 0), (226, 62, 87), (74, 144, 226), (126, 211, 33)]
    for index, (x0, y0, x1, y1) in enumerate(boxes):
        draw.rectangle(
            (
                int(x0 * scale_x),
                int(y0 * scale_y),
                int(x1 * scale_x),
                int(y1 * scale_y),
            ),
            outline=colors[index % len(colors)],
            width=4,
        )

    canvas = Image.new("RGBA", (TILE_WIDTH, TILE_HEIGHT), (255, 255, 255, 0))
    canvas.alpha_composite(source.convert("RGBA"), (PAD, PAD))
    rounded_label(canvas, (PAD + 7, PAD + 7), "Candidates", (132, 75, 173))
    footer_y = PAD + source.height + 13
    ImageDraw.Draw(canvas).text(
        (PAD, footer_y),
        "Mask-derived boxes",
        font=font(20, bold=True),
        fill=(80, 80, 86, 255),
    )

    output = OUT_DIR / output_name
    canvas.save(output)
    return output, {
        "image": str(image_path.relative_to(ROOT)),
        "mask": str(mask_path.relative_to(ROOT)),
        "box_count": len(boxes),
    }


def composite_preview(
    files: list[Path],
    task_titles: list[str],
    output_name: str,
    planned_last: bool = False,
) -> Path:
    column_width = 390
    title_height = 52
    footer_height = 52
    canvas = Image.new(
        "RGBA",
        (column_width * len(files), title_height + TILE_HEIGHT + footer_height),
        (255, 255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    for index, (path, title) in enumerate(zip(files, task_titles)):
        x = index * column_width
        tile = Image.open(path).convert("RGBA")
        canvas.alpha_composite(tile, (x + (column_width - TILE_WIDTH) // 2, title_height))
        title_font = font(22, bold=True)
        title_box = draw.textbbox((0, 0), title, font=title_font)
        title_x = x + (column_width - (title_box[2] - title_box[0])) // 2
        draw.text((title_x, 13), title, font=title_font, fill=(30, 35, 44, 255))
        if index < len(files) - 1:
            draw.line(
                (x + column_width - 1, 18, x + column_width - 1, canvas.height - 18),
                fill=(215, 218, 224, 255),
                width=2,
            )

    if planned_last:
        note = "* Candidate boxes from segmentation masks; detector is planned."
        draw.text(
            (column_width * 3 + 16, title_height + TILE_HEIGHT + 12),
            note,
            font=font(14),
            fill=(100, 75, 120, 255),
        )

    output = OUT_DIR / output_name
    canvas.save(output)
    return output


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {}

    wbc_file, wbc_sources = classification_tile(
        ROOT / "data" / "2018WBC_patch_650x600_overlap_0x0_to_256x256_minmax_bands50",
        ["B", "E", "L", "M", "N"],
        "wbc_cell_classification.png",
        columns=2,
    )
    manifest["wbc_cell_classification"] = wbc_sources

    plgc_file, plgc_sources = classification_tile(
        ROOT / "data" / "PLGC_class_patch_512x512_overlap_0x0_to_256x256_minmax",
        ["Normal", "IM", "CINI"],
        "plgc_tissue_classification.png",
        columns=2,
    )
    manifest["plgc_tissue_classification"] = plgc_sources

    gpcc_root = (
        ROOT
        / "data"
        / "GPCC_Resized_512_640_to_256_256_overlap_0_0_preprocessed"
    )
    gpcc_stem = "2021-83319-3-10x-roi13-mono_roi_0"
    gpcc_seg_file, gpcc_seg_sources = segmentation_tile(
        gpcc_root, gpcc_stem, "gpcc_semantic_segmentation.png"
    )
    manifest["gpcc_semantic_segmentation"] = gpcc_seg_sources

    detection_file, detection_sources = detection_tile(
        gpcc_root, gpcc_stem, "gpcc_candidate_detection.png"
    )
    manifest["gpcc_candidate_detection"] = detection_sources

    mdc_root = (
        ROOT
        / "data"
        / "MDC_Resized_256_320_to_256_256_overlap_0_0_preprocessed"
    )
    mdc_seg_file, mdc_seg_sources = segmentation_tile(
        mdc_root, "051417-20x-roi2_roi_0", "mdc_semantic_segmentation.png"
    )
    manifest["mdc_semantic_segmentation"] = mdc_seg_sources

    strict_preview = composite_preview(
        [wbc_file, plgc_file, gpcc_seg_file, mdc_seg_file],
        [
            "WBC Classification",
            "PLGC Classification",
            "GPCC Segmentation",
            "MDC Segmentation",
        ],
        "downstream_tasks_strict_preview.png",
    )
    planned_preview = composite_preview(
        [wbc_file, plgc_file, gpcc_seg_file, detection_file],
        [
            "WBC Classification",
            "PLGC Classification",
            "Semantic Segmentation",
            "Detection (Planned)*",
        ],
        "downstream_tasks_planned_preview.png",
        planned_last=True,
    )
    manifest["previews"] = {
        "strict": str(strict_preview.relative_to(ROOT)),
        "with_planned_detection": str(planned_preview.relative_to(ROOT)),
    }
    manifest["notes"] = {
        "classification": "Both classification datasets are single-label multi-class.",
        "detection": (
            "Boxes are connected components converted from GPCC semantic masks; "
            "they are not predictions of a trained detector."
        ),
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for path in sorted(OUT_DIR.iterdir()):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
