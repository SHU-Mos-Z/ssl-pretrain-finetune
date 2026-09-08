#!/usr/bin/env python3
"""Launch full-dataset SAM3 text-based WBC candidate generation.

This script intentionally stops at *candidate* masks and provisional mask-derived
boxes.  It never exports final detection ground truth.  Human review and the
subsequent spectral filtering stage operate on its output.

Run from the project root, for example::

    python scripts/generate_wbc_sam3_text_candidates.py

Per-source results make the run resumable.  Re-running the same command skips
completed source images unless ``--overwrite`` is specified.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


DEFAULT_DATASET_NAME = (
    "2018WBC_cellcrop_512x512_noresize_contiguous20_entropy_"
    "multicandidate_manualoverride_filtered_minmax_20260830_2233"
)
DEFAULT_TEXT_PROMPTS = (
    "white blood cell",
    "leukocyte",
    "large purple cell",
    "cell with dark nucleus",
    "nucleated blood cell",
)


def project_root_from_script() -> Path:
    root = Path(__file__).resolve().parents[1]
    if not (root / "data/original").is_dir():
        raise FileNotFoundError(f"Cannot identify project root from script path: {root}")
    return root


def parse_args() -> argparse.Namespace:
    project_root = project_root_from_script()
    data_root = project_root / "data" / DEFAULT_DATASET_NAME
    parser = argparse.ArgumentParser(
        description="Generate high-recall, single-class WBC candidates with SAM3 text prompts."
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--data-root", type=Path, default=data_root)
    parser.add_argument(
        "--raw-root", type=Path, default=project_root / "data/original/2018WBC"
    )
    parser.add_argument("--sam3-repo", type=Path, default=project_root / "sam3")
    parser.add_argument(
        "--checkpoint", type=Path, default=project_root / "sam3/checkpoints/sam3.pt"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=data_root / "sam_prompt_experiments/sam3_text_all_wbc",
    )
    parser.add_argument("--conda-env", default="sam3")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--confidence-threshold", type=float, default=0.10)
    parser.add_argument("--dedup-mask-iou", type=float, default=0.80)
    parser.add_argument(
        "--text-prompt",
        action="append",
        dest="text_prompts",
        help="Repeat this option to replace the default prompt list.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Debug only. Omit to process every source listed in crop_coord_prompts.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute sources that already have per-source result JSON files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the child command without loading SAM3.",
    )
    return parser.parse_args()


def resolve_sam3_python(conda_env: str) -> Path:
    conda_executable = shutil.which("conda")
    if conda_executable is None:
        raise FileNotFoundError("Cannot find conda in PATH")
    conda_base = Path(
        subprocess.check_output(
            [conda_executable, "info", "--base"], text=True
        ).strip()
    )
    python_path = conda_base / f"envs/{conda_env}/bin/python"
    if not python_path.is_file():
        raise FileNotFoundError(
            f"Python executable for conda environment {conda_env!r} is missing: "
            f"{python_path}"
        )
    return python_path


def validate_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    args.project_root = args.project_root.resolve()
    args.data_root = args.data_root.resolve()
    args.raw_root = args.raw_root.resolve()
    args.sam3_repo = args.sam3_repo.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_root = args.output_root.resolve()
    helper = args.project_root / "data/original/wbc_sam3_text_candidates.py"
    required = (
        helper,
        args.data_root / "crop_coord_prompts.json",
        args.raw_root,
        args.sam3_repo / "sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        args.checkpoint,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required input is missing: {path}")
    if not 0.0 < args.confidence_threshold < 1.0:
        raise ValueError("--confidence-threshold must be in (0, 1)")
    if not 0.0 < args.dedup_mask_iou <= 1.0:
        raise ValueError("--dedup-mask-iou must be in (0, 1]")
    if args.max_images is not None and args.max_images < 1:
        raise ValueError("--max-images must be >= 1")
    return helper, resolve_sam3_python(args.conda_env)


def build_child_command(
    args: argparse.Namespace, helper: Path, sam3_python: Path
) -> list[str]:
    command = [
        str(sam3_python),
        str(helper),
        "--data-root",
        str(args.data_root),
        "--raw-root",
        str(args.raw_root),
        "--repo-root",
        str(args.sam3_repo),
        "--checkpoint",
        str(args.checkpoint),
        "--output-root",
        str(args.output_root),
        "--device",
        args.device,
        "--confidence-threshold",
        str(args.confidence_threshold),
        "--dedup-mask-iou",
        str(args.dedup_mask_iou),
        "--no-use-positive-exemplar",
        "--overwrite-existing" if args.overwrite else "--no-overwrite-existing",
    ]
    for prompt in args.text_prompts or DEFAULT_TEXT_PROMPTS:
        command.extend(("--text-prompt", prompt))
    if args.max_images is not None:
        command.extend(("--max-images", str(args.max_images)))
    return command


def main() -> None:
    args = parse_args()
    helper, sam3_python = validate_inputs(args)
    command = build_child_command(args, helper, sam3_python)
    source_manifest = json.loads(
        (args.data_root / "crop_coord_prompts.json").read_text(encoding="utf-8")
    )
    requested_sources = (
        min(len(source_manifest), args.max_images)
        if args.max_images is not None
        else len(source_manifest)
    )
    print("=" * 78)
    print("SAM3 high-recall single-class WBC candidate generation")
    print(f"Source manifest:       {args.data_root / 'crop_coord_prompts.json'}")
    print(f"Requested sources:     {requested_sources}")
    print(f"Output root:           {args.output_root}")
    print(f"Text prompts:          {args.text_prompts or list(DEFAULT_TEXT_PROMPTS)}")
    print(f"Confidence threshold:  {args.confidence_threshold}")
    print(f"Resume existing:       {not args.overwrite}")
    print("Positive box exemplar: disabled (a 512x512 crop is not a tight instance box)")
    print("Final GT export:       disabled; outputs are candidates only")
    print("=" * 78)
    if args.dry_run:
        print("Dry-run child command:")
        print(" ".join(command))
        return

    subprocess.run(command, cwd=args.project_root, check=True)
    print("\nCandidate generation completed or resumed successfully.")
    print(f"Candidate summary: {args.output_root / 'candidate_instances.json'}")
    print(f"Review template:   {args.output_root / 'manual_review_template.json'}")
    print(f"Review index:      {args.output_root / 'review_index.md'}")


if __name__ == "__main__":
    main()
