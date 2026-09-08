#!/usr/bin/env python3
"""Replace image/mask symlinks under data/ with real file copies."""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"

IMAGE_MASK_DIR_SUFFIXES = ("/images", "/masks")


def is_image_mask_dir(path: Path) -> bool:
    s = path.as_posix()
    return s.endswith("/images") or s.endswith("/masks")


def collect_symlink_dirs(data_root: Path) -> dict[str, list[Path]]:
    by_dir: dict[str, list[Path]] = defaultdict(list)
    for p in data_root.rglob("*"):
        if p.is_symlink() and is_image_mask_dir(p.parent):
            by_dir[str(p.parent)].append(p)
    return dict(by_dir)


def materialize_symlink(link: Path, *, dry_run: bool = False) -> tuple[str, str | None]:
    if not link.is_symlink():
        return "skip_not_symlink", None

    try:
        target = link.resolve()
    except OSError as exc:
        return "broken_resolve", str(exc)

    if not target.is_file():
        return "missing_target", str(target)

    if dry_run:
        return "would_copy", str(target)

    tmp = link.with_name(link.name + ".materialize_tmp")
    try:
        if tmp.exists():
            tmp.unlink()
        shutil.copy2(target, tmp)
        link.unlink()
        tmp.rename(link)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return "copied", str(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="only report, do not copy")
    parser.add_argument(
        "--dir",
        action="append",
        dest="dirs",
        help="limit to specific images/masks directory under data/",
    )
    args = parser.parse_args()

    if args.dirs:
        selected = {str((DATA_ROOT / d).resolve() if not Path(d).is_absolute() else Path(d)) for d in args.dirs}
        by_dir = {d: [p for p in Path(d).iterdir() if p.is_symlink()] for d in selected if Path(d).is_dir()}
    else:
        by_dir = collect_symlink_dirs(DATA_ROOT)

    if not by_dir:
        print("No image/mask symlinks found.")
        return 0

    print("=" * 80)
    print("Image/Mask symlink directories:")
    total = 0
    for d in sorted(by_dir):
        n = len(by_dir[d])
        total += n
        sample = by_dir[d][0]
        try:
            sample_target = sample.resolve()
        except OSError:
            sample_target = sample.readlink()
        print(f"  [{n:5d}] {d}")
        print(f"         -> {sample_target.parent}")
    print(f"Total symlinks: {total}")
    print("=" * 80)

    stats: dict[str, int] = defaultdict(int)
    errors: list[tuple[str, str, str]] = []

    for d in sorted(by_dir):
        links = sorted(by_dir[d], key=lambda p: p.name)
        print(f"\nProcessing {d} ({len(links)} files)...")
        for i, link in enumerate(links, start=1):
            status, detail = materialize_symlink(link, dry_run=args.dry_run)
            stats[status] += 1
            if status in {"missing_target", "broken_resolve"}:
                errors.append((str(link), status, detail or ""))
            if i % 200 == 0 or i == len(links):
                print(f"  progress {i}/{len(links)}")

    print("\n" + "=" * 80)
    print("Summary:")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for link, status, detail in errors[:20]:
            print(f"  {status}: {link} -> {detail}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
    print("=" * 80)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
