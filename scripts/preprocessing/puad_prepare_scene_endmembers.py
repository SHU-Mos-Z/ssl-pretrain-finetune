"""Prepare normalized scene-level E* caches for the PUAD official-like split.

This is an optional experiment artifact.  The historical patch-level NMF
caches remain untouched and are still the default fine-tuning input.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.physics.beer_lambert import intensity_to_od_np
from utils.preprocessing.offline_nmf_cuda import run_offline_nmf_on_cube


_ENVI_DTYPES = {
    1: np.dtype("u1"),
    2: np.dtype("i2"),
    3: np.dtype("i4"),
    4: np.dtype("f4"),
    5: np.dtype("f8"),
    12: np.dtype("u2"),
    13: np.dtype("u4"),
    14: np.dtype("i8"),
    15: np.dtype("u8"),
}


def _read_envi_hws(hdr_path: Path, raw_path: Path) -> np.ndarray:
    """Read an ENVI cube as an HWS memmap without requiring spectral-python."""

    text = hdr_path.read_text(encoding="utf-8", errors="replace")

    def scalar(name: str) -> str:
        match = re.search(
            rf"(?im)^\s*{re.escape(name)}\s*=\s*([^\r\n]+)", text
        )
        if match is None:
            raise ValueError(f"{hdr_path}: missing ENVI field {name!r}")
        return match.group(1).strip().strip("{} ")

    lines = int(scalar("lines"))
    samples = int(scalar("samples"))
    bands = int(scalar("bands"))
    data_type = int(scalar("data type"))
    header_offset = int(scalar("header offset"))
    byte_order = int(scalar("byte order"))
    interleave = scalar("interleave").lower()
    if data_type not in _ENVI_DTYPES:
        raise ValueError(f"{hdr_path}: unsupported ENVI data type {data_type}")
    dtype = _ENVI_DTYPES[data_type].newbyteorder("<" if byte_order == 0 else ">")
    if interleave == "bip":
        return np.memmap(
            raw_path, dtype=dtype, mode="r", offset=header_offset,
            shape=(lines, samples, bands),
        )
    if interleave == "bil":
        raw = np.memmap(
            raw_path, dtype=dtype, mode="r", offset=header_offset,
            shape=(lines, bands, samples),
        )
        return raw.transpose(0, 2, 1)
    if interleave == "bsq":
        raw = np.memmap(
            raw_path, dtype=dtype, mode="r", offset=header_offset,
            shape=(bands, lines, samples),
        )
        return raw.transpose(1, 2, 0)
    raise ValueError(f"{hdr_path}: unsupported ENVI interleave {interleave!r}")


def _scene_sources(dataset_root: Path) -> list[tuple[str, Path, Path]]:
    sources: dict[str, tuple[Path, Path]] = {}
    for split in ("train", "val", "test"):
        manifest = dataset_root / split / "manifest.csv"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                hdr = Path(row["source_hdr"])
                raw = Path(row["source_raw"])
                scene_stem = hdr.stem
                previous = sources.get(scene_stem)
                if previous is not None and previous != (hdr, raw):
                    raise ValueError(f"conflicting source paths for {scene_stem}")
                sources[scene_stem] = (hdr, raw)
    return [(stem, *sources[stem]) for stem in sorted(sources)]


def _load_normalization(dataset_root: Path) -> tuple[float, float]:
    path = dataset_root / "normalization_params.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    minimum = float(payload["global_min"])
    maximum = float(payload["global_max"])
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        raise ValueError(f"invalid normalization range in {path}")
    return minimum, maximum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="data/LUAD_PUAD_official224_centerbalanced_3660",
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--l1", type=float, default=5e-4)
    parser.add_argument("--l2", type=float, default=2e-4)
    parser.add_argument("--l3", type=float, default=1e-2)
    parser.add_argument("--lam-e", type=float, default=0.05)
    parser.add_argument("--e-clamp-max", type=float, default=3.0)
    parser.add_argument("--od-max", type=float, default=3.0)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else dataset_root / (
            f"scene_endmembers_K{args.k}_l1{args.l1:g}_l2{args.l2:g}_"
            f"l3{args.l3:g}_le{args.lam_e:g}_ec{args.e_clamp_max:g}_simplex"
        )
    )
    minimum, maximum = _load_normalization(dataset_root)
    sources = _scene_sources(dataset_root)
    if args.max_scenes > 0:
        sources = sources[: args.max_scenes]
    if args.dry_run:
        print(
            f"Validated {len(sources)} unique scene source pairs; "
            f"normalization=[{minimum}, {maximum}], output={output_root}"
        )
        return
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for index, (stem, hdr_path, raw_path) in enumerate(
        tqdm(sources, desc="PUAD scene-level NMF")
    ):
        output_path = output_root / f"{stem}_E.npy"
        if output_path.is_file() and output_path.stat().st_size > 0 and not args.overwrite:
            records.append({"stem": stem, "status": "cached", "e_path": str(output_path)})
            continue
        if not hdr_path.is_file() or not raw_path.is_file():
            raise FileNotFoundError(f"missing ENVI pair: {hdr_path}, {raw_path}")
        cube_hws = np.asarray(_read_envi_hws(hdr_path, raw_path), dtype=np.float32)
        normalized = np.clip(
            (cube_hws - minimum) / (maximum - minimum), 0.0, 1.0
        )
        od = np.clip(
            intensity_to_od_np(normalized.transpose(2, 0, 1)), 0.0, args.od_max
        ).astype(np.float32)
        result = run_offline_nmf_on_cube(
            od,
            k=args.k,
            lam1=args.l1,
            lam2=args.l2,
            lam3=args.l3,
            max_iter=args.max_iter,
            seed=args.seed + index,
            use_simplex=True,
            lam_e=args.lam_e,
            e_clamp_max=args.e_clamp_max,
            track_objective=False,
            compute_stats=False,
            device=args.device,
            dtype=args.dtype,
        )
        np.save(output_path, result.e_star.astype(np.float32))
        records.append(
            {
                "stem": stem,
                "status": "generated",
                "source_hdr": str(hdr_path),
                "source_raw": str(raw_path),
                "e_path": str(output_path),
                "shape": list(result.e_star.shape),
                "reconstruction_metrics": result.metrics,
                "n_iter": result.n_iter,
            }
        )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "normalization": {"global_min": minimum, "global_max": maximum},
        "nmf": {
            "k": args.k,
            "l1": args.l1,
            "l2": args.l2,
            "l3": args.l3,
            "simplex": True,
            "lam_e": args.lam_e,
            "e_clamp_max": args.e_clamp_max,
            "od_max": args.od_max,
            "max_iter": args.max_iter,
            "seed": args.seed,
            "device": args.device,
            "dtype": args.dtype,
        },
        "records": records,
    }
    (output_root / "scene_endmember_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(records)} scene endmember records to {output_root}")


if __name__ == "__main__":
    main()
