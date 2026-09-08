"""Dataset-level wavelength metadata loading."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

_WAVELENGTH_CACHE: dict[tuple[str, int, bool], np.ndarray] = {}


def load_wavelengths(
    data_root: str | Path,
    num_bands: int,
    wavelength_file: str | None = None,
    allow_index_fallback: bool = False,
) -> np.ndarray:
    path = Path(wavelength_file) if wavelength_file else Path(data_root) / "wavelengths.npy"
    cache_key = (str(path.resolve()), num_bands, allow_index_fallback)
    cached = _WAVELENGTH_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if path.is_file():
        wavelengths = np.load(path).astype(np.float32).reshape(-1)
        if wavelengths.size != num_bands:
            raise ValueError(f"{path} has {wavelengths.size} values, expected {num_bands}")
        if not np.all(np.diff(wavelengths) > 0):
            raise ValueError(f"wavelengths must be strictly increasing: {path}")
        _WAVELENGTH_CACHE[cache_key] = wavelengths
        return wavelengths
    if not allow_index_fallback:
        raise FileNotFoundError(
            f"missing wavelength table {path}; provide wavelengths.npy or enable index fallback"
        )
    warnings.warn(
        f"{path} is missing; using normalized band indices for {data_root}",
        RuntimeWarning,
        stacklevel=2,
    )
    wavelengths = np.linspace(0.0, 1.0, num_bands, dtype=np.float32)
    _WAVELENGTH_CACHE[cache_key] = wavelengths
    return wavelengths


def token_spectral_positions(
    wavelengths: np.ndarray, h_p: int, w_p: int, spectral_patch_size: int
) -> np.ndarray:
    if wavelengths.size % spectral_patch_size:
        raise ValueError("number of wavelengths must be divisible by spectral_patch_size")
    groups = wavelengths.reshape(-1, spectral_patch_size).mean(axis=1)
    lo, hi = float(wavelengths.min()), float(wavelengths.max())
    groups = (groups - lo) / max(hi - lo, 1e-6)
    return np.tile(groups, h_p * w_p).astype(np.float32)
