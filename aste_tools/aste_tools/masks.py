"""Mask and reduction helpers."""

from __future__ import annotations

import numpy as np


def wet_indices_in_mask(mask: np.ndarray, wet_flat_indices: np.ndarray) -> np.ndarray:
    """Return indices into ``wet_flat_indices`` whose cells are inside ``mask``."""

    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {arr.shape}")
    flat = arr.reshape(-1, order="C")
    wet = np.asarray(wet_flat_indices, dtype=int)
    in_mask = np.isfinite(flat) & (flat == 1)
    return np.flatnonzero(in_mask[wet])


def masked_area_mean(field: np.ndarray, mask: np.ndarray, area: np.ndarray, *, intensive: bool = True) -> tuple[float, float]:
    """Return an area-weighted mean/sum-like diagnostic and contributing area."""

    fld = np.asarray(field, dtype=float)
    msk = np.asarray(mask, dtype=float).copy()
    rac = np.asarray(area, dtype=float)
    if fld.ndim == 2:
        fld = fld[np.newaxis, :, :]
    if msk.ndim == 2:
        msk = np.broadcast_to(msk, fld.shape).copy()
    if msk.shape != fld.shape:
        raise ValueError(f"mask shape {msk.shape} is not compatible with field shape {fld.shape}")
    if rac.shape != fld.shape[-2:]:
        raise ValueError(f"area shape {rac.shape} must match horizontal field shape {fld.shape[-2:]}")

    msk[msk == 0] = np.nan
    msk[~np.isfinite(fld)] = np.nan
    area_mask = np.broadcast_to(rac, fld.shape) * msk
    total_area = float(np.nansum(area_mask))
    if intensive:
        value = float(np.nansum(fld * area_mask) / total_area)
    else:
        value = float(np.nansum(fld * msk) / total_area)
    return value, total_area


def expand_to_3d(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Broadcast a 1D vertical or 2D horizontal array to ``target`` shape."""

    arr = np.asarray(values)
    tgt = np.asarray(target)
    if tgt.ndim != 3:
        raise ValueError(f"target must be 3D, got shape {tgt.shape}")
    if arr.ndim == 2:
        if arr.shape != tgt.shape[-2:]:
            raise ValueError(f"2D values shape {arr.shape} does not match target horizontal shape {tgt.shape[-2:]}")
        return np.broadcast_to(arr, tgt.shape).copy()
    if arr.ndim == 1:
        if arr.shape[0] != tgt.shape[0]:
            raise ValueError(f"1D values length {arr.shape[0]} does not match target vertical size {tgt.shape[0]}")
        return np.broadcast_to(arr[:, np.newaxis, np.newaxis], tgt.shape).copy()
    if arr.shape == tgt.shape:
        return arr.copy()
    raise ValueError(f"cannot expand shape {arr.shape} to {tgt.shape}")

