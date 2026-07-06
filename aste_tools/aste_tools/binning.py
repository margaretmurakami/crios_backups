"""T-S binning utilities used by ASTE budget diagnostics."""

from __future__ import annotations

import numpy as np


def bin_array(values: np.ndarray, edges: np.ndarray, *, right: bool = False, invalid: int = -1) -> np.ndarray:
    """Return zero-based bin indices for ``values``.

    Values outside ``edges`` or equal to NaN are set to ``invalid``. This is safer
    for downstream ``np.add.at`` calls than the notebook-era helper, which could
    produce negative or out-of-range indices for missing values.
    """

    arr = np.asarray(values)
    edge_arr = np.asarray(edges)
    if edge_arr.ndim != 1 or edge_arr.size < 2:
        raise ValueError("edges must be a one-dimensional array with at least two values")
    idx = np.digitize(arr, edge_arr, right=right) - 1
    idx = np.where(np.isfinite(arr) & (idx >= 0) & (idx < edge_arr.size - 1), idx, invalid)
    return idx.astype(int)


def accumulate_ts_mesh(
    values: np.ndarray,
    salinity_bins: np.ndarray,
    theta_bins: np.ndarray,
    *,
    n_salinity: int,
    n_theta: int,
    point_axis: bool = False,
    scale: float = 1.0,
) -> np.ndarray:
    """Accumulate ``values`` onto a salinity-theta grid.

    Parameters
    ----------
    values, salinity_bins, theta_bins
        Arrays with matching shape. Bin arrays should contain zero-based indices
        and use negative values for invalid cells.
    n_salinity, n_theta
        Number of salinity and potential-temperature bins.
    point_axis
        If True, preserve a final point axis and return ``(nS, nT, npoints)``.
        This mirrors the old ``create_TS_mesh`` notebook helper.
    scale
        Multiplicative factor applied after accumulation, for example
        ``1 / (dT * dS)``.
    """

    val = np.asarray(values)
    sb = np.asarray(salinity_bins)
    tb = np.asarray(theta_bins)
    if val.shape != sb.shape or val.shape != tb.shape:
        raise ValueError(f"values, salinity_bins, and theta_bins must match; got {val.shape}, {sb.shape}, {tb.shape}")

    valid = np.isfinite(val) & (sb >= 0) & (sb < n_salinity) & (tb >= 0) & (tb < n_theta)
    if point_axis:
        if val.ndim == 0:
            raise ValueError("point_axis=True requires at least one-dimensional input")
        npoints = val.shape[-1]
        out = np.zeros((n_salinity, n_theta, npoints), dtype=float)
        point_ids = np.broadcast_to(np.arange(npoints), val.shape)
        np.add.at(out, (sb[valid].astype(int), tb[valid].astype(int), point_ids[valid]), val[valid])
    else:
        out = np.zeros((n_salinity, n_theta), dtype=float)
        np.add.at(out, (sb[valid].astype(int), tb[valid].astype(int)), val[valid])
    return out * scale


def create_ts_mesh(
    values_by_time: np.ndarray,
    salinity_bins_by_time: np.ndarray,
    theta_bins_by_time: np.ndarray,
    *,
    n_salinity: int,
    n_theta: int,
    d_salinity: float = 1.0,
    d_theta: float = 1.0,
    point_axis: bool = False,
) -> np.ndarray:
    """Create a time series of T-S meshes from binned fields."""

    values = np.asarray(values_by_time)
    salinity = np.asarray(salinity_bins_by_time)
    theta = np.asarray(theta_bins_by_time)
    if values.shape != salinity.shape or values.shape != theta.shape:
        raise ValueError("input arrays must have matching shapes")
    scale = 1.0 / (d_salinity * d_theta)
    meshes = [
        accumulate_ts_mesh(v, s, t, n_salinity=n_salinity, n_theta=n_theta, point_axis=point_axis, scale=scale)
        for v, s, t in zip(values, salinity, theta)
    ]
    return np.stack(meshes, axis=0)


def calc_g_term(
    tendency: np.ndarray,
    theta_bins: np.ndarray,
    salinity_bins: np.ndarray,
    *,
    n_theta: int,
    n_salinity: int,
    sv_scale: float = 1e-6,
) -> np.ndarray:
    """Accumulate a tendency term into G-space and scale from m^3/s to Sv."""

    out = accumulate_ts_mesh(
        tendency,
        salinity_bins,
        theta_bins,
        n_salinity=n_salinity,
        n_theta=n_theta,
        scale=sv_scale,
    )
    out[out == 0] = np.nan
    return out

