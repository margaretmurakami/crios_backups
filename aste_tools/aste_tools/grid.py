"""ASTE compact-grid reshaping utilities.

The original notebooks use functions such as ``get_aste_tracer`` directly from
``an_helper_functions``. This module provides stable, typed names around the same
grid layout so scripts and notebooks can share one import path.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np


@dataclass(frozen=True)
class ASTEGridSpec:
    """Face dimensions for an ASTE compact grid."""

    nx: int
    ncut1: int
    ncut2: int
    nz: int = 50
    name: str = "aste"

    @property
    def ny(self) -> int:
        return 2 * self.ncut1 + self.nx + self.ncut2

    @property
    def nfx(self) -> np.ndarray:
        return np.array([self.nx, 0, self.nx, self.ncut2, self.ncut1])

    @property
    def nfy(self) -> np.ndarray:
        return np.array([self.ncut1, 0, self.nx, self.nx, self.nx])

    @property
    def compact_shape_2d(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def compact_shape_3d(self) -> tuple[int, int, int]:
        return (self.nz, self.ny, self.nx)

    @property
    def tracer_shape_2d(self) -> tuple[int, int]:
        return (self.ncut1 + self.nx + self.ncut2, 2 * self.nx)

    @property
    def tracer_shape_3d(self) -> tuple[int, int, int]:
        return (self.nz, *self.tracer_shape_2d)


def aste90(nz: int = 50) -> ASTEGridSpec:
    """Return the ASTE 90 x 150 x 60 grid specification used by mini-ASTE."""

    return ASTEGridSpec(nx=90, ncut1=150, ncut2=60, nz=nz, name="aste90")


def aste270(nz: int = 50) -> ASTEGridSpec:
    """Return the ASTE 270 x 450 x 180 grid specification."""

    return ASTEGridSpec(nx=270, ncut1=450, ncut2=180, nz=nz, name="aste270")


def _as_3d(field: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(field)
    if arr.ndim == 2:
        return arr[np.newaxis, :, :], True
    if arr.ndim == 3:
        return arr, False
    raise ValueError(f"expected a 2D or 3D array, got shape {arr.shape}")


def _grid_arrays(grid: ASTEGridSpec | None, nfx=None, nfy=None) -> tuple[np.ndarray, np.ndarray]:
    if grid is not None:
        return grid.nfx, grid.nfy
    if nfx is None or nfy is None:
        raise ValueError("pass either grid=ASTEGridSpec or both nfx and nfy")
    return np.asarray(nfx), np.asarray(nfy)


def compact_to_faces(field: np.ndarray, grid: ASTEGridSpec | None = None, *, nfx=None, nfy=None) -> SimpleNamespace:
    """Split an ASTE compact array into faces 1, 3, 4, and 5."""

    nfx, nfy = _grid_arrays(grid, nfx, nfy)
    arr, _ = _as_3d(field)
    nx = int(nfx[0])
    return SimpleNamespace(
        f1=arr[:, 0 : nfy[0], 0:nx],
        f3=arr[:, nfy[0] : nfy[0] + nfy[2], 0:nx],
        f4=np.reshape(arr[:, nfy[0] + nfy[2] : nfy[0] + nfy[2] + nfx[3], 0:nx], [-1, nx, nfx[3]]),
        f5=np.reshape(
            arr[:, nfy[0] + nfy[2] + nfx[3] : nfy[0] + nfy[2] + nfx[3] + nfx[4], 0:nx],
            [-1, nx, nfx[4]],
        ),
    )


def faces_to_compact(faces: SimpleNamespace, grid: ASTEGridSpec | None = None, *, nfx=None, nfy=None) -> np.ndarray:
    """Combine face arrays into ASTE compact layout."""

    nfx, nfy = _grid_arrays(grid, nfx, nfy)
    f1, squeeze = _as_3d(faces.f1)
    f3, _ = _as_3d(faces.f3)
    f4, _ = _as_3d(faces.f4)
    f5, _ = _as_3d(faces.f5)
    nz = f1.shape[0]
    nx = int(nfx[0])
    out = np.full((nz, 2 * nfy[0] + nx + nfx[3], nx), np.nan)
    out[:, 0 : nfy[0], :] = f1
    out[:, nfy[0] : nfy[0] + nfy[2], :] = f3
    out[:, nfy[0] + nfy[2] : nfy[0] + nfy[2] + nfx[3], :] = np.reshape(f4, [nz, nfx[3], nfy[3]])
    out[:, nfy[0] + nfy[2] + nfx[3] : nfy[0] + nfy[2] + nfx[3] + nfx[4], :] = np.reshape(
        f5, [nz, nfx[4], nfy[4]]
    )
    return out[0] if squeeze else out


def compact_to_tracer(field: np.ndarray, grid: ASTEGridSpec | None = None, *, nfx=None, nfy=None) -> np.ndarray:
    """Rotate and place compact ASTE tracer data into map/tracer layout."""

    nfx, nfy = _grid_arrays(grid, nfx, nfy)
    arr, squeeze = _as_3d(field)
    nz, _, nx = arr.shape
    out = np.full((nz, nfy[0] + nx + nfx[3], 2 * nx), np.nan)

    out[:, 0 : nfy[0], nx : 2 * nx] = arr[:, 0 : nfy[0], 0:nx]
    out[:, nfy[0] : nfy[0] + nx, nx : 2 * nx] = np.rot90(arr[:, nfy[0] : nfy[0] + nx, 0:nx], k=-1, axes=(1, 2))

    face4 = np.reshape(arr[:, nfy[0] + nx : nfy[0] + nx + nfx[3], 0:nx], [nz, nx, nfx[3]])
    out[:, nfy[0] + nx : nfy[0] + nx + nfx[3], nx : 2 * nx] = np.rot90(face4, k=-1, axes=(1, 2))

    face5 = np.reshape(arr[:, nfy[0] + nx + nfx[3] : nfy[0] + nx + nfx[3] + nfx[4], 0:nx], [nz, nx, nfx[4]])
    out[:, 0 : nfx[4], 0:nx] = np.rot90(face5, k=1, axes=(1, 2))

    return out[0] if squeeze else out


def tracer_to_compact(field: np.ndarray, grid: ASTEGridSpec | None = None, *, nfx=None, nfy=None) -> np.ndarray:
    """Reverse :func:`compact_to_tracer`."""

    nfx, nfy = _grid_arrays(grid, nfx, nfy)
    arr, squeeze = _as_3d(field)
    nx = int(nfx[2])
    tmp1 = arr[:, : nfy[0], nx:]
    tmp3 = np.rot90(arr[:, nfy[0] : nfy[0] + nx, nx:], k=1, axes=(1, 2))
    tmp4 = np.rot90(arr[:, nfy[0] + nx :, nx:], k=1, axes=(1, 2))
    tmp5 = np.rot90(arr[:, 0 : nfy[0], 0:nx], k=-1, axes=(1, 2))
    compact = faces_to_compact(SimpleNamespace(f1=tmp1, f3=tmp3, f4=tmp4, f5=tmp5), nfx=nfx, nfy=nfy)
    return compact[0] if squeeze and compact.ndim == 3 else compact


def compact_uv_to_tracer(
    u: np.ndarray,
    v: np.ndarray,
    grid: ASTEGridSpec | None = None,
    *,
    nfx=None,
    nfy=None,
    absolute: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Map compact U/V arrays to padded tracer-layout vector arrays."""

    nfx, nfy = _grid_arrays(grid, nfx, nfy)
    uarr, squeeze = _as_3d(u)
    varr, _ = _as_3d(v)
    if uarr.shape != varr.shape:
        raise ValueError(f"u and v must have the same shape, got {uarr.shape} and {varr.shape}")

    nz, _, nx = uarr.shape
    unew = np.full((nz, nfy[0] + nfx[2] + nfx[3], nfy[4] + nfx[0]), np.nan)
    vnew = np.full_like(unew, np.nan)

    unew[:, 0 : nfy[0], nx : 2 * nx] = uarr[:, 0 : nfy[0], 0:nx]
    vnew[:, 0 : nfy[0], nx : 2 * nx] = varr[:, 0 : nfy[0], 0:nx]

    fu3 = np.rot90(uarr[:, nfy[0] : nfy[0] + nx, 0:nx], k=-1, axes=(1, 2))
    fv3 = np.rot90(varr[:, nfy[0] : nfy[0] + nx, 0:nx], k=-1, axes=(1, 2))
    unew[:, nfy[0] : nfy[0] + nx, nx : 2 * nx] = -fu3
    vnew[:, nfy[0] : nfy[0] + nx, nx : 2 * nx] = -fv3

    fu4 = np.reshape(uarr[:, nfy[0] + nx : nfy[0] + nx + nfx[3], 0:nx], [nz, nx, nfx[3]])
    fv4 = np.reshape(varr[:, nfy[0] + nx : nfy[0] + nx + nfx[3], 0:nx], [nz, nx, nfx[3]])
    fu4 = np.rot90(fu4, k=-1, axes=(1, 2))
    fv4 = np.rot90(fv4, k=-1, axes=(1, 2))
    unew[:, nfy[0] + nx : nfy[0] + nx + nfx[3], nx : 2 * nx] = -fv4
    vnew[:, nfy[0] + nx : nfy[0] + nx + nfx[3], nx : 2 * nx] = fu4

    fu5 = np.reshape(uarr[:, nfy[0] + nx + nfx[3] : nfy[0] + nx + nfx[3] + nfx[4], 0:nx], [nz, nx, nfx[4]])
    fv5 = np.reshape(varr[:, nfy[0] + nx + nfx[3] : nfy[0] + nx + nfx[3] + nfx[4], 0:nx], [nz, nx, nfx[4]])
    fu5 = np.rot90(fu5, k=1, axes=(1, 2))
    fv5 = np.rot90(fv5, k=1, axes=(1, 2))
    unew[:, 0 : nfx[4], 0:nx] = fv5
    vnew[:, 0 : nfx[4], 0:nx] = -fu5

    if absolute:
        unew = np.abs(unew)
        vnew = np.abs(vnew)

    out_shape = (nz, unew.shape[1] + 1, unew.shape[2] + 1)
    up = np.full(out_shape, np.nan)
    vp = np.full(out_shape, np.nan)
    up[:, : nfy[0], : nx * 2] = unew[:, : nfy[0], : nx * 2]
    up[:, nfy[0] : unew.shape[1], 1 : nx * 2 + 1] = unew[:, nfy[0] : unew.shape[1], : nx * 2]
    vp[:, : nfy[0] + nfx[2] + nfx[3], nx : 2 * nx] = vnew[:, : nfy[0] + nfx[2] + nfx[3], nx : 2 * nx]
    vp[:, 1 : unew.shape[1] + 1, :nx] = vnew[:, : unew.shape[1], :nx]

    if squeeze:
        return up[0], vp[0]
    return up, vp

