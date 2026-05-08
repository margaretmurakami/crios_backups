"""
Shared run configuration helpers for the regional fullASTE budget notebooks.

Use this from a notebook setup cell:

    from fullaste_run_config import install_fullaste_config

    cfg = install_fullaste_config(
        globals(),
        profile="aste90",
        budget="Heat",
        dirroot="/scratch3/atnguyen/aste_90x150x60/",
        runstr="run_c68v_heffmosm3x_layers_lessmem1_viscAHp5em2_it0000_pk0000000001_r8_07May2026_pfe/",
    )

After that cell, the existing notebooks can keep using the old variable names:
dirroot, dirGrid, dirIn, dirdiags, dirstate, nx, ny, nz, nfx, nfy, myparms,
strbudg, kBudget, bigaste, labsea, oneface, etc.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional

import numpy as np


DEFAULT_HELPER_PATH = "/home/mmurakami/crios_backups/an_helper_functions"
DEFAULT_MDS_PATH = (
    "/home/mmurakami/MITgcm/MITgcm_c68r/"
    "MITgcm-checkpoint68r/utils/python/MITgcmutils/MITgcmutils/"
)


def configure_import_paths(
    *,
    mds_path: str = DEFAULT_MDS_PATH,
    helper_path: str = DEFAULT_HELPER_PATH,
) -> None:
    """Add the shared MITgcm and analysis helper paths once."""
    for path in (mds_path, helper_path):
        if path and path not in sys.path:
            sys.path.append(path)


PROFILE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "aste270": {
        "dirroot": "/scratch2/atnguyen/aste_270x450x180/",
        "dirrun": "/scratch/atnguyen/aste_270x450x180/OFFICIAL_ASTE_R1_Sep2019/",
        "dirGrid": "/scratch2/atnguyen/aste_270x450x180/GRID_real8/",
        "dirgridnb": "/scratch2/atnguyen/aste_270x450x180/GRID_noblank/",
        "dirgridw": "/scratch2/atnguyen/aste_270x450x180/GRID_wet/",
        "nx": 270,
        "ncut1": 450,
        "ncut2": 180,
        "nz": 50,
        "bigaste": True,
        "labsea": False,
        "oneface": False,
        "dt": 600,
    },
    "aste90": {
        "dirroot": "/scratch3/atnguyen/aste_90x150x60/",
        "runstr": "run_c68v_heffmosm3x_layers_lessmem1_viscAHp5em2_it0000_pk0000000001_r8_07May2026_pfe/",
        "dirGrid": "/scratch3/atnguyen/aste_90x150x60/GRID_real8/",
        "dirgridnb": "/scratch3/atnguyen/aste_90x150x60/GRID_noblank/",
        "nx": 90,
        "ncut1": 150,
        "ncut2": 60,
        "nz": 50,
        "bigaste": False,
        "labsea": False,
        "oneface": False,
        "dt": 3600,
    },
    "labsea": {
        "dirroot": "/scratch/mmurakami/labsea/layers/",
        "runstr": "run_c68v_layers_lessmem1_5ts_06May2026/",
        "dirGrid": "/scratch2/atnguyen/labsea/GRID_r8/",
        "nx": 20,
        "ny": 16,
        "nz": 23,
        "ncut1": 0,
        "ncut2": 0,
        "bigaste": False,
        "labsea": True,
        "oneface": True,
        "dt": 3600,
    },
}


DEFAULT_MYPARMS: Dict[str, Any] = {
    "yearFirst": 1979,
    "yearLast": 1979,
    "yearInAv": [1979, 1979],
    "timeStep": 3600,
    "iceModel": 1,
    "useRFWF": 1,
    "useNLFS": 4,
    "rStar": 2,
    "rhoconst": 1029,
    "rcp": 1029 * 3994,
    "rhoi": 910,
    "rhosn": 330,
    "flami": 334000,
    "flamb": 2500000,
    "SIsal0": 4,
    "diagsAreMonthly": 0,
    "diagsAreAnnual": 0,
}


@dataclass
class FullASTERunConfig:
    profile: str
    budget: str
    dirroot: str
    dirGrid: str
    nx: int
    ny: int
    nz: int
    ncut1: int = 0
    ncut2: int = 0
    runstr: Optional[str] = None
    dirrun: Optional[str] = None
    dirgridnb: Optional[str] = None
    dirgridw: Optional[str] = None
    bigaste: bool = False
    labsea: bool = False
    oneface: bool = False
    dt: int = 3600
    iB: int = 6
    tsstr: Optional[np.ndarray] = None
    myparms: Dict[str, Any] = field(default_factory=dict)

    @property
    def paths(self) -> Dict[str, str]:
        return make_paths(
            self.dirroot,
            runstr=self.runstr,
            dirrun=self.dirrun,
            dirGrid=self.dirGrid,
            dirgridnb=self.dirgridnb,
            dirgridw=self.dirgridw,
        )

    @property
    def geometry(self) -> Dict[str, Any]:
        return make_geometry(
            profile=self.profile,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            ncut1=self.ncut1,
            ncut2=self.ncut2,
            oneface=self.oneface,
        )

    @property
    def namespace(self) -> Dict[str, Any]:
        ns: Dict[str, Any] = {}
        ns.update(self.paths)
        ns.update(self.geometry)
        ns.update(
            {
                "profile": self.profile,
                "bigaste": self.bigaste,
                "labsea": self.labsea,
                "oneface": self.oneface,
                "iB": self.iB,
                "dt": self.dt,
                "strbudg": self.budget,
                "kBudget": 1,
                "test3d": True,
                "plot_fig": 1,
                "save_budg_3d": 0,
                "save_budg_2d": 1,
                "save_budg_scalar": 0,
                "save_budg_lev": 0,
                "myparms": self.myparms,
            }
        )
        if self.tsstr is not None:
            ns["tsstr"] = np.asarray(self.tsstr)
            ns["t1"] = int(ns["tsstr"][0])
            ns["t2"] = int(ns["tsstr"][1])
        return ns


def normalize_dir(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return os.path.join(path, "")


def make_paths(
    dirroot: str,
    *,
    runstr: Optional[str] = None,
    dirrun: Optional[str] = None,
    dirGrid: Optional[str] = None,
    dirgridnb: Optional[str] = None,
    dirgridw: Optional[str] = None,
) -> Dict[str, str]:
    dirroot = normalize_dir(dirroot)
    if dirrun is None:
        if runstr is None:
            raise ValueError("Provide either runstr or dirrun.")
        dirrun = os.path.join(dirroot, runstr)
    dirrun = normalize_dir(dirrun)

    paths = {
        "dirroot": dirroot,
        "dirrun": dirrun,
        "layers_path": dirrun,
        "dirIn": os.path.join(dirrun, "diags/BUDG/"),
        "dirbudg": os.path.join(dirrun, "diags/BUDG/"),
        "dirdiags": os.path.join(dirrun, "diags/BUDG/"),
        "dirDiags": os.path.join(dirrun, "diags/"),
        "dirState": os.path.join(dirrun, "diags/STATE/"),
        "dirstate": os.path.join(dirrun, "diags/STATE/"),
        "dirlayers": os.path.join(dirrun, "diags/LAYERS/"),
        "dirtrsp": os.path.join(dirrun, "diags/TRSP/"),
        "dirmask": os.path.join(dirroot, "run_template/input_maskTransport/"),
        "dirGrid": normalize_dir(dirGrid) or os.path.join(dirroot, "GRID_real8/"),
        "dirgridnb": normalize_dir(dirgridnb) or os.path.join(dirroot, "GRID_noblank/"),
    }
    if dirgridw is not None:
        paths["dirgridw"] = normalize_dir(dirgridw)
    return paths


def make_geometry(
    *,
    profile: str,
    nx: int,
    ny: Optional[int],
    nz: int,
    ncut1: int = 0,
    ncut2: int = 0,
    oneface: bool = False,
) -> Dict[str, Any]:
    if ny is None:
        ny = 2 * ncut1 + nx + ncut2

    out: Dict[str, Any] = {
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz),
        "ncut1": int(ncut1),
        "ncut2": int(ncut2),
    }
    if oneface:
        faces = np.array([[nx, ny]])
        out.update(
            {
                "nFaces": 1,
                "facesSize": faces,
                "facesExpand": faces.copy(),
            }
        )
    else:
        out.update(
            {
                "nfx": np.array([nx, 0, nx, ncut2, ncut1]),
                "nfy": np.array([ncut1, 0, nx, nx, nx]),
            }
        )
    return out


def default_myparms(*, dt: int = 3600, overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    myparms = dict(DEFAULT_MYPARMS)
    myparms["timeStep"] = dt
    if overrides:
        myparms.update(dict(overrides))
    return myparms


def build_fullaste_config(
    *,
    profile: str,
    budget: str,
    dirroot: Optional[str] = None,
    runstr: Optional[str] = None,
    dirrun: Optional[str] = None,
    dirGrid: Optional[str] = None,
    dirgridnb: Optional[str] = None,
    dirgridw: Optional[str] = None,
    nx: Optional[int] = None,
    ny: Optional[int] = None,
    nz: Optional[int] = None,
    ncut1: Optional[int] = None,
    ncut2: Optional[int] = None,
    dt: Optional[int] = None,
    iB: int = 6,
    tsstr: Optional[Any] = None,
    myparms_overrides: Optional[Mapping[str, Any]] = None,
) -> FullASTERunConfig:
    profile_key = profile.lower()
    if profile_key not in PROFILE_DEFAULTS:
        raise ValueError(f"Unknown profile {profile!r}. Expected one of {sorted(PROFILE_DEFAULTS)}.")

    defaults = dict(PROFILE_DEFAULTS[profile_key])
    dt_final = int(dt if dt is not None else defaults["dt"])
    nx_final = int(nx if nx is not None else defaults["nx"])
    ncut1_final = int(ncut1 if ncut1 is not None else defaults.get("ncut1", 0))
    ncut2_final = int(ncut2 if ncut2 is not None else defaults.get("ncut2", 0))
    nz_final = int(nz if nz is not None else defaults["nz"])
    oneface = bool(defaults.get("oneface", False))
    ny_default = defaults.get("ny")
    ny_final = int(ny if ny is not None else (ny_default if ny_default is not None else 2 * ncut1_final + nx_final + ncut2_final))

    return FullASTERunConfig(
        profile=profile_key,
        budget=budget,
        dirroot=dirroot or defaults["dirroot"],
        runstr=runstr if runstr is not None else defaults.get("runstr"),
        dirrun=dirrun if dirrun is not None else defaults.get("dirrun"),
        dirGrid=dirGrid or defaults["dirGrid"],
        dirgridnb=dirgridnb if dirgridnb is not None else defaults.get("dirgridnb"),
        dirgridw=dirgridw if dirgridw is not None else defaults.get("dirgridw"),
        nx=nx_final,
        ny=ny_final,
        nz=nz_final,
        ncut1=ncut1_final,
        ncut2=ncut2_final,
        bigaste=bool(defaults.get("bigaste", False)),
        labsea=bool(defaults.get("labsea", False)),
        oneface=oneface,
        dt=dt_final,
        iB=iB,
        tsstr=np.asarray(tsstr) if tsstr is not None else None,
        myparms=default_myparms(dt=dt_final, overrides=myparms_overrides),
    )


def install_fullaste_config(namespace: MutableMapping[str, Any], **kwargs: Any) -> FullASTERunConfig:
    cfg = build_fullaste_config(**kwargs)
    namespace.update(cfg.namespace)
    return cfg


def discover_budget_timesteps(dirIn: str, prefix: str = "budg2d_snap_set1") -> np.ndarray:
    """Return sorted timestep strings available for a budget diagnostic prefix."""
    vals = []
    for fname in os.listdir(dirIn):
        if not (fname.startswith(prefix + ".") and fname.endswith(".data")):
            continue
        parts = fname.split(".")
        if len(parts) >= 3:
            vals.append(parts[-2])
    return np.array(sorted(set(vals)))


def make_mygrid_spec(
    dirGrid: str,
    *,
    nx: int,
    ny: int,
    nz: int,
    oneface: bool = False,
    facesSize: Optional[Any] = None,
    facesExpand: Optional[Any] = None,
) -> Dict[str, Any]:
    if oneface:
        facesSize = np.asarray(facesSize if facesSize is not None else [[nx, ny]])
        facesExpand = np.asarray(facesExpand if facesExpand is not None else facesSize)
    else:
        facesSize = [ny, nx]
        facesExpand = [ny, nx]

    return {
        "dirGrid": dirGrid,
        "nFaces": 1,
        "fileFormat": "compact",
        "memoryLimit": 2,
        "ioSize": [nx * ny, 1],
        "facesSize": facesSize,
        "facesExpand": facesExpand,
        "missVal": 0,
    }


def load_mygrid(
    rdmds_func: Any,
    dirGrid: str,
    *,
    nx: int,
    ny: int,
    nz: int,
    oneface: bool = False,
    facesSize: Optional[Any] = None,
    facesExpand: Optional[Any] = None,
) -> Dict[str, Any]:
    mygrid = make_mygrid_spec(
        dirGrid,
        nx=nx,
        ny=ny,
        nz=nz,
        oneface=oneface,
        facesSize=facesSize,
        facesExpand=facesExpand,
    )
    fldstr2d = ["XC", "YC", "XG", "YG", "RAC", "Depth", "DXG", "DYG", "DXC", "DYC"]
    fldstr3d = ["hFacC", "hFacW", "hFacS", "mskC", "mskS", "mskW"]
    fldstr3dp = ["hFacC", "hFacW", "hFacS", "maskCtrlC", "maskCtrlS", "maskCtrlW"]
    fldstr1d = ["RC", "RF", "DRC", "DRF"]

    for fld in fldstr1d:
        mygrid[fld] = np.squeeze(rdmds_func(os.path.join(dirGrid, fld)))
    for fld, disk_name in zip(fldstr3d, fldstr3dp):
        mygrid[fld] = rdmds_func(os.path.join(dirGrid, disk_name)).reshape(nz, ny, nx)
    for fld in fldstr2d:
        mygrid[fld] = rdmds_func(os.path.join(dirGrid, fld)).reshape(ny, nx)

    return mygrid


def prepare_grid_metrics(mygrid: Mapping[str, Any], mk3D_mod_func: Any, *, nz: int, ny: int, nx: int) -> Dict[str, Any]:
    RAC = mygrid["RAC"]
    hfC = np.array(mygrid["hFacC"], copy=True)
    hfC[hfC == 0] = np.nan
    hfC1 = np.array(hfC[0], copy=True)
    hfC1[hfC1 == 0] = np.nan
    zero3d = np.zeros((nz, ny, nx))

    return {
        "RAC": RAC,
        "RAC3": np.tile(RAC, (nz, 1, 1)),
        "hfC": hfC,
        "hfC1": hfC1,
        "DD": mygrid["Depth"],
        "dxg": mygrid["DXG"],
        "dyg": mygrid["DYG"],
        "dxg3d": np.tile(mygrid["DXG"], (nz, 1, 1)),
        "dyg3d": np.tile(mygrid["DYG"], (nz, 1, 1)),
        "drf3d": mk3D_mod_func(mygrid["DRF"], zero3d),
        "DD3d": mk3D_mod_func(mygrid["Depth"], zero3d),
        "RACg": RAC * hfC1,
        "RACgp": _masked_edge_area(RAC, hfC1),
    }


def _masked_edge_area(RAC: np.ndarray, hfC1: np.ndarray) -> np.ndarray:
    hfC1p = np.array(hfC1, copy=True)
    hfC1p[:, -1] = np.nan
    hfC1p[-1, :] = np.nan
    return RAC * hfC1p


def apply_aste_open_boundary_mask(
    hfac_c: np.ndarray,
    *,
    nx: int,
    nz: int,
    nfx: np.ndarray,
    nfy: np.ndarray,
    get_aste_tracer_func: Any,
    aste_tracer2compact_func: Any,
) -> np.ndarray:
    """Replicate the repeated ASTE open-boundary hFac mask used by the notebooks."""
    hfac = np.array(hfac_c, copy=True)
    hfac[hfac == 0] = np.nan
    hf1 = get_aste_tracer_func(hfac[0], nfx, nfy)
    if nx == 90:
        hf1[:, 281, :] = np.nan
        hf1[:, 7, :] = np.nan
        hf1[:, 86, 122] = np.nan
    elif nx == 270:
        hf1[:, 844, :] = np.nan
        hf1[:, 23, :] = np.nan
        hf1[:, 365, 260:261] = np.nan
    hf1 = aste_tracer2compact_func(hf1, nfx, nfy)
    return hfac * np.tile(hf1, (nz, 1, 1))
