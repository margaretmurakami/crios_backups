# some sample functions (to be continued) to plot items in map view

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import binary_dilation

import sys
sys.path.append("/home/mmurakami/MITgcm/MITgcm_c68r/MITgcm-checkpoint68r/utils/python/MITgcmutils/MITgcmutils/") # go to parent dir
from mds import *
import os
sys.path.append("/home/mmurakami/crios_backups/an_helper_functions")
sys.path.append("/home/mmurakami/crios_backups/an_helper_functions")
from read_binary import *
from calc_UV_conv_1face import calc_UV_conv_1face
from calc_mskmean_T_mod import calc_mskmean_T_mod
from mk3D_mod import mk3D_mod
from aste_helper_funcs import *
from timing_functions import *           # ts2dte, get_fnames, etc.
from binning import *


def SI_masked(
    tsstr, *,  # expects month timesteps; for JFM mean use Jan..Apr so you can form Jan,Feb,Mar month-means
    dirState, ny, nx, nfx, nfy,
    landmsk, mymsk,
    file_name="state_2d_set1",
    varname="SIarea",
    land_buffer_iters=1,
):
    """
    Returns:
      SI_masked_mean_tr : (tile) 2D on tracer grid (get_aste_tracer(...)[0]) AFTER land-buffer masking,
                          averaged over JFM monthly means.
      SI_mean_tr        : same but without land-buffer masking
      meta              : dict with used pairs and timesteps

    Convention:
      Each monthly mean uses the *second* timestep in the pair (matches your read=[int(tsstr[1])] pattern).
      JFM mean is the average of the 3 monthly fields from pairs:
        Jan=[ts0,ts1], Feb=[ts1,ts2], Mar=[ts2,ts3]
      So you should pass tsstr = [Jan, Feb, Mar, Apr] (length 4).
    """
    tsstr = list(tsstr)
    if len(tsstr) < 2:
        raise ValueError("Need at least 2 timesteps to form one monthly field.")
    npairs = len(tsstr) - 1
    if npairs < 1:
        raise ValueError("Need at least one pair.")

    # --- parse meta once to find record index ---
    meta_state_2d = parsemeta(os.path.join(dirState, f"{file_name}.{tsstr[0]}.meta"))
    fldlist = np.array(meta_state_2d["fldList"])
    idx = np.where(fldlist == varname)[0]
    if len(idx) == 0:
        raise KeyError(f"{varname} not in fldList. Available: {list(fldlist)}")
    rec = int(idx[0])

    # --- land buffer mask on tracer grid (constant across time) ---
    landmask_tr = (get_aste_tracer(landmsk, nfx, nfy)[0] > 0)
    landbuffer_tr = binary_dilation(landmask_tr, iterations=land_buffer_iters)

    # --- loop month-pairs and accumulate on tracer grid ---
    acc_tr = None
    acc_masked_tr = None

    pairs = []
    for i in range(npairs):
        pairs.append((tsstr[i], tsstr[i + 1]))

        # match your convention: use the second timestep of the pair
        read = [int(tsstr[i + 1])]
        SIarea_frac, its, meta = rdmds(
            os.path.join(dirState, file_name),
            read,
            returnmeta=True,
            rec=rec
        )  # m^2/m^2
        SIarea = np.asarray(SIarea_frac).reshape((ny, nx))

        # map to tracer-tile (your plotting face)
        SI_tr = get_aste_tracer(SIarea, nfx, nfy)[0]

        # apply land-buffer mask
        SI_masked = np.where(landbuffer_tr, np.nan, SI_tr)

        if acc_tr is None:
            acc_tr = np.zeros_like(SI_tr, dtype=np.float64)
            acc_masked_tr = np.zeros_like(SI_tr, dtype=np.float64)
            cnt_tr = np.zeros_like(SI_tr, dtype=np.float64)
            cnt_masked = np.zeros_like(SI_tr, dtype=np.float64)

        # nan-safe accumulate (mean over time)
        ok = np.isfinite(SI_tr)
        acc_tr[ok] += SI_tr[ok]
        cnt_tr[ok] += 1.0

        okm = np.isfinite(SI_masked)
        acc_masked_tr[okm] += SI_masked[okm]
        cnt_masked[okm] += 1.0

    SI_mean_tr = acc_tr / np.where(cnt_tr > 0, cnt_tr, np.nan)
    SI_masked_mean_tr = acc_masked_tr / np.where(cnt_masked > 0, cnt_masked, np.nan)

    meta_out = {"ts_list": tsstr, "pairs": pairs, "npairs": npairs, "read_second_of_pair": True}
    return SI_masked_mean_tr, SI_mean_tr, meta_out

def plot_budget_miniaste(
    terms3D,
    mygrid,
    mymsk,
    landmsk,
    SI_masked,
    nfx,
    nfy,
    get_aste_tracer,
    klev=0,
    kind="T",
    xlim=(140, 180),
    ylim=(145, 180),
    figsize=(12, 12),
    cbar_label=None,
    suptitle=None,
    contour_interval=100,
    save=False,
    save_name=None,
):
    """
    Plot a 3x3 mini-ASTE panel of budget terms for either heat or salt.

    Parameters
    ----------
    terms3D : dict
        Dictionary of 3D budget terms.
        For heat: must contain keys
            'ADVh', 'ADVr', 'DFhT', 'DFrT', 'KPP', 'surf', 'tend'
        For salt: must contain keys
            'ADVh', 'ADVr', 'DFhS', 'DFrS', 'KPP', 'surf', 'tend'
    mygrid : dict
        Grid dictionary containing at least 'Depth'.
    mymsk : ndarray
        Mask applied to plotted fields.
    landmsk : ndarray
        Land mask.
    SI_masked : ndarray
        Sea-ice mask/contour field.
    nfx, nfy : any
        Inputs required by get_aste_tracer.
    get_aste_tracer : callable
        Function that maps a tracer field to ASTE layout.
    klev : int, optional
        Vertical level to plot.
    kind : {'T', 'S'}, optional
        Budget type. 'T' for heat, 'S' for salt.
    xlim, ylim : tuple, optional
        Axis limits.
    figsize : tuple, optional
        Figure size.
    cbar_label : str, optional
        Colorbar label. If None, a default is used.
    suptitle : str, optional
        Figure title. If None, a default is used.
    contour_interval : int or float, optional
        Bathymetry contour interval in meters.
    save : bool, optional
        If True, save figure locally in ./figs/.
    save_name : str, optional
        Output filename. If None, a default name is generated.

    Returns
    -------
    fig, axes : matplotlib Figure and Axes
    """
    kind = kind.upper()
    if kind not in {"T", "S"}:
        raise ValueError("kind must be 'T' or 'S'")

    dfh_key = f"DFh{kind}"
    dfr_key = f"DFr{kind}"

    required_keys = ["ADVh", "ADVr", dfh_key, dfr_key, "KPP", "surf", "tend"]
    missing = [k for k in required_keys if k not in terms3D]
    if missing:
        raise KeyError(f"Missing required keys in terms3D: {missing}")

    if cbar_label is None:
        cbar_label = "degC.m^3/s" if kind == "T" else "psu.m^3/s"

    if suptitle is None:
        suptitle = f"Contributions to {'Heat' if kind == 'T' else 'Salt'} Budget at k={klev}"

    fig, axes = plt.subplots(3, 3, figsize=figsize)

    # --- precompute static fields once ---
    land_plot = landmsk.copy().astype(float)
    land_plot[land_plot == 0] = np.nan
    land_plot_aste = get_aste_tracer(land_plot, nfx, nfy)[0]

    depth_plot = mygrid["Depth"].copy().astype(float)
    depth_plot[depth_plot <= 0] = np.nan
    depth_aste = get_aste_tracer(depth_plot, nfx, nfy)[0]

    depth_levels = np.arange(
        contour_interval,
        np.nanmax(depth_plot) + contour_interval,
        contour_interval,
    )

    # --- define all panels once ---
    plot_defs = [
        ("a) ADVh", terms3D["ADVh"][klev] * mymsk),
        ("b) ADVh + ADVr", (terms3D["ADVh"][klev] + terms3D["ADVr"][klev]) * mymsk),
        ("c) " + dfh_key + " + " + dfr_key, (terms3D[dfh_key][klev] + terms3D[dfr_key][klev]) * mymsk),
        ("d) " + dfh_key, terms3D[dfh_key][klev] * mymsk),
        ("e) " + dfr_key, terms3D[dfr_key][klev] * mymsk),
        ("f) KPP", terms3D["KPP"][klev] * mymsk),
        ("g) surf", terms3D["surf"][klev] * mymsk),
        ("h) tend", terms3D["tend"][klev] * mymsk),
    ]

    resid = (
        terms3D["tend"][klev]
        - terms3D["ADVh"][klev]
        - terms3D["ADVr"][klev]
        - terms3D[dfh_key][klev]
        - terms3D[dfr_key][klev]
        - terms3D["KPP"][klev]
        - terms3D["surf"][klev]
    ) * mymsk

    plot_defs.append(("i) resid", resid))

    # --- plot loop ---
    for ax, (title, field) in zip(axes.flat, plot_defs):
        b = get_aste_tracer(field, nfx, nfy)[0]
        vlev = np.nanmax(np.abs(b))

        cb = ax.pcolormesh(b, cmap="seismic", vmin=-vlev, vmax=vlev)

        ax.pcolormesh(land_plot_aste, cmap="Greys", vmin=-2, vmax=4)
        ax.contour(depth_aste, levels=depth_levels, colors="k", linewidths=0.4, alpha=0.3)
        ax.contour(
            SI_masked,
            levels=[0.15],
            colors="lightgreen",
            linewidths=1.5,
            linestyles="--",
            zorder=10,
        )

        plt.colorbar(cb, ax=ax, label=cbar_label)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(title)

    plt.suptitle(suptitle)
    plt.tight_layout()

    if save:
        os.makedirs("figs", exist_ok=True)

        if save_name is None:
            budget_type = "heat" if kind == "T" else "salt"
            save_name = f"miniaste_{budget_type}_budget_k{klev}.png"

        save_path = os.path.join("figs", save_name)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    return fig, axes
