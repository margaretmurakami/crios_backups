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

        For heat, minimum required keys:
            'ADVh', 'ADVr', 'DFhT', 'DFrT', 'KPP', 'tend'
        and either:
            - 'surf'
          or
            - 'TFLUX'
        Optional heat surface split keys:
            - 'zconv_top'
            - 'swtop'

        For salt, required keys:
            'ADVh', 'ADVr', 'DFhS', 'DFrS', 'KPP', 'surf', 'tend'

    Returns
    -------
    fig, axes : matplotlib Figure and Axes
    """
    kind = kind.upper()
    if kind not in {"T", "S"}:
        raise ValueError("kind must be 'T' or 'S'")

    dfh_key = f"DFh{kind}"
    dfr_key = f"DFr{kind}"

    # ----------------------------
    # required key handling
    # ----------------------------
    base_required = ["ADVh", "ADVr", dfh_key, dfr_key, "KPP", "tend"]
    missing = [k for k in base_required if k not in terms3D]
    if missing:
        raise KeyError(f"Missing required keys in terms3D: {missing}")

    if kind == "S":
        if "surf" not in terms3D:
            raise KeyError("For salt, terms3D must contain key 'surf'")
        surf_key = "surf"
        has_surface_split = False
    else:
        # for heat, accept either 'surf' or 'TFLUX'
        if "TFLUX" in terms3D:
            surf_key = "TFLUX"
        elif "surf" in terms3D:
            surf_key = "surf"
        else:
            raise KeyError("For heat, terms3D must contain either 'TFLUX' or 'surf'")

        has_surface_split = ("zconv_top" in terms3D) and ("swtop" in terms3D)

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

    # ----------------------------
    # panel definitions
    # ----------------------------
    if kind == "T" and has_surface_split:
        plot_defs = [
            ("a) ADVh", terms3D["ADVh"][klev] * mymsk),
            ("b) ADVh + ADVr", (terms3D["ADVh"][klev] + terms3D["ADVr"][klev]) * mymsk),
            ("c) " + dfh_key + " + " + dfr_key,
             (terms3D[dfh_key][klev] + terms3D[dfr_key][klev]) * mymsk),
            ("d) " + dfh_key, terms3D[dfh_key][klev] * mymsk),
            ("e) " + dfr_key, terms3D[dfr_key][klev] * mymsk),
            ("f) zconv_top", terms3D["zconv_top"][klev] * mymsk),
            ("g) swtop", terms3D["swtop"][klev] * mymsk),
            ("h) KPP", terms3D["KPP"][klev] * mymsk),
        ]

        resid = (
            terms3D["tend"][klev]
            - terms3D["ADVh"][klev]
            - terms3D["ADVr"][klev]
            - terms3D[dfh_key][klev]
            - terms3D[dfr_key][klev]
            - terms3D["zconv_top"][klev]
            - terms3D["swtop"][klev]
            - terms3D["KPP"][klev]
        ) * mymsk

        plot_defs.append(("i) resid", resid))

    else:
        plot_defs = [
            ("a) ADVh", terms3D["ADVh"][klev] * mymsk),
            ("b) ADVh + ADVr", (terms3D["ADVh"][klev] + terms3D["ADVr"][klev]) * mymsk),
            ("c) " + dfh_key + " + " + dfr_key,
             (terms3D[dfh_key][klev] + terms3D[dfr_key][klev]) * mymsk),
            ("d) " + dfh_key, terms3D[dfh_key][klev] * mymsk),
            ("e) " + dfr_key, terms3D[dfr_key][klev] * mymsk),
            ("f) KPP", terms3D["KPP"][klev] * mymsk),
            ("g) surf", terms3D[surf_key][klev] * mymsk),
            ("h) tend", terms3D["tend"][klev] * mymsk),
        ]

        resid = (
            terms3D["tend"][klev]
            - terms3D["ADVh"][klev]
            - terms3D["ADVr"][klev]
            - terms3D[dfh_key][klev]
            - terms3D[dfr_key][klev]
            - terms3D["KPP"][klev]
            - terms3D[surf_key][klev]
        ) * mymsk

        plot_defs.append(("i) resid", resid))

    # --- plot loop ---
    for ax, (title, field) in zip(axes.flat, plot_defs):
        b = get_aste_tracer(field, nfx, nfy)[0]
        vlev = np.nanmax(np.abs(b))

        if not np.isfinite(vlev) or vlev == 0:
            vlev = 1.0

        cb = ax.pcolormesh(b, cmap="seismic", vmin=-vlev, vmax=vlev)

        ax.pcolormesh(land_plot_aste, cmap="Greys", vmin=-2, vmax=4)
        ax.contour(depth_aste, levels=depth_levels, colors="k", linewidths=0.4, alpha=0.3)
        ax.contour(
            SI_masked,
            levels=[0.01, 0.99],
            colors=["lightgreen", "green"],
            linewidths=[1.0, 1.5],
            linestyles=["--", "--"],
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

def plot_budget_miniastefull(
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
    figsize=(18, 12),
    cbar_label=None,
    suptitle=None,
    contour_interval=100,
    save=False,
    save_name=None,
):
    """
    Plot a full mini-ASTE panel of budget terms for either heat or salt.

    Heat panels can include:
        ADVh, ADVr, ADVh+ADVr,
        DFhT, DFrT, DFhT+DFrT,
        surf/TFLUX, zconv_top, swtop,
        KPP, tend, resid

    Salt panels can include:
        ADVh, ADVr, ADVh+ADVr,
        DFhS, DFrS, DFhS+DFrS,
        surf, zconv_top, sptop,
        KPP, tend, resid
    """
    kind = kind.upper()
    if kind not in {"T", "S"}:
        raise ValueError("kind must be 'T' or 'S'")

    dfh_key = f"DFh{kind}"
    dfr_key = f"DFr{kind}"

    # ----------------------------
    # required keys
    # ----------------------------
    base_required = ["ADVh", "ADVr", dfh_key, dfr_key, "KPP", "tend"]
    missing = [k for k in base_required if k not in terms3D]
    if missing:
        raise KeyError(f"Missing required keys in terms3D: {missing}")

    # total surface term key
    if kind == "T":
        if "surf" in terms3D:
            surf_key = "surf"
        elif "TFLUX" in terms3D:
            surf_key = "TFLUX"
        else:
            raise KeyError("For heat, terms3D must contain either 'surf' or 'TFLUX'")
        split_flux_key = "swtop"
    else:
        if "surf" not in terms3D:
            raise KeyError("For salt, terms3D must contain key 'surf'")
        surf_key = "surf"
        split_flux_key = "sptop"

    has_surface_split = ("zconv_top" in terms3D) and (split_flux_key in terms3D)

    if cbar_label is None:
        cbar_label = "degC.m^3/s" if kind == "T" else "psu.m^3/s"

    if suptitle is None:
        suptitle = f"Contributions to {'Heat' if kind == 'T' else 'Salt'} Budget at k={klev}"

    fig, axes = plt.subplots(3, 4, figsize=figsize)
    axes_flat = list(axes.flat)

    # ----------------------------
    # static fields
    # ----------------------------
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

    # ----------------------------
    # residual
    # ----------------------------
    # Use only the total surface term here to avoid double counting.
    resid = (
        terms3D["tend"][klev]
        - terms3D["ADVh"][klev]
        - terms3D["ADVr"][klev]
        - terms3D[dfh_key][klev]
        - terms3D[dfr_key][klev]
        - terms3D["KPP"][klev]
        - terms3D[surf_key][klev]
    ) * mymsk

    # ----------------------------
    # panel definitions
    # ----------------------------
    plot_defs = [
        ("a) ADVh", terms3D["ADVh"][klev] * mymsk),
        ("b) ADVr", terms3D["ADVr"][klev] * mymsk),
        ("c) ADVh + ADVr", (terms3D["ADVh"][klev] + terms3D["ADVr"][klev]) * mymsk),
        ("d) " + dfh_key, terms3D[dfh_key][klev] * mymsk),
        ("e) " + dfr_key, terms3D[dfr_key][klev] * mymsk),
        ("f) " + dfh_key + " + " + dfr_key,
         (terms3D[dfh_key][klev] + terms3D[dfr_key][klev]) * mymsk),
        ("g) surf", terms3D[surf_key][klev] * mymsk),
    ]

    if has_surface_split:
        plot_defs.extend([
            ("h) zconv_top", terms3D["zconv_top"][klev] * mymsk),
            (f"i) {split_flux_key}", terms3D[split_flux_key][klev] * mymsk),
            ("j) KPP", terms3D["KPP"][klev] * mymsk),
            ("k) tend", terms3D["tend"][klev] * mymsk),
            ("l) resid", resid),
        ])
    else:
        plot_defs.extend([
            ("h) KPP", terms3D["KPP"][klev] * mymsk),
            ("i) tend", terms3D["tend"][klev] * mymsk),
            ("j) resid", resid),
        ])

    # ----------------------------
    # plotting helper
    # ----------------------------
    def _plot_one(ax, title, field):
        b = get_aste_tracer(field, nfx, nfy)[0]
        vlev = np.nanmax(np.abs(b))
        if not np.isfinite(vlev) or vlev == 0:
            vlev = 1.0

        cb = ax.pcolormesh(b, cmap="seismic", vmin=-vlev, vmax=vlev)

        ax.pcolormesh(land_plot_aste, cmap="Greys", vmin=-2, vmax=4)
        ax.contour(
            depth_aste,
            levels=depth_levels,
            colors="k",
            linewidths=0.4,
            alpha=0.3,
        )
        ax.contour(
            SI_masked,
            levels=[0.01,0.5, 0.99],
            colors=["lightgreen", "yellowgreen","green"],
            linewidths=[1.0, 1.8],
            linestyles=["--", "--"],
            zorder=10,
        )

        plt.colorbar(cb, ax=ax, label=cbar_label)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(title)

    # ----------------------------
    # draw panels
    # ----------------------------
    for ax, (title, field) in zip(axes_flat, plot_defs):
        _plot_one(ax, title, field)

    # turn off unused panels
    for ax in axes_flat[len(plot_defs):]:
        ax.axis("off")

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

# another function to plot the map view of theta and salt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D


def plot_theta_salt_maps_miniaste(
    theta,
    salt,
    mygrid,
    mymsk,
    SI_masked,
    nfx,
    nfy,
    get_aste_tracer,
    landmsk,
    section_mask,
    klev=0,
    xlim=(140, 180),
    ylim=(145, 180),
    figsize=(14, 6),
    salt_cmap="viridis",
    theta_vmin=-2,
    theta_vmax=7,
    salt_vmin=None,
    salt_vmax=None,
    theta_label="Theta (°C)",
    salt_label="Salinity (psu)",
    suptitle=None,
    contour_interval=100,
    save=False,
    save_name=None,
):
    """
    Plot 2D maps of theta and salinity on the mini-ASTE layout at a given z level.

    Features
    --------
    - custom red-white-blue theta colormap over [-2, 7]
    - 0 is forced to white using TwoSlopeNorm
    - user-specified vmin/vmax for salt
    - dashed contour line showing theta[0] = 3 on both panels
    - pcolormesh fields masked using hFacC
    - gray land mask derived from hFacC at the selected klev
    """

    if suptitle is None:
        suptitle = f"Theta and Salinity at k={klev}"
    landmsk[landmsk==0] = np.nan
    pflev = 0   # for the polar front level

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- custom theta colormap: blue -> white -> red ---
    theta_cmap = LinearSegmentedColormap.from_list(
        "custom_theta_rwbr",
        [
            (0.00, "#08306b"),  # deep blue
            (0.35, "#6baed6"),  # light blue
            (0.50, "#ffffff"),  # white at 0
            (0.65, "#fcae91"),  # light red
            (1.00, "#99000d"),  # deep red
        ],
    )
    theta_norm = TwoSlopeNorm(vmin=theta_vmin, vcenter=0.0, vmax=theta_vmax)

    # --- static fields ---
    depth_plot = mygrid["Depth"].copy().astype(float)
    depth_plot[depth_plot <= 0] = np.nan
    depth_aste = get_aste_tracer(depth_plot, nfx, nfy)[0]

    depth_levels = np.arange(
        contour_interval,
        np.nanmax(depth_plot) + contour_interval,
        contour_interval,
    )

    # --- hFacC masks ---
    hfC = mygrid["hFacC"].copy().astype(float)

    # mask data for plotting
    hfCmsk = hfC.copy()
    hfCmsk[hfCmsk < 1] = np.nan

    # gray land/blocked-cell overlay at this k level
    land_plot_k = hfC[klev].copy()
    land_plot_k[land_plot_k > 0] = np.nan   # wet cells transparent
    land_plot_k[land_plot_k == 0] = 1       # dry cells gray
    land_plot_k_aste = get_aste_tracer(land_plot_k, nfx, nfy)[0]

    # --- fields at plotting level ---
    theta2d = theta[klev] * mymsk * hfCmsk[klev]
    salt2d  = salt[klev]  * mymsk * hfCmsk[klev]

    theta_aste = get_aste_tracer(theta2d, nfx, nfy)[0]
    salt_aste  = get_aste_tracer(salt2d,  nfx, nfy)[0]

    # --- theta[0] = 3 contour, independent of klev ---
    theta0_line = theta[0] * mymsk * hfCmsk[0]
    theta0_line_aste = get_aste_tracer(theta0_line, nfx, nfy)[0]

    # --- infer salt limits if not provided ---
    if salt_vmin is None:
        salt_vmin = np.nanmin(salt_aste)
    if salt_vmax is None:
        salt_vmax = np.nanmax(salt_aste)

    # --- theta panel ---
    pm0 = axes[0].pcolormesh(
        theta_aste,
        cmap=theta_cmap,
        norm=theta_norm,
    )
    axes[0].pcolormesh(get_aste_tracer(landmsk,nfx,nfy)[0], cmap="Greys", vmin=0, vmax=2)
    axes[0].contour(depth_aste, levels=depth_levels, colors="k", linewidths=0.4, alpha=0.3)
    axes[0].contour(
        SI_masked,
        levels=[0.01, 1.0],
        colors=["lightgreen", "green"],
        linewidths=[1.0, 1.8],
        linestyles=["--", "-"],
        zorder=10,
    )
    axes[0].contour(
        theta0_line_aste,
        levels=[pflev],
        colors="k",
        linewidths=1.5,
        linestyles="--",
        zorder=11,
    )
    axes[0].pcolormesh(get_aste_tracer(section_mask,nfx,nfy)[0],cmap="Greys",vmin=-5,vmax=1,alpha=0.5)
    plt.colorbar(pm0, ax=axes[0], label=theta_label,extend="max")
    axes[0].set_xlim(*xlim)
    axes[0].set_ylim(*ylim)
    axes[0].set_title("Theta")

    # --- salt panel ---
    pm1 = axes[1].pcolormesh(
        salt_aste,
        cmap=salt_cmap,
        vmin=salt_vmin,
        vmax=salt_vmax,
    )
    axes[1].pcolormesh(get_aste_tracer(landmsk,nfx,nfy)[0], cmap="Greys", vmin=0, vmax=2)
    axes[1].contour(depth_aste, levels=depth_levels, colors="k", linewidths=0.4, alpha=0.3)
    axes[1].contour(
        SI_masked,
        levels=[0.01, 1.0],
        colors=["lightgreen", "green"],
        linewidths=[1.0, 1.8],
        linestyles=["--", "-"],
        zorder=10,
    )
    axes[1].contour(
        theta0_line_aste,
        levels=[pflev],
        colors="k",
        linewidths=1.5,
        linestyles="--",
        zorder=11,
    )
    plt.colorbar(pm1, ax=axes[1], label=salt_label,extend="both")
    axes[1].set_xlim(*xlim)
    axes[1].set_ylim(*ylim)
    axes[1].set_title("Salinity")
    axes[1].pcolormesh(get_aste_tracer(section_mask,nfx,nfy)[0],cmap="Greys",vmin=-5,vmax=1,alpha=0.5)

    plt.suptitle(suptitle)
    plt.tight_layout()

    legend_handles = [
        Line2D([0], [0], color="k", linestyle="--", linewidth=1.5,
               label=f"SST = PF ({pflev}°C)"),
        Line2D([0], [0], color="lightgreen", linestyle="--", linewidth=1.5,
               label="Ice edge (SIarea = 0.15)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2,
               frameon=True, bbox_to_anchor=(0.1, 1.02))

    if save:
        os.makedirs("figs", exist_ok=True)

        if save_name is None:
            save_name = f"miniaste_theta_salt_k{klev}.png"

        save_path = os.path.join("figs", save_name)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    return fig, axes

# add a function to plot the budget terms as vertical layers

def plot_budget_section(
    terms3D,
    x_section,
    y_section,
    mygrid,
    kind="T",
    figsize=(14, 10),
    zmax=350,
    rho_for_contours=None,
    rho_clevs=np.arange(27.0, 28.3, 0.05),
    cmap="seismic",
    cbar_label=None,
    suptitle=None,
    save=False,
    save_name=None,
):
    """
    Plot a 3x3 panel of section plots for either heat or salt budget terms.
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
        cbar_label = "degC m$^3$/s" if kind == "T" else "psu m$^3$/s"

    if suptitle is None:
        suptitle = f"{'Heat' if kind == 'T' else 'Salt'} budget terms along section"

    x_section = np.asarray(x_section).astype(int)
    y_section = np.asarray(y_section).astype(int)

    if x_section.shape != y_section.shape:
        raise ValueError("x_section and y_section must have the same shape")

    # vertical coordinate
    z = np.cumsum(mygrid["DRF"].squeeze())

    # section distance
    if "DYG" in mygrid:
        dsec = mygrid["DYG"][x_section, y_section]
        dist_km = np.cumsum(dsec) / 1000.0
        dist_km = np.round(dist_km)
    else:
        dist_km = np.arange(len(x_section))

    sec_idx = np.arange(len(x_section))

    # land mask for section
    land_hfc = mygrid["hFacC"][:, x_section, y_section]
    land_sec = np.full_like(land_hfc, np.nan, dtype=float)
    land_sec[np.isnan(land_hfc)] = 1

    # extract section terms
    advh_sec = terms3D["ADVh"][:, x_section, y_section].copy()
    advr_sec = terms3D["ADVr"][:, x_section, y_section].copy()
    dfh_sec  = terms3D[dfh_key][:, x_section, y_section].copy()
    dfr_sec  = terms3D[dfr_key][:, x_section, y_section].copy()
    kpp_sec  = terms3D["KPP"][:, x_section, y_section].copy()
    surf_sec = terms3D["surf"][:, x_section, y_section].copy()
    tend_sec = terms3D["tend"][:, x_section, y_section].copy()

    resid_sec = (
        tend_sec
        - advh_sec
        - advr_sec
        - dfh_sec
        - dfr_sec
        - kpp_sec
        - surf_sec
    )

    plot_defs = [
        ("a) ADVh", advh_sec),
        ("b) ADVh + ADVr", advh_sec + advr_sec),
        ("c) " + dfh_key + " + " + dfr_key, dfh_sec + dfr_sec),
        ("d) " + dfh_key, dfh_sec),
        ("e) " + dfr_key, dfr_sec),
        ("f) KPP", kpp_sec),
        ("g) surf", surf_sec),
        ("h) tend", tend_sec),
        ("i) resid", resid_sec),
    ]

    rho_sec = None
    if rho_for_contours is not None:
        rho_sec = rho_for_contours[:, x_section, y_section].copy()
        rho_sec[rho_sec == 0] = np.nan

    fig, axes = plt.subplots(3, 3, figsize=figsize, sharex=True, sharey=True)

    tick_idx = sec_idx[::max(1, len(sec_idx)//6)]
    tick_labels = dist_km.astype(int)[::max(1, len(sec_idx)//6)]

    for ax, (title, field) in zip(axes.flat, plot_defs):
        field = field.copy()
        field[field == 0] = np.nan

        vlev = np.nanmax(np.abs(field))
        if not np.isfinite(vlev) or vlev == 0:
            vlev = 1.0

        pcm = ax.pcolormesh(
            sec_idx,
            z,
            field,
            shading="auto",
            cmap=cmap,
            vmin=-vlev,
            vmax=vlev,
        )

        ax.pcolormesh(
            sec_idx,
            z,
            land_sec,
            cmap="Greys",
            vmin=0,
            vmax=2,
            shading="auto",
        )

        if rho_sec is not None:
            ax.contour(
                sec_idx,
                z,
                rho_sec,
                levels=rho_clevs,
                colors="k",
                linewidths=0.5,
                alpha=0.7,
            )

        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels)

        ax.invert_yaxis()
        ax.invert_xaxis()
        ax.set_ylim(zmax, 5)

        ax.set_title(title, fontsize=10)
        plt.colorbar(pcm, ax=ax, label=cbar_label)

    for ax in axes[-1, :]:
        ax.set_xlabel("Distance along section (km)")

    for ax in axes[:, 0]:
        ax.set_ylabel("Depth (m)")

    plt.suptitle(suptitle)
    plt.tight_layout()

    if save:
        os.makedirs("figs", exist_ok=True)

        if save_name is None:
            budget_type = "heat" if kind == "T" else "salt"
            save_name = f"{budget_type}_budget_section.png"

        save_path = os.path.join("figs", save_name)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    return fig, axes

# function to peek at any individual field in the ASTE grid
def plot_any_BS(fld,nfx,nfy,landmsk,mymsk,unit):
    fig = plt.figure(figsize = (5,5))
    ax = plt.subplot(111)

    fld_plot = get_aste_tracer(fld, nfx, nfy)[0] * get_aste_tracer(mymsk,nfx,nfy)[0]

    vmax = np.nanmax(np.abs(fld))
    vmin = -vmax

    cb = ax.pcolormesh(
        fld_plot,
        cmap='seismic',
        vmin=vmin,
        vmax=vmax
    )

    ax.pcolormesh(
        get_aste_tracer(landmsk, nfx, nfy)[0],
        cmap='Greys',
        vmin=0,
        vmax=2
    )

    plt.colorbar(cb, ax=ax, label=unit)
    ax.set_xlim(140, 180)
    ax.set_ylim(140, 180)