"""
Plotting helpers for Lab Sea SFLUX/TFLUX diagnostic breakdowns.

Use from labsea_surfacebreakdown.ipynb after the existing diagnostic arrays
have been loaded into dictionaries such as budg1, budg2, extra_seaice,
extra_exf, and state2d.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np


RHO_CONST_FRESH = 1000.0


DIAGNOSTIC_GROUPS = OrderedDict(
    [
        (
            "budg2d_zflux_set1",
            [
                "oceFWflx",
                "SIatmFW",
                "TFLUX",
                "SItflux",
                "SFLUX",
                "oceQsw",
                "oceSPflx",
            ],
        ),
        (
            "budg2d_zflux_set2",
            [
                "WTHMASS",
                "WSLTMASS",
                "oceSflux",
                "oceQnet",
                "SIatmQnt",
                "SIaaflux",
                "SIsnPrcp",
                "SIacSubl",
            ],
        ),
        (
            "flux_breakdown_extra_seaice",
            [
                "surForcT",
                "surForcS",
                "SIempmr",
                "SIfwSubl",
                "SIrsSubl",
                "SIhl",
                "SIqneto",
                "SIqneti",
                "oceFreez",
            ],
        ),
        ("flux_breakdown_extra_exf", ["EXFsnow"]),
        (
            "state_2d_set1",
            [
                "ETAN",
                "SIarea",
                "SIheff",
                "SIhsnow",
                "DETADT2",
                "PHIBOT",
                "sIceLoad",
                "MXLDEPTH",
                "SIatmQnt",
                "SIatmFW",
                "oceQnet",
                "oceFWflx",
                "oceTAUX",
                "oceTAUY",
                "ADVxHEFF",
                "ADVyHEFF",
                "ADVxSNOW",
                "ADVySNOW",
                "SIuice",
                "SIvice",
                "ETANSQ",
                "oceSPDep",
            ],
        ),
    ]
)


def _has(data, *names):
    return all(data.get(name) is not None for name in names)


def _add(out, data, name, value, required):
    if all(data.get(key) is not None for key in required):
        out[name] = value


def combine_diagnostic_dicts(*dicts):
    """Merge rdmds diagnostic dictionaries and strip MITgcm field padding."""
    out = {}
    for dct in dicts:
        if dct is None:
            continue
        for key, value in dct.items():
            out[key.strip()] = value
    return out


def available_diagnostic_text(data):
    """Return a compact report of available and missing expected diagnostics."""
    lines = []
    for group, names in DIAGNOSTIC_GROUPS.items():
        available = [name for name in names if data.get(name) is not None]
        missing = [name for name in names if data.get(name) is None]
        lines.append(f"{group}:")
        lines.append("  available: " + (", ".join(available) if available else "none"))
        if missing:
            lines.append("  missing:   " + ", ".join(missing))
    return "\n".join(lines)


def sflux_options(data, salt_surface=None, rho_const_fresh=RHO_CONST_FRESH):
    """
    Candidate SFLUX breakdown maps.

    From diags_oceanic_surf_flux.F:
      SFLUX = surForcS + PmEpR*SALT_surface when real freshwater flux is active.

    The diagnostics use positive-down conventions for oceFWflx/oceSflux.
    Since oceFWflx = -EmPmR and PmEpR is the precipitation-minus-evaporation
    mass flux, the freshwater advection contribution is expected to be close to
    oceFWflx*SALT_surface when salt_surface is supplied in g/kg.
    """
    out = OrderedDict()
    for name in ["SFLUX", "surForcS", "oceSflux", "oceSPflx", "oceFWflx", "SIatmFW", "WSLTMASS"]:
        if data.get(name) is not None:
            out[name] = data[name]

    _add(out, data, "oceSflux - oceSPflx", data["oceSflux"] - data["oceSPflx"], ["oceSflux", "oceSPflx"])
    _add(out, data, "SFLUX - surForcS", data["SFLUX"] - data["surForcS"], ["SFLUX", "surForcS"])
    _add(out, data, "SFLUX - oceSflux", data["SFLUX"] - data["oceSflux"], ["SFLUX", "oceSflux"])
    _add(
        out,
        data,
        "SFLUX - (oceSflux - oceSPflx)",
        data["SFLUX"] - (data["oceSflux"] - data["oceSPflx"]),
        ["SFLUX", "oceSflux", "oceSPflx"],
    )

    if salt_surface is not None and data.get("oceFWflx") is not None:
        fw_salt = data["oceFWflx"] * salt_surface
        out["oceFWflx * SALT_surface"] = fw_salt
        if _has(data, "surForcS"):
            out["surForcS + oceFWflx*SALT_surface"] = data["surForcS"] + fw_salt
        if _has(data, "SFLUX", "surForcS"):
            out["(SFLUX - surForcS) - oceFWflx*SALT_surface"] = data["SFLUX"] - data["surForcS"] - fw_salt
        if _has(data, "oceSflux"):
            out["oceSflux + oceFWflx*SALT_surface"] = data["oceSflux"] + fw_salt

    # If SALT is not available, a crude seawater reference is still useful for
    # checking whether the residual has the scale/sign of real freshwater flux.
    if data.get("oceFWflx") is not None:
        out["oceFWflx * 34.8 reference"] = data["oceFWflx"] * 34.8
        out["oceFWflx * 35.0 reference"] = data["oceFWflx"] * 35.0

    if data.get("SFLUX") is not None and data.get("WSLTMASS") is not None:
        out["SFLUX + WSLTMASS"] = data["SFLUX"] + data["WSLTMASS"]
        out["SFLUX - WSLTMASS"] = data["SFLUX"] - data["WSLTMASS"]

    return out


def tflux_options(data, theta_surface=None, heat_capacity_cp=3994.0):
    """
    Candidate TFLUX breakdown maps.

    From diags_oceanic_surf_flux.F:
      TFLUX = surForcT + oceFreez - raw_Qsw + PmEpR*THETA_surface*Cp
    when SHORTWAVE_HEATING and real freshwater flux are active.

    Here surForcT and oceFreez are already W/m2 diagnostics. The output
    diagnostic oceQsw is -raw_Qsw, so the direct reconstruction uses
    +oceQsw. The -oceQsw option is kept as a sign-check panel.
    """
    out = OrderedDict()
    for name in [
        "TFLUX",
        "surForcT",
        "oceFreez",
        "oceQnet",
        "oceQsw",
        "SItflux",
        "SIatmQnt",
        "SIaaflux",
        "SIhl",
        "SIqneto",
        "SIqneti",
        "WTHMASS",
    ]:
        if data.get(name) is not None:
            out[name] = data[name]

    if _has(data, "surForcT", "oceFreez"):
        out["surForcT + oceFreez"] = data["surForcT"] + data["oceFreez"]
    if _has(data, "surForcT", "oceQsw"):
        out["surForcT + oceQsw"] = data["surForcT"] + data["oceQsw"]
        out["surForcT - oceQsw"] = data["surForcT"] - data["oceQsw"]
    if _has(data, "surForcT", "oceFreez", "oceQsw"):
        out["surForcT + oceFreez + oceQsw"] = data["surForcT"] + data["oceFreez"] + data["oceQsw"]
        out["surForcT + oceFreez - oceQsw"] = data["surForcT"] + data["oceFreez"] - data["oceQsw"]

    _add(out, data, "TFLUX - surForcT", data["TFLUX"] - data["surForcT"], ["TFLUX", "surForcT"])
    _add(out, data, "TFLUX - oceQnet", data["TFLUX"] - data["oceQnet"], ["TFLUX", "oceQnet"])
    _add(out, data, "TFLUX - oceQnet - SIaaflux", data["TFLUX"] - data["oceQnet"] - data["SIaaflux"], ["TFLUX", "oceQnet", "SIaaflux"])
    _add(out, data, "SItflux - SIatmQnt", data["SItflux"] - data["SIatmQnt"], ["SItflux", "SIatmQnt"])
    _add(out, data, "SItflux - SIatmQnt - SIhl", data["SItflux"] - data["SIatmQnt"] - data["SIhl"], ["SItflux", "SIatmQnt", "SIhl"])

    if theta_surface is not None and data.get("oceFWflx") is not None:
        fw_heat = data["oceFWflx"] * theta_surface * heat_capacity_cp
        out["oceFWflx * THETA_surface * Cp"] = fw_heat
        if _has(data, "surForcT"):
            out["surForcT + FW_heat"] = data["surForcT"] + fw_heat
        if _has(data, "surForcT", "oceFreez", "oceQsw"):
            out["surForcT + oceFreez + oceQsw + FW_heat"] = (
                data["surForcT"] + data["oceFreez"] + data["oceQsw"] + fw_heat
            )
        if _has(data, "TFLUX", "surForcT"):
            out["(TFLUX - surForcT) - FW_heat"] = data["TFLUX"] - data["surForcT"] - fw_heat

    return out


def mask_options(options, mask=None):
    """Apply a 2-D mask, usually hFacC[0], to every non-empty map."""
    if mask is None:
        return options
    return OrderedDict((name, value * mask) for name, value in options.items())


def plot_option_grid(options, units="", ncols=4, scale=1.0, robust_pct=99.0, cmap="seismic"):
    """
    Plot every map in options with a symmetric color scale per panel.

    Set scale=10 or scale=100 if you want to intentionally widen the limits
    for very spiky fields.
    """
    options = OrderedDict((k, v) for k, v in options.items() if v is not None)
    if not options:
        raise ValueError("No fields to plot.")

    ncols = min(ncols, len(options))
    nrows = int(math.ceil(len(options) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows), squeeze=False)

    for ax, (name, fld) in zip(axes.ravel(), options.items()):
        values = np.asarray(fld, dtype=float)
        finite = np.isfinite(values)
        if finite.any():
            vlev = np.nanpercentile(np.abs(values[finite]), robust_pct) * scale
        else:
            vlev = 1.0
        if not np.isfinite(vlev) or vlev == 0:
            vlev = 1.0
        pcm = ax.pcolormesh(values, cmap=cmap, vmin=-vlev, vmax=vlev)
        fig.colorbar(pcm, ax=ax, label=units)
        ax.set_title(name, fontsize=10)
        ax.set_aspect("equal")

    for ax in axes.ravel()[len(options) :]:
        ax.axis("off")

    fig.tight_layout()
    return fig, axes


def print_integrals(options, area=None, label=""):
    """Print min/max/mean and optionally area integrals for each option."""
    prefix = f"{label}: " if label else ""
    for name, fld in options.items():
        values = np.asarray(fld, dtype=float)
        finite = np.isfinite(values)
        if not finite.any():
            print(f"{prefix}{name}: all nan")
            continue
        msg = (
            f"{prefix}{name}: "
            f"min={np.nanmin(values): .6e}, "
            f"mean={np.nanmean(values): .6e}, "
            f"max={np.nanmax(values): .6e}"
        )
        if area is not None:
            msg += f", area_sum={np.nansum(values * area): .6e}"
        print(msg)
