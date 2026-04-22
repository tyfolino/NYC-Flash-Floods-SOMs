"""
Supplementary Figure S4 — FFE SOM Node Weights (left) and Raw Composites (right)

Default (4-row × 2-column) layout. Each row is one SOM node (A1, A2, B1, B2).
Left column : standardized Z500 (contoured) + theta-e (shaded) node weights.
Right column : raw mean Z500 in dam (contoured) + theta-e in K (shaded)
               composited over FF days assigned to that node.

Wide layout (--wide flag): 2-row × 4-column.
Left 2×2  : node weights (A1/A2 top, B1/B2 bottom)
Right 2×2 : raw composites (A1/A2 top, B1/B2 bottom)

Usage:
    python -m figure_scripts.figS06_ffe_som_node_weights_and_composites
    python -m figure_scripts.figS06_ffe_som_node_weights_and_composites --wide
"""

import argparse
import os

import cartopy.crs as ccrs
import cmweather  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from som_analysis.config import setup_plotting
from som_analysis.helpers import add_map_features, get_node_indices, node_label

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures/figS06")
os.makedirs(OUT_DIR, exist_ok=True)

CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "figs",
    "Z500-and-thetae-SOM",
    ".cache",
    "som_results.npz",
)
THETAE_FFE_PATH = "/mnt/drive2/SOM_intermediate_files/era5_thetae_ffe.nc"
Z500_FFE_PATH = "/mnt/drive2/SOM_intermediate_files/era5_Z500_ffe.nc"

# ── Figure / SOM parameters ───────────────────────────────────────────────────
XDIM, YDIM = 2, 2
FIG_WIDTH = 6.0
FIG_HEIGHT = 5.8
FIG_WIDTH_WIDE = 11.0
FIG_HEIGHT_WIDE = 3.5
DPI_RASTER = 300

# Contour levels
LEVELS_Z_WEIGHTS = np.arange(-1.4, 1.41, 0.2)  # standardized anomaly
LEVELS_MOIST_WEIGHTS = np.arange(-1.2, 1.21, 0.2)
LEVELS_Z_RAW = range(549, 598, 2)  # dam — shared range with Fig. S3
LEVELS_THETAE_RAW = np.arange(295, 341, 5)  # K — shared range with Fig. S3

# Node traversal order for 4 rows: A1, A2, B1, B2
NODE_ORDER = [(i, j) for j in range(YDIM) for i in range(XDIM)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--wide",
        action="store_true",
        help="Use wide 2×4 layout for presentations",
    )
    return p.parse_args()


def main():
    args = parse_args()
    setup_plotting()

    # ── Load cache ────────────────────────────────────────────────────────────
    print(f"Loading cache from {CACHE_PATH} ...")
    cached = np.load(CACHE_PATH)
    z500_nodes = cached["z500_nodes"]  # (xdim, ydim, nlat, nlon) — standardized
    moist_nodes = cached["moist_nodes"]  # (xdim, ydim, nlat, nlon) — standardized
    bmus = cached["bmus"]  # (121, 2)

    # ── Load raw FFE fields for composites ───────────────────────────────────
    print("Loading raw FFE fields ...")
    ds_thetae = xr.load_dataset(THETAE_FFE_PATH)
    ds_z500 = xr.load_dataset(Z500_FFE_PATH)
    thetae_da = ds_thetae["theta_e"]  # (time, lat, lon) in K
    z500_da = ds_z500["z"]  # (time, lat, lon) in m^2/s^2 → divide by 98.1 for dam

    lat = ds_thetae["latitude"].values
    lon = ds_thetae["longitude"].values

    # ── Build figure ──────────────────────────────────────────────────────────
    proj = ccrs.PlateCarree()
    if args.wide:
        nrows, ncols = 2, 4
        fig_w, fig_h = FIG_WIDTH_WIDE, FIG_HEIGHT_WIDE
    else:
        nrows, ncols = 4, 2
        fig_w, fig_h = FIG_WIDTH, FIG_HEIGHT

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        subplot_kw={"projection": proj},
        constrained_layout=True,
        dpi=DPI_RASTER,
    )
    fig.get_layout_engine().set(hspace=0.0, wspace=0.02)

    im_weight = None
    im_comp = None

    for i, j in NODE_ORDER:
        if args.wide:
            ax_wt = axes[j, i]
            ax_comp = axes[j, i + 2]
        else:
            row = j * XDIM + i
            ax_wt = axes[row, 0]
            ax_comp = axes[row, 1]
        lbl = node_label(i, j)
        idx = get_node_indices(bmus, i, j)
        n = len(idx)

        # ── Left: node weight pattern ─────────────────────────────────────────
        im_weight = ax_wt.contourf(
            lon,
            lat,
            moist_nodes[i, j],
            cmap="balance",
            levels=LEVELS_MOIST_WEIGHTS,
            transform=proj,
            extend="both",
        )
        cn = ax_wt.contour(
            lon,
            lat,
            z500_nodes[i, j],
            colors="black",
            linewidths=0.6,
            levels=LEVELS_Z_WEIGHTS,
            transform=proj,
        )
        ax_wt.clabel(cn, inline=True, fontsize=6.5, fmt="%.1f")
        add_map_features(ax_wt)

        # Node label outside upper-left; N outside upper-right
        ax_wt.text(
            0.0,
            1.01,
            lbl,
            transform=ax_wt.transAxes,
            fontsize=7.5,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        ax_wt.text(
            1.0,
            1.01,
            f"$n$={n}",
            transform=ax_wt.transAxes,
            fontsize=8.0,
            ha="right",
            va="bottom",
        )

        # ── Right: raw composite ──────────────────────────────────────────────
        thetae_comp = thetae_da.isel(valid_time=idx).mean("valid_time").values
        z500_comp = z500_da.isel(valid_time=idx).mean("valid_time").values / 98.1

        im_comp = ax_comp.contourf(
            lon,
            lat,
            thetae_comp,
            cmap="BuPu",
            levels=LEVELS_THETAE_RAW,
            transform=proj,
            extend="both",
        )
        cn2 = ax_comp.contour(
            lon,
            lat,
            z500_comp,
            colors="black",
            linewidths=0.4,
            levels=LEVELS_Z_RAW,
            transform=proj,
        )
        ax_comp.clabel(cn2, inline=True, fontsize=6.5, fmt="%.0f")
        add_map_features(ax_comp)

        # Node label outside upper-left on composite panel too
        ax_comp.text(
            0.0,
            1.01,
            lbl,
            transform=ax_comp.transAxes,
            fontsize=7.5,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    # ── Colorbars ─────────────────────────────────────────────────────────────
    if args.wide:
        ax_wt_group = axes[:, :2].ravel().tolist()
        ax_comp_group = axes[:, 2:].ravel().tolist()
    else:
        ax_wt_group = axes[:, 0]
        ax_comp_group = axes[:, 1]

    cb_wt = fig.colorbar(im_weight, ax=ax_wt_group, shrink=0.6, pad=0.02, aspect=25)
    cb_wt.set_label(r"Standardized 850-hPa $\theta_e$ Anomaly", fontsize=8)
    cb_wt.ax.tick_params(labelsize=7)

    cb_comp = fig.colorbar(im_comp, ax=ax_comp_group, shrink=0.6, pad=0.02, aspect=25)
    cb_comp.set_label(r"850-hPa $\theta_e$ (K)", fontsize=8)
    cb_comp.ax.tick_params(labelsize=7)

    # ── Save ──────────────────────────────────────────────────────────────────
    suffix = "_wide" if args.wide else ""
    base = os.path.join(OUT_DIR, f"figS06_ffe_som{suffix}")
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.png", dpi=DPI_RASTER)
    fig.savefig(f"{base}.tiff", dpi=DPI_RASTER)
    print(f"Saved to {OUT_DIR}/")
    plt.close()


if __name__ == "__main__":
    main()
