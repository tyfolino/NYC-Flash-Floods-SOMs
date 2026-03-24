"""
Supplementary Figure S3 — All-Days SOM Raw Composites (Z500 + 850-hPa theta-e)

5-row × 4-column layout matching Fig. 1, but showing the raw composite mean
Z500 (contoured, dam) and 850-hPa theta-e (shaded, K) for all days assigned
to each node.

Usage:
    python -m figure_scripts.figS03_alldays_raw_composites
"""

import os

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from som_analysis.config import SOM_INTERMEDIATE_PATH, setup_plotting
from som_analysis.helpers import add_map_features, compute_composites, node_label

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures", "figS03")
os.makedirs(OUT_DIR, exist_ok=True)

CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "figs",
    "Z500-and-thetae-alldays-SOM",
    ".cache",
    "som_results.npz",
)

# ── SOM / figure parameters ───────────────────────────────────────────────────
XDIM, YDIM = 5, 4
FIG_WIDTH = 7.0
FIG_HEIGHT = 3.8
DPI_RASTER = 300

LEVELS_THETAE = np.arange(295, 341, 5)  # K — shared range with Fig. 2
LEVELS_Z = range(549, 598, 2)  # dam — shared range with Fig. 2


def _ff_label(counts, totals, risk, i, j):
    n = int(counts[i, j])
    tot = int(totals[i, j])
    r = risk[i, j]
    if np.isnan(r) or n == 0:
        return f"FF=0/{tot}"
    return f"FF={n}/{tot} ({r * 100:.1f}\\%)"


def main():
    setup_plotting()

    # ── Load cache ────────────────────────────────────────────────────────────
    print(f"Loading cache from {CACHE_PATH} ...")
    cached = np.load(CACHE_PATH)
    bmus = cached["bmus"]  # (n_days, 2)
    counts = cached["counts"]
    totals = cached["totals"]
    risk = cached["risk"]
    lat = cached["lat"]
    lon = cached["lon"]

    # ── Load raw daily fields ─────────────────────────────────────────────────
    print("Loading raw daily data ...")
    z500_daily = xr.load_dataarray(f"{SOM_INTERMEDIATE_PATH}era5_Z500_daily.nc")
    thetae_daily = xr.load_dataarray(f"{SOM_INTERMEDIATE_PATH}era5_thetae_daily.nc")

    # Auto-detect time dimension
    z500_time_dim = "valid_time" if "valid_time" in z500_daily.dims else "time"
    thetae_time_dim = "valid_time" if "valid_time" in thetae_daily.dims else "time"

    # ── Compute node composites ───────────────────────────────────────────────
    print("Computing composites ...")
    z500_comp, _ = compute_composites(
        z500_daily, bmus, XDIM, YDIM, time_dim=z500_time_dim
    )
    thetae_comp, _ = compute_composites(
        thetae_daily, bmus, XDIM, YDIM, time_dim=thetae_time_dim
    )

    # Z500: Pa → dam
    z500_comp = z500_comp / 98.1

    # ── Build figure ──────────────────────────────────────────────────────────
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        YDIM,
        XDIM,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        subplot_kw={"projection": proj},
        constrained_layout=True,
        dpi=DPI_RASTER,
    )
    fig.get_layout_engine().set(hspace=0.0, wspace=0.02)

    for i in range(XDIM):
        for j in range(YDIM):
            ax = axes[j, i]

            # θe shaded
            im = ax.contourf(
                lon,
                lat,
                thetae_comp[i, j],
                cmap="BuPu",
                levels=LEVELS_THETAE,
                transform=proj,
                extend="both",
            )

            # Z500 contoured
            cn = ax.contour(
                lon,
                lat,
                z500_comp[i, j],
                colors="black",
                linewidths=0.4,
                levels=LEVELS_Z,
                transform=proj,
            )
            ax.clabel(cn, inline=True, fontsize=3.5, fmt="%d")

            add_map_features(ax)

            # Node ID — bold, outside upper-left
            ax.text(
                0.0,
                1.01,
                node_label(i, j),
                transform=ax.transAxes,
                fontsize=5.5,
                fontweight="bold",
                ha="left",
                va="bottom",
            )

            # FF stats — outside upper-right
            ax.text(
                1.0,
                1.01,
                _ff_label(counts, totals, risk, i, j),
                transform=ax.transAxes,
                fontsize=5.0,
                ha="right",
                va="bottom",
            )

    # ── Shared colorbar ───────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02, aspect=30)
    cbar.set_label(r"850-hPa $\theta_e$ (K)", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    # ── Save ─────────────────────────────────────────────────────────────────
    base = os.path.join(OUT_DIR, "figS03_alldays_raw_composites")
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.png", dpi=DPI_RASTER)
    fig.savefig(f"{base}.tiff", dpi=DPI_RASTER)
    print(f"Saved to {OUT_DIR}/")
    plt.close()


if __name__ == "__main__":
    main()
