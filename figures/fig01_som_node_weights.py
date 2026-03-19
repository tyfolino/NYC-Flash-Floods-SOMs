"""
Figure 1 — All-Days SOM Node Weight Patterns (Z500 + 850-hPa theta-e)

Publication figure for NYC Flash Flood SOMs paper.
Saves PDF, PNG (300 dpi), and TIFF (300 dpi) to output/fig01/.

Usage:
    python -m figures.fig01_som_node_weights
"""

import os

import cartopy.crs as ccrs
import cmweather  # noqa: F401 — registers 'balance' and other cmweather colormaps
import matplotlib.pyplot as plt
import numpy as np

from som_analysis.config import MOISTURE_CONFIGS, setup_plotting
from som_analysis.helpers import add_map_features, node_label

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output", "fig01")
os.makedirs(OUT_DIR, exist_ok=True)

# ── SOM / figure parameters ───────────────────────────────────────────────────
XDIM, YDIM = 5, 4
MOISTURE_VAR = "thetae"

# Cache produced by train_alldays_som.py in the original analysis repo
CACHE_PATH = (
    "/home/janoski/nyc_flash_flood/figs/"
    "Z500-and-thetae-alldays-SOM/.cache/som_results.npz"
)

# GRL double-column width = 7 in; height tuned to 5x4 map grid
FIG_WIDTH = 7.0   # inches
FIG_HEIGHT = 3.8  # inches
DPI_RASTER = 300

# Z500 contour levels (standardized anomaly weights)
LEVELS_Z = np.arange(-1.4, 1.41, 0.2)
LEVELS_MOIST = np.arange(-1.2, 1.21, 0.2)  # tighter range; extend="both" handles outliers


def _ff_label(counts, totals, risk, i, j):
    """Build the FF risk annotation string for node (i, j)."""
    n = int(counts[i, j])
    tot = int(totals[i, j])
    r = risk[i, j]
    if np.isnan(r) or n == 0:
        return f"FF=0/{tot}"
    return f"FF={n}/{tot} ({r * 100:.1f}\\%)"


def main():
    setup_plotting()

    # ── Load cached SOM results ───────────────────────────────────────────────
    print(f"Loading cache from {CACHE_PATH} ...")
    cached = np.load(CACHE_PATH)
    z500_nodes = cached["z500_nodes"]   # (xdim, ydim, nlat, nlon)
    moist_nodes = cached["moist_nodes"] # (xdim, ydim, nlat, nlon)
    counts = cached["counts"]           # (xdim, ydim) — FF days per node
    totals = cached["totals"]           # (xdim, ydim) — all days per node
    risk = cached["risk"]               # (xdim, ydim)
    lat = cached["lat"]
    lon = cached["lon"]

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

            # Shaded: theta-e standardized anomaly
            im = ax.contourf(
                lon, lat, moist_nodes[i, j],
                cmap="balance",
                levels=LEVELS_MOIST,
                transform=proj,
                extend="both",
            )

            # Contoured: Z500 standardized anomaly
            cn = ax.contour(
                lon, lat, z500_nodes[i, j],
                colors="black",
                linewidths=0.4,
                levels=LEVELS_Z,
                transform=proj,
            )
            ax.clabel(cn, inline=True, fontsize=3.5, fmt="%.1f")

            add_map_features(ax)

            # Node ID — bold, just outside upper-left
            ax.text(
                0.0, 1.01,
                node_label(i, j),
                transform=ax.transAxes,
                fontsize=5.5,
                fontweight="bold",
                ha="left",
                va="bottom",
            )

            # FF risk stats — just outside upper-right
            ax.text(
                1.0, 1.01,
                _ff_label(counts, totals, risk, i, j),
                transform=ax.transAxes,
                fontsize=5.0,
                ha="right",
                va="bottom",
            )

    # ── Shared colorbar ───────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02, aspect=30)
    cbar.set_label(
        r"Standardized 850-hPa $\theta_e$ Anomaly",
        fontsize=6,
    )
    cbar.ax.tick_params(labelsize=5)

    # ── Save ──────────────────────────────────────────────────────────────────
    base = os.path.join(OUT_DIR, "fig01_som_node_weights")
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.png", dpi=DPI_RASTER)
    fig.savefig(f"{base}.tiff", dpi=DPI_RASTER)
    print(f"Saved to {OUT_DIR}/")
    plt.close()


if __name__ == "__main__":
    main()
