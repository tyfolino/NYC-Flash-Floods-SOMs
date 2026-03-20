"""
Figure 4 — Evolution SOM Node Weights at Key Hours

4-row × 4-column layout.
Rows  : SOM nodes A1, A2, B1, B2.
Cols  : T−18h, T−12h, T−6h, T=0 (6-hourly, equally spaced within 24h window).
Style : standardized Z500 (contoured) + 850-hPa θe (shaded) — same as Fig. 1.

Node labels appear outside upper-left of the leftmost panel in each row.
Time labels appear above each panel in the top row.
Single shared colorbar on the right.

Usage:
    python -m figure_scripts.fig04_evsom_key_hours
"""

import os

import cartopy.crs as ccrs
import cmweather  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from som_analysis.config import setup_plotting
from som_analysis.helpers import add_map_features, node_label

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures", "fig04")
os.makedirs(OUT_DIR, exist_ok=True)

CACHE_PATH = (
    "/home/janoski/nyc_flash_flood/figs/"
    "Z500-and-thetae-evSOM-24h/.cache/som_results.npz"
)

# ── Figure / SOM parameters ───────────────────────────────────────────────────
XDIM, YDIM = 2, 2
FIG_WIDTH  = 7.0
FIG_HEIGHT = 4.2
DPI_RASTER = 300

# Key hours (T−18 to T=0, 6-hourly) and their indices in the 24h window
KEY_HOURS   = [-18, -12, -6, 0]
KEY_INDICES = [5, 11, 17, 23]   # index into hour_offset axis (T-23=0 … T=0=23)
KEY_LABELS  = [r"T$-$18h", r"T$-$12h", r"T$-$6h", r"T$=$0"]

# Contour levels — identical to Fig. 1
LEVELS_MOIST = np.arange(-1.2, 1.21, 0.2)
LEVELS_Z     = np.arange(-1.4, 1.41, 0.2)

# Node traversal order: A1, A2, B1, B2
NODE_ORDER = [(i, j) for j in range(YDIM) for i in range(XDIM)]


def main():
    setup_plotting()

    # ── Load cached evSOM results ─────────────────────────────────────────────
    print(f"Loading cache from {CACHE_PATH} ...")
    cached      = np.load(CACHE_PATH)
    z500_nodes  = cached["z500_nodes"]   # (xdim, ydim, n_hours, nlat, nlon)
    moist_nodes = cached["moist_nodes"]
    bmus        = cached["bmus"]         # (121, 2)
    lat         = cached["lat_z"]
    lon         = cached["lon_z"]

    # ── Build figure: 4 rows (nodes) × 4 cols (hours) ────────────────────────
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        4, 4,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        subplot_kw={"projection": proj},
        constrained_layout=True,
        dpi=DPI_RASTER,
    )
    fig.get_layout_engine().set(hspace=0.0, wspace=0.02)

    im = None
    for row, (i, j) in enumerate(NODE_ORDER):
        lbl = node_label(i, j)
        # Count events in this node
        n = int(np.sum((bmus[:, 0] == i) & (bmus[:, 1] == j)))

        for col, (hr_idx, hr_label) in enumerate(zip(KEY_INDICES, KEY_LABELS)):
            ax = axes[row, col]

            # θe shaded
            im = ax.contourf(
                lon, lat, moist_nodes[i, j, hr_idx],
                cmap="balance",
                levels=LEVELS_MOIST,
                transform=proj,
                extend="both",
            )

            # Z500 contoured
            cn = ax.contour(
                lon, lat, z500_nodes[i, j, hr_idx],
                colors="black",
                linewidths=0.4,
                levels=LEVELS_Z,
                transform=proj,
            )
            ax.clabel(cn, inline=True, fontsize=3.0, fmt="%.1f")
            add_map_features(ax)

            # Node label: outside upper-left of leftmost column only
            if col == 0:
                ax.text(
                    0.0, 1.01, f"{lbl}  ($n$={n})",
                    transform=ax.transAxes,
                    fontsize=5.5, fontweight="bold",
                    ha="left", va="bottom",
                )

            # Time label: above top row only
            if row == 0:
                ax.text(
                    0.5, 1.01, hr_label,
                    transform=ax.transAxes,
                    fontsize=5.5,
                    ha="center", va="bottom",
                )

    # ── Shared colorbar ───────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02, aspect=30)
    cbar.set_label(r"Standardized 850-hPa $\theta_e$ Anomaly", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    # ── Save ─────────────────────────────────────────────────────────────────
    base = os.path.join(OUT_DIR, "fig04_evsom_key_hours")
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.png", dpi=DPI_RASTER)
    fig.savefig(f"{base}.tiff", dpi=DPI_RASTER)
    print(f"Saved to {OUT_DIR}/")
    plt.close()


if __name__ == "__main__":
    main()
