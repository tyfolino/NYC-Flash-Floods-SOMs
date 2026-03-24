"""
Figure 3 — evSOM Raw Mean Composites at Key Hours

4-row × 4-column layout.
Rows  : SOM nodes A1, A2, B1, B2.
Cols  : T−18h, T−12h, T−6h, T=0 (6-hourly, equally spaced within 24h window).
Style : mean Z500 in dam (contoured) + mean 850-hPa θe in K (shaded),
        composited over events assigned to each node.

Node labels appear outside upper-left of the leftmost panel in each row.
Time labels appear above each panel in the top row.
Single shared colorbar on the right.

Usage:
    python -m figure_scripts.fig03_evsom_composite_key_hours
"""

import os

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from som_analysis.config import setup_plotting
from som_analysis.helpers import add_map_features, node_label

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures", "fig03")
os.makedirs(OUT_DIR, exist_ok=True)

CACHE_PATH = (
    "/home/janoski/nyc_flash_flood/figs/"
    "Z500-and-thetae-evSOM-24h/.cache/som_results.npz"
)
Z500_PATH = "/mnt/drive2/SOM_intermediate_files/era5_Z500_ffe_evsom.nc"
THETAE_PATH = "/mnt/drive2/SOM_intermediate_files/era5_thetae_ffe_evsom.nc"

# ── Figure / SOM parameters ───────────────────────────────────────────────────
XDIM, YDIM = 2, 2
FIG_WIDTH = 7.0
FIG_HEIGHT = 4.2
DPI_RASTER = 300

# Key hours and their hour_offset coordinate values
KEY_OFFSETS = [-18, -12, -6, 0]
KEY_LABELS = [r"T$-$18h", r"T$-$12h", r"T$-$6h", r"T$=$0"]

# Contour levels for raw fields
LEVELS_THETAE = np.arange(295, 341, 5)  # K
LEVELS_Z = range(549, 598, 2)  # dam

# Node traversal order: A1, A2, B1, B2
NODE_ORDER = [(i, j) for j in range(YDIM) for i in range(XDIM)]


def main():
    setup_plotting()

    # ── Load cache for BMU assignments ────────────────────────────────────────
    print(f"Loading cache from {CACHE_PATH} ...")
    cached = np.load(CACHE_PATH)
    bmus = cached["bmus"]  # (121, 2)

    # ── Load raw evSOM fields ─────────────────────────────────────────────────
    print("Loading raw evSOM fields ...")
    z500_var = list(xr.open_dataset(Z500_PATH).data_vars)[0]
    thetae_var = list(xr.open_dataset(THETAE_PATH).data_vars)[0]
    z500_da = xr.open_dataset(Z500_PATH)[z500_var]  # (event, hour_offset, lat, lon)
    thetae_da = xr.open_dataset(THETAE_PATH)[thetae_var]

    lat = z500_da["latitude"].values
    lon = z500_da["longitude"].values

    # ── Build figure: 4 rows (nodes) × 4 cols (hours) ────────────────────────
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        4,
        4,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        subplot_kw={"projection": proj},
        constrained_layout=True,
        dpi=DPI_RASTER,
    )
    fig.get_layout_engine().set(hspace=0.0, wspace=0.02)

    im = None
    for row, (i, j) in enumerate(NODE_ORDER):
        lbl = node_label(i, j)
        idx = np.where((bmus[:, 0] == i) & (bmus[:, 1] == j))[0]
        n = len(idx)

        for col, (hr, hr_label) in enumerate(
            zip(KEY_OFFSETS, KEY_LABELS, strict=False)
        ):
            ax = axes[row, col]

            # Composite means for this node at this hour
            thetae_comp = (
                thetae_da.isel(event_time=idx)
                .sel(hour_offset=hr)
                .mean("event_time")
                .values
            )
            z500_comp = (
                z500_da.isel(event_time=idx)
                .sel(hour_offset=hr)
                .mean("event_time")
                .values
                / 98.1
            )

            # θe shaded
            im = ax.contourf(
                lon,
                lat,
                thetae_comp,
                cmap="BuPu",
                levels=LEVELS_THETAE,
                transform=proj,
                extend="both",
            )

            # Z500 contoured
            cn = ax.contour(
                lon,
                lat,
                z500_comp,
                colors="black",
                linewidths=0.4,
                levels=LEVELS_Z,
                transform=proj,
            )
            ax.clabel(cn, inline=True, fontsize=3.0, fmt="%d")
            add_map_features(ax)

            # Node label: outside upper-left of leftmost column only
            if col == 0:
                ax.text(
                    0.0,
                    1.01,
                    f"{lbl}  ($n$={n})",
                    transform=ax.transAxes,
                    fontsize=5.5,
                    fontweight="bold",
                    ha="left",
                    va="bottom",
                )

            # Time label: above top row only
            if row == 0:
                ax.text(
                    0.5,
                    1.01,
                    hr_label,
                    transform=ax.transAxes,
                    fontsize=5.5,
                    ha="center",
                    va="bottom",
                )

    # ── Shared colorbar ───────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02, aspect=30)
    cbar.set_label(r"850-hPa $\theta_e$ (K)", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    # ── Save ─────────────────────────────────────────────────────────────────
    base = os.path.join(OUT_DIR, "fig03_evsom_composites_key_hours")
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.png", dpi=DPI_RASTER)
    fig.savefig(f"{base}.tiff", dpi=DPI_RASTER)
    print(f"Saved to {OUT_DIR}/")
    plt.close()


if __name__ == "__main__":
    main()
