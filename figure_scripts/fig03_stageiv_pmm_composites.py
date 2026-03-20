"""
Figure 3 — Stage IV PMM Precipitation Composites by SOM Node

4-row × 2-column layout. Each row is one SOM node (A1, A2, B1, B2).
Left column  : regional-scale PMM composite (38.5–44°N, 78–70°W)
Right column : NYC-scale PMM composite (40.3–41.2°N, 74.8–73.4°W)

Single shared NWS-style colorbar on the right.
Star marks NYC on regional panels only.

Usage:
    python -m figure_scripts.fig03_stageiv_pmm_composites
"""

import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmweather  # noqa: F401 — registers ChaseSpectral and other cmweather colormaps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap

from som_analysis.config import DATA_DIR, setup_plotting
from som_analysis.helpers import node_label
from som_analysis.node_statistics import (
    compute_stageiv_node_composites,
    probability_matched_mean,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures", "fig03")
os.makedirs(OUT_DIR, exist_ok=True)

BMU_CSV = os.path.join(DATA_DIR, "som_2x2_bmus_thetae.csv")

# ── Figure / SOM parameters ───────────────────────────────────────────────────
XDIM, YDIM = 2, 2
FIG_WIDTH  = 5.5
FIG_HEIGHT = 6.0
DPI_RASTER = 300

EXTENT_REG = [-78.0, -70.0, 38.5, 44.0]
EXTENT_NYC = [-74.8, -73.4, 40.3, 41.2]

# ChaseSpectral precipitation colormap sampled across our bins
# "over" color is black — clearly distinct from the 3–4" bin
NWS_LEVELS = [0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00]
_N_BINS = len(NWS_LEVELS) - 1  # 10
_chase = plt.get_cmap("ChaseSpectral")
_colors = [_chase(i / (_N_BINS - 1)) for i in range(_N_BINS)]
CMAP_NWS = ListedColormap(_colors, name="ChaseSpectral_disc")
CMAP_NWS.set_over("white")
NORM_NWS = BoundaryNorm(NWS_LEVELS, ncolors=_N_BINS, clip=False)

# Node order: A1, A2, B1, B2
NODE_ORDER = [(i, j) for j in range(YDIM) for i in range(XDIM)]


def _add_map_features(ax, scale):
    ax.add_feature(cfeature.LAND.with_scale(scale), facecolor="#ebebeb", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale(scale), facecolor="#dde8f2", zorder=0)
    ax.add_feature(cfeature.STATES.with_scale(scale), linewidth=0.3, zorder=4)
    ax.add_feature(cfeature.COASTLINE.with_scale(scale), linewidth=0.4, zorder=4)


def _plot_pmm(ax, node_fields, i, j, lat2d, lon2d, extent, show_star):
    """Compute and plot PMM for node (i, j) on ax."""
    fields = node_fields[(i, j)]
    n = int(np.sum(~np.all(np.isnan(fields), axis=(1, 2))))
    pmm = probability_matched_mean(fields)
    pmm_plot = np.where(pmm >= NWS_LEVELS[0], pmm, np.nan)

    lon_span = extent[1] - extent[0]
    scale = "10m" if lon_span < 4 else "50m"

    ax.set_extent(extent)
    _add_map_features(ax, scale)
    ax.pcolormesh(
        lon2d, lat2d, pmm_plot,
        cmap=CMAP_NWS, norm=NORM_NWS,
        shading="auto", zorder=2,
        transform=ccrs.PlateCarree(),
    )
    if show_star:
        ax.scatter(
            -74.0, 40.7,
            color="black", s=12, marker="*", zorder=5,
            transform=ccrs.PlateCarree(),
        )
    return n


def main():
    setup_plotting()

    # ── Load BMU assignments ──────────────────────────────────────────────────
    print(f"Loading BMU assignments from {BMU_CSV} ...")
    bmu_df = pd.read_csv(BMU_CSV)
    bmu_df["timestamp"] = pd.to_datetime(bmu_df["timestamp"])

    # ── Compute Stage IV composites ───────────────────────────────────────────
    node_fields, lat2d, lon2d = compute_stageiv_node_composites(bmu_df, XDIM, YDIM)

    # ── Build figure ──────────────────────────────────────────────────────────
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        4, 2,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        subplot_kw={"projection": proj},
        constrained_layout=True,
        dpi=DPI_RASTER,
    )
    fig.get_layout_engine().set(hspace=0.0, wspace=0.02)

    for row, (i, j) in enumerate(NODE_ORDER):
        ax_reg = axes[row, 0]
        ax_nyc = axes[row, 1]
        lbl = node_label(i, j)

        n_reg = _plot_pmm(ax_reg, node_fields, i, j, lat2d, lon2d, EXTENT_REG,
                          show_star=False)
        n_nyc = _plot_pmm(ax_nyc, node_fields, i, j, lat2d, lon2d, EXTENT_NYC,
                          show_star=False)

        # Node label outside upper-left; N outside upper-right
        for ax, n in [(ax_reg, n_reg), (ax_nyc, n_nyc)]:
            ax.text(0.0, 1.01, lbl, transform=ax.transAxes,
                    fontsize=5.5, fontweight="bold", ha="left", va="bottom")
            ax.text(1.0, 1.01, f"$n$={n}", transform=ax.transAxes,
                    fontsize=6.0, ha="right", va="bottom")

    # ── Shared colorbar ───────────────────────────────────────────────────────
    sm = ScalarMappable(cmap=CMAP_NWS, norm=NORM_NWS)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axes.ravel().tolist(),
        orientation="vertical", pad=0.02, shrink=0.6,
        extend="max", aspect=25,
    )
    cbar.set_label("PMM Precipitation (in)", fontsize=6)
    cbar.set_ticks(NWS_LEVELS)
    cbar.ax.tick_params(labelsize=5)

    # ── Save ──────────────────────────────────────────────────────────────────
    base = os.path.join(OUT_DIR, "fig03_stageiv_pmm")
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.png", dpi=DPI_RASTER)
    fig.savefig(f"{base}.tiff", dpi=DPI_RASTER)
    print(f"Saved to {OUT_DIR}/")
    plt.close()


if __name__ == "__main__":
    main()
