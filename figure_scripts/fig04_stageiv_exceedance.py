"""
Figure 4 (alternate) — Stage IV Conditional Mean Precipitation by SOM Node

For each evSOM node, shows the mean of the maximum hourly Stage IV accumulation
within ±6 hours of flash flood onset, conditioned on events that produced
>= MIN_PRECIP_IN at each grid point, and only shown where >= MIN_EVENTS events
met that threshold. This suppresses the influence of single extreme outliers.

Default (4-row × 2-column) layout. Each row is one SOM node (A1, A2, B1, B2).
Left column  : regional-scale composite (38.5–44°N, 78–70°W)
Right column : NYC-scale composite (40.3–41.2°N, 74.8–73.4°W)

Wide layout (--wide flag): 2-row × 4-column.

Usage:
    python -m figure_scripts.fig04_stageiv_exceedance
    python -m figure_scripts.fig04_stageiv_exceedance --wide
"""

import argparse
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
    compute_stageiv_conditional_mean,
    compute_stageiv_node_composites,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures", "fig04")
os.makedirs(OUT_DIR, exist_ok=True)

BMU_CSV = os.path.join(DATA_DIR, "som_2x2_evsom_24h_bmus_thetae.csv")

# ── Figure / SOM parameters ───────────────────────────────────────────────────
XDIM, YDIM = 2, 2
FIG_WIDTH = 5.5
FIG_HEIGHT = 6.0
FIG_WIDTH_WIDE = 11.0
FIG_HEIGHT_WIDE = 3.5
DPI_RASTER = 300

EXTENT_REG = [-78.0, -70.0, 38.5, 44.0]
EXTENT_NYC = [-74.8, -73.4, 40.3, 41.2]

MIN_PRECIP_IN = 0.10  # minimum precip to count an event as producing rain
MIN_EVENTS = 3  # minimum number of such events to show a grid point

# NWS ChaseSpectral colormap (same as PMM figure)
NWS_LEVELS = [0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00]
_N_BINS = len(NWS_LEVELS) - 1
_chase = plt.get_cmap("ChaseSpectral")
_colors = [_chase(i / (_N_BINS - 1)) for i in range(_N_BINS)]
CMAP_NWS = ListedColormap(_colors, name="ChaseSpectral_disc")
CMAP_NWS.set_over("white")
NORM_NWS = BoundaryNorm(NWS_LEVELS, ncolors=_N_BINS, clip=False)

# Node order: A1, A2, B1, B2
NODE_ORDER = [(i, j) for j in range(YDIM) for i in range(XDIM)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--wide",
        action="store_true",
        help="Use wide 2×4 layout for presentations",
    )
    return p.parse_args()


def _add_map_features(ax, scale):
    ax.add_feature(cfeature.LAND.with_scale(scale), facecolor="#ebebeb", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale(scale), facecolor="#dde8f2", zorder=0)
    ax.add_feature(cfeature.STATES.with_scale(scale), linewidth=0.3, zorder=4)
    ax.add_feature(cfeature.COASTLINE.with_scale(scale), linewidth=0.4, zorder=4)


def _plot_condmean(ax, cond_mean, node_fields, i, j, lat2d, lon2d, extent):
    """Plot conditional mean precipitation for node (i, j) on ax."""
    field = cond_mean[(i, j)]
    fields = node_fields[(i, j)]
    n = int(np.sum(~np.all(np.isnan(fields), axis=(1, 2))))

    field_plot = np.where(field >= NWS_LEVELS[0], field, np.nan)

    lon_span = extent[1] - extent[0]
    scale = "10m" if lon_span < 4 else "50m"

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    _add_map_features(ax, scale)
    ax.pcolormesh(
        lon2d,
        lat2d,
        field_plot,
        cmap=CMAP_NWS,
        norm=NORM_NWS,
        shading="auto",
        zorder=2,
        transform=ccrs.PlateCarree(),
    )
    return n


def main():
    args = parse_args()
    setup_plotting()

    # ── Load BMU assignments ──────────────────────────────────────────────────
    print(f"Loading BMU assignments from {BMU_CSV} ...")
    bmu_df = pd.read_csv(BMU_CSV)
    bmu_df["timestamp"] = pd.to_datetime(bmu_df["timestamp"])

    # ── Compute Stage IV composites and conditional mean ──────────────────────
    node_fields, lat2d, lon2d = compute_stageiv_node_composites(bmu_df, XDIM, YDIM)
    cond_mean = compute_stageiv_conditional_mean(
        node_fields, min_precip_in=MIN_PRECIP_IN, min_events=MIN_EVENTS
    )

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

    for i, j in NODE_ORDER:
        if args.wide:
            ax_reg = axes[j, i]
            ax_nyc = axes[j, i + 2]
        else:
            row = j * XDIM + i
            ax_reg = axes[row, 0]
            ax_nyc = axes[row, 1]

        lbl = node_label(i, j)

        n_reg = _plot_condmean(
            ax_reg, cond_mean, node_fields, i, j, lat2d, lon2d, EXTENT_REG
        )
        n_nyc = _plot_condmean(
            ax_nyc, cond_mean, node_fields, i, j, lat2d, lon2d, EXTENT_NYC
        )

        for ax, n in [(ax_reg, n_reg), (ax_nyc, n_nyc)]:
            ax.text(
                0.0,
                1.01,
                lbl,
                transform=ax.transAxes,
                fontsize=5.5,
                fontweight="bold",
                ha="left",
                va="bottom",
            )
            ax.text(
                1.0,
                1.01,
                f"$n$={n}",
                transform=ax.transAxes,
                fontsize=6.0,
                ha="right",
                va="bottom",
            )

    # ── Shared colorbar ───────────────────────────────────────────────────────
    sm = ScalarMappable(cmap=CMAP_NWS, norm=NORM_NWS)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        orientation="vertical",
        pad=0.02,
        shrink=0.6,
        extend="max",
        aspect=25,
    )
    cbar.set_label("Conditional Mean Precipitation (in)", fontsize=6)
    cbar.set_ticks(NWS_LEVELS)
    cbar.ax.tick_params(labelsize=5)

    # ── Save ──────────────────────────────────────────────────────────────────
    suffix = "_wide" if args.wide else ""
    base = os.path.join(OUT_DIR, f"fig04_stageiv_condmean{suffix}")
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.png", dpi=DPI_RASTER)
    fig.savefig(f"{base}.tiff", dpi=DPI_RASTER)
    print(f"Saved to {OUT_DIR}/")
    plt.close()


if __name__ == "__main__":
    main()
