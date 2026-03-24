"""
Supplementary Figure S6 — evSOM Moisture Composites: IVT (left) and TCWV (right)

4-row × 2-column layout. Each row is one evSOM node (A1, A2, B1, B2).
Left column  : IVT composite mean (shaded, kg m⁻¹ s⁻¹) + Z500 (contoured, dam)
Right column : TCWV composite mean (shaded, kg m⁻²) + Z500 (contoured, dam)

FFE events are grouped by evSOM node (T=0 snapshot).
Separate shared colorbars for each column.

Usage:
    python -m figure_scripts.figS07_evsom_ivt_tcwv_composites
"""

import os

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from som_analysis.config import SOM_INTERMEDIATE_PATH, setup_plotting
from som_analysis.helpers import add_map_features, get_node_indices, node_label

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures", "figS07")
os.makedirs(OUT_DIR, exist_ok=True)

CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "figs",
    "Z500-and-thetae-evSOM-24h",
    ".cache",
    "som_results.npz",
)
IVT_PATH = f"{SOM_INTERMEDIATE_PATH}era5_ivt_ffe.nc"
TCWV_PATH = f"{SOM_INTERMEDIATE_PATH}era5_tcwv_ffe.nc"
Z500_PATH = f"{SOM_INTERMEDIATE_PATH}era5_Z500_ffe.nc"

# ── Figure / SOM parameters ───────────────────────────────────────────────────
XDIM, YDIM = 2, 2
FIG_WIDTH = 6.0
FIG_HEIGHT = 5.8
DPI_RASTER = 300

LEVELS_IVT = np.arange(0, 701, 100)  # kg m⁻¹ s⁻¹
LEVELS_TCWV = np.arange(20, 56, 5)  # kg m⁻²
LEVELS_Z = range(549, 598, 2)  # dam

# Node traversal order: A1, A2, B1, B2
NODE_ORDER = [(i, j) for j in range(YDIM) for i in range(XDIM)]


def main():
    setup_plotting()

    # ── Load cache ────────────────────────────────────────────────────────────
    print(f"Loading cache from {CACHE_PATH} ...")
    cached = np.load(CACHE_PATH)
    bmus = cached["bmus"]  # (118, 2)

    # ── Load raw FFE fields ───────────────────────────────────────────────────
    print("Loading raw FFE fields ...")
    ivt_da = xr.open_dataset(IVT_PATH)["ivt"]
    tcwv_da = xr.open_dataset(TCWV_PATH)["tcwv"]
    z500_da = xr.open_dataset(Z500_PATH)["z"]

    lat = xr.open_dataset(IVT_PATH)["latitude"].values
    lon = xr.open_dataset(IVT_PATH)["longitude"].values

    # ── Build figure ──────────────────────────────────────────────────────────
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        4,
        2,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        subplot_kw={"projection": proj},
        constrained_layout=True,
        dpi=DPI_RASTER,
    )
    fig.get_layout_engine().set(hspace=0.0, wspace=0.02)

    im_ivt = im_tcwv = None

    for row, (i, j) in enumerate(NODE_ORDER):
        ax_ivt = axes[row, 0]
        ax_tcwv = axes[row, 1]
        lbl = node_label(i, j)
        idx = get_node_indices(bmus, i, j)
        n = len(idx)

        # Composite means for this node
        ivt_comp = ivt_da.isel(valid_time=idx).mean("valid_time").values
        tcwv_comp = tcwv_da.isel(valid_time=idx).mean("valid_time").values
        z500_comp = z500_da.isel(valid_time=idx).mean("valid_time").values / 98.1

        # ── Left: IVT ────────────────────────────────────────────────────────
        im_ivt = ax_ivt.contourf(
            lon,
            lat,
            ivt_comp,
            cmap="YlOrRd",
            levels=LEVELS_IVT,
            transform=proj,
            extend="max",
        )
        cn = ax_ivt.contour(
            lon,
            lat,
            z500_comp,
            colors="black",
            linewidths=0.4,
            levels=LEVELS_Z,
            transform=proj,
        )
        ax_ivt.clabel(cn, inline=True, fontsize=3.5, fmt="%d")
        add_map_features(ax_ivt)

        ax_ivt.text(
            0.0,
            1.01,
            lbl,
            transform=ax_ivt.transAxes,
            fontsize=5.5,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        ax_ivt.text(
            1.0,
            1.01,
            f"$n$={n}",
            transform=ax_ivt.transAxes,
            fontsize=6.0,
            ha="right",
            va="bottom",
        )

        # ── Right: TCWV ──────────────────────────────────────────────────────
        im_tcwv = ax_tcwv.contourf(
            lon,
            lat,
            tcwv_comp,
            cmap="Blues",
            levels=LEVELS_TCWV,
            transform=proj,
            extend="both",
        )
        cn2 = ax_tcwv.contour(
            lon,
            lat,
            z500_comp,
            colors="black",
            linewidths=0.4,
            levels=LEVELS_Z,
            transform=proj,
        )
        ax_tcwv.clabel(cn2, inline=True, fontsize=3.5, fmt="%d")
        add_map_features(ax_tcwv)

        ax_tcwv.text(
            0.0,
            1.01,
            lbl,
            transform=ax_tcwv.transAxes,
            fontsize=5.5,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        ax_tcwv.text(
            1.0,
            1.01,
            f"$n$={n}",
            transform=ax_tcwv.transAxes,
            fontsize=6.0,
            ha="right",
            va="bottom",
        )

    # ── Colorbars ─────────────────────────────────────────────────────────────
    cbar_ivt = fig.colorbar(
        im_ivt,
        ax=axes[:, 0].tolist(),
        orientation="vertical",
        pad=0.02,
        shrink=0.7,
        aspect=25,
    )
    cbar_ivt.set_label(r"$|\mathrm{IVT}|$ (kg m$^{-1}$ s$^{-1}$)", fontsize=6)
    cbar_ivt.ax.tick_params(labelsize=5)

    cbar_tcwv = fig.colorbar(
        im_tcwv,
        ax=axes[:, 1].tolist(),
        orientation="vertical",
        pad=0.02,
        shrink=0.7,
        aspect=25,
    )
    cbar_tcwv.set_label(r"TCWV (kg m$^{-2}$)", fontsize=6)
    cbar_tcwv.ax.tick_params(labelsize=5)

    # ── Save ─────────────────────────────────────────────────────────────────
    base = os.path.join(OUT_DIR, "figS07_evsom_ivt_tcwv_composites")
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.png", dpi=DPI_RASTER)
    fig.savefig(f"{base}.tiff", dpi=DPI_RASTER)
    print(f"Saved to {OUT_DIR}/")
    plt.close()


if __name__ == "__main__":
    main()
