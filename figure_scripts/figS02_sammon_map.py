"""
Supplementary Figure S2 — Sammon / MDS Map of All-Days SOM

Single-panel figure showing the Sammon projection of the 5×4 all-days SOM.
Nodes are colored by the U-matrix (mean distance to neighbors) and sized by
hit frequency (number of days assigned). Topology-preserving neighbor edges
are drawn as faint gray lines.

Usage:
    python -m figure_scripts.figS02_sammon_map
"""

import os

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from som_analysis.config import setup_plotting
from som_analysis.helpers import node_label

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures", "figS02")
os.makedirs(OUT_DIR, exist_ok=True)

CACHE_PATH = (
    "/home/janoski/nyc_flash_flood/figs/"
    "Z500-and-thetae-alldays-SOM/.cache/som_results.npz"
)

# ── Figure parameters ──────────────────────────────────────────────────────────
XDIM, YDIM = 5, 4
FIG_WIDTH = 3.5
FIG_HEIGHT = 3.2
DPI_RASTER = 300

# Marker size range (points²)
S_MIN, S_MAX = 40, 280


def main():
    setup_plotting()

    # ── Load cache ────────────────────────────────────────────────────────────
    print(f"Loading cache from {CACHE_PATH} ...")
    cached = np.load(CACHE_PATH)
    u_matrix = cached["u_matrix"]  # (xdim, ydim)
    hit_map = cached["hit_map"]  # (xdim, ydim)
    coords = cached["coords"]  # (n_nodes, 2) Sammon coords

    # Flatten in same order as node = i*YDIM+j
    U_flat = u_matrix.T.reshape(-1)
    hits_flat = hit_map.T.reshape(-1)
    hits_norm = hits_flat / hits_flat.max()
    sizes = S_MIN + (S_MAX - S_MIN) * hits_norm

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI_RASTER, constrained_layout=True
    )

    # Neighbor edges
    for i in range(XDIM):
        for j in range(YDIM):
            node = i * YDIM + j
            for di, dj in [(0, 1), (1, 0)]:
                ni, nj = i + di, j + dj
                if ni < XDIM and nj < YDIM:
                    nbr = ni * YDIM + nj
                    ax.plot(
                        [coords[node, 0], coords[nbr, 0]],
                        [coords[node, 1], coords[nbr, 1]],
                        color="0.70",
                        lw=0.8,
                        zorder=1,
                    )

    # Scatter
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=U_flat,
        s=sizes,
        cmap="RdYlBu_r",
        edgecolors="k",
        linewidths=0.4,
        zorder=3,
    )

    # Node labels centered on each dot (white outline for legibility)
    for idx, (x, y) in enumerate(coords):
        ix, iy = divmod(idx, YDIM)
        ax.text(
            x,
            y,
            node_label(ix, iy),
            fontsize=4.5,
            ha="center",
            va="center",
            zorder=5,
            path_effects=[
                pe.withStroke(linewidth=1.5, foreground="white"),
            ],
        )

    ax.axis("off")

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, aspect=22)
    cbar.set_label("U-matrix (mean neighbor distance)", fontsize=5.5)
    cbar.ax.tick_params(labelsize=4.5)

    # Size legend — standard round hit counts, same sizing formula as scatter
    hit_vals = [150, 300, 450]
    legend_handles = []
    for hv in hit_vals:
        s = S_MIN + (S_MAX - S_MIN) * (hv / hits_flat.max())
        h = ax.scatter(
            [], [], s=s, c="0.55", edgecolors="k", linewidths=0.4, label=f"{hv:d} days"
        )
        legend_handles.append(h)
    ax.legend(
        handles=legend_handles,
        title="Hit frequency",
        title_fontsize=5,
        fontsize=4.5,
        loc="upper left",
        framealpha=0.8,
        handletextpad=0.8,
        borderpad=1.8,
        labelspacing=2.8,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    base = os.path.join(OUT_DIR, "figS02_sammon_map")
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.png", dpi=DPI_RASTER)
    fig.savefig(f"{base}.tiff", dpi=DPI_RASTER)
    print(f"Saved to {OUT_DIR}/")
    plt.close()


if __name__ == "__main__":
    main()
