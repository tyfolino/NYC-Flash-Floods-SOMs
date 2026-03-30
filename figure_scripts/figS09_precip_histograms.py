"""
Supplementary Figure S9 — Per-Node Precipitation Histograms (Stage IV + ASOS)

Default (4-row × 2-column) layout, matching the row order of Figure 2.
Left column  : Stage IV max hourly precip over NYC within ±6 h of episode onset.
Right column : ASOS max hourly precip (JFK, LGA, Central Park, EWR) within ±6 h.
Red dashed line marks the per-node median.

Wide layout (--wide flag): 2-row × 4-column.
Left 2×2  : Stage IV (A1/A2 top, B1/B2 bottom)
Right 2×2 : ASOS (A1/A2 top, B1/B2 bottom)

Usage:
    python -m figure_scripts.figS09_precip_histograms
    python -m figure_scripts.figS09_precip_histograms --wide
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from som_analysis.config import DATA_DIR, setup_plotting
from som_analysis.helpers import node_label
from som_analysis.node_statistics import (
    compute_asos_max_precip,
    compute_stageiv_max_precip,
    load_asos_precip,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures", "figS09")
os.makedirs(OUT_DIR, exist_ok=True)

BMU_CSV = os.path.join(DATA_DIR, "som_2x2_evsom_24h_bmus_thetae.csv")

# ── Figure parameters ──────────────────────────────────────────────────────────
XDIM, YDIM = 2, 2
FIG_WIDTH = 7.0
FIG_HEIGHT = 7.0
FIG_WIDTH_WIDE = 12.0
FIG_HEIGHT_WIDE = 3.5
DPI_RASTER = 300

BINS = np.arange(0, 3.76, 0.25)
XLIM = (0, 3.75)
YLIM_MAX = 15

NODE_ORDER = [(i, j) for j in range(YDIM) for i in range(XDIM)]

COL_COLORS = {"Stage IV": "#3a7ebf", "ASOS": "#e07b3a"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--wide",
        action="store_true",
        help="Use wide 2×4 layout for presentations",
    )
    return p.parse_args()


def _plot_node_hist(ax, data, color, label):
    """Plot histogram with median line for one node/source."""
    valid = pd.to_numeric(data, errors="coerce").dropna()
    valid = valid[valid > 0]
    n = len(valid)

    ax.hist(
        valid,
        bins=BINS,
        color=color,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.5,
    )
    if n > 0:
        median = valid.median()
        ax.axvline(
            median,
            color="red",
            linestyle="--",
            linewidth=1.0,
            label=f'Median: {median:.2f}"',
        )
        ax.legend(fontsize=4, loc="upper right", framealpha=0.7)

    ax.set_xlim(*XLIM)
    ax.set_ylim(0, YLIM_MAX)
    ax.set_yticks(np.arange(0, YLIM_MAX + 1, 3))
    ax.tick_params(axis="both", labelsize=5)
    ax.grid(True, linewidth=0.3, alpha=0.5, axis="y")
    return n


def main():
    args = parse_args()
    setup_plotting()

    # ── Load BMUs ─────────────────────────────────────────────────────────────
    bmu_df = pd.read_csv(BMU_CSV, parse_dates=["timestamp"])
    # ASOS uses timestamp_local; timestamps are already UTC-converted EST so
    # using the same column is consistent with ±6 h window logic
    bmu_df["timestamp_local"] = bmu_df["timestamp"]

    # ── Compute precip values ─────────────────────────────────────────────────
    print("Computing Stage IV max precip ...")
    s4_vals, *_ = compute_stageiv_max_precip(bmu_df, window_hours=6)
    bmu_df["s4_max"] = s4_vals

    print("Loading ASOS data ...")
    precip_dfs = load_asos_precip()
    print("Computing ASOS max precip ...")
    asos_vals = compute_asos_max_precip(bmu_df, precip_dfs, window_hours=6)
    bmu_df["asos_max"] = asos_vals

    # ── Build figure ──────────────────────────────────────────────────────────
    if args.wide:
        nrows, ncols = 2, 4
        fig_w, fig_h = FIG_WIDTH_WIDE, FIG_HEIGHT_WIDE
    else:
        nrows, ncols = XDIM * YDIM, 2
        fig_w, fig_h = FIG_WIDTH, FIG_HEIGHT

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        constrained_layout=True,
        dpi=DPI_RASTER,
        sharex=True,
        sharey=True,
    )
    if args.wide:
        for ax in axes.ravel():
            ax.tick_params(axis="y", labelleft=True)

    for i, j in NODE_ORDER:
        if args.wide:
            ax_s4 = axes[j, i]
            ax_as = axes[j, i + 2]
        else:
            row = j * XDIM + i
            ax_s4 = axes[row, 0]
            ax_as = axes[row, 1]

        lbl = node_label(i, j)
        subset = bmu_df[(bmu_df["node_i"] == i) & (bmu_df["node_j"] == j)]

        n_s4 = _plot_node_hist(
            ax_s4, subset["s4_max"], COL_COLORS["Stage IV"], "Stage IV"
        )
        n_as = _plot_node_hist(ax_as, subset["asos_max"], COL_COLORS["ASOS"], "ASOS")

        # Node label + n inside upper-left of each panel
        ax_s4.text(
            0.03,
            0.97,
            f"{lbl}  ($n$={n_s4})",
            transform=ax_s4.transAxes,
            fontsize=6,
            fontweight="bold",
            ha="left",
            va="top",
        )
        ax_as.text(
            0.03,
            0.97,
            f"{lbl}  ($n$={n_as})",
            transform=ax_as.transAxes,
            fontsize=6,
            fontweight="bold",
            ha="left",
            va="top",
        )

        # y-axis label on left-most column of each group
        ax_s4.set_ylabel("Count", fontsize=5.5)
        if args.wide:
            ax_as.set_ylabel("Count", fontsize=5.5)

    # Column headers on top row
    if args.wide:
        axes[0, 0].set_title("Stage IV", fontsize=7, fontweight="bold")
        axes[0, 2].set_title("ASOS", fontsize=7, fontweight="bold")
    else:
        axes[0, 0].set_title("Stage IV", fontsize=7, fontweight="bold")
        axes[0, 1].set_title("ASOS", fontsize=7, fontweight="bold")

    # x-axis label on bottom row only
    for ax in axes[-1, :]:
        ax.set_xlabel("Max Hourly Precip (in)", fontsize=5.5)

    # ── Save ─────────────────────────────────────────────────────────────────
    suffix = "_wide" if args.wide else ""
    base = os.path.join(OUT_DIR, f"figS09_precip_histograms{suffix}")
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.png", dpi=DPI_RASTER)
    fig.savefig(f"{base}.tiff", dpi=DPI_RASTER)
    print(f"Saved to {OUT_DIR}/")
    plt.close()


if __name__ == "__main__":
    main()
