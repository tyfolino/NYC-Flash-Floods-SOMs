"""
Supplementary Figure S6 — FFE SOM Monthly Node Frequency

2×2 grid (A1, A2, B1, B2). Each panel shows the fraction of FFEs assigned to
that node within each calendar month (May–October), normalised by the total
number of FFEs in that month across all nodes.

Usage:
    python -m figure_scripts.figS06_ffe_monthly_histograms
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from som_analysis.config import DATA_DIR, setup_plotting
from som_analysis.helpers import node_label

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures", "figS06")
os.makedirs(OUT_DIR, exist_ok=True)

BMU_CSV = os.path.join(DATA_DIR, "som_2x2_bmus_thetae.csv")

# ── Figure parameters ──────────────────────────────────────────────────────────
XDIM, YDIM = 2, 2
FIG_WIDTH  = 5.5
FIG_HEIGHT = 4.0
DPI_RASTER = 300

MONTHS      = range(5, 11)        # May–October
MONTH_LABELS = ["May", "Jun", "Jul", "Aug", "Sep", "Oct"]
BAR_COLOR   = "#3a7ebf"

# Node order: A1, A2 (top row), B1, B2 (bottom row)
NODE_ORDER = [(i, j) for j in range(YDIM) for i in range(XDIM)]



def main():
    setup_plotting()

    # ── Load BMU assignments ──────────────────────────────────────────────────
    df = pd.read_csv(BMU_CSV, parse_dates=["timestamp"])
    df["month"] = df["timestamp"].dt.month

    # Total FFEs per month (across all nodes)
    monthly_totals = df.groupby("month").size().reindex(MONTHS, fill_value=0)

    # ── Build figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        YDIM, XDIM,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        constrained_layout=True,
        dpi=DPI_RASTER,
        sharex=True, sharey=True,
    )

    for idx, (i, j) in enumerate(NODE_ORDER):
        ax  = axes[j, i]
        lbl = node_label(i, j)

        # FFEs in this node per month, normalised by monthly total
        node_mask    = (df["node_i"] == i) & (df["node_j"] == j)
        node_monthly = df[node_mask].groupby("month").size().reindex(MONTHS, fill_value=0)
        fractions    = node_monthly / monthly_totals.replace(0, np.nan)

        ax.bar(np.arange(len(MONTHS)), fractions.values, width=0.7,
               color=BAR_COLOR, linewidth=0)
        ax.set_xticks(np.arange(len(MONTHS)))
        ax.set_xticklabels(MONTH_LABELS, fontsize=5.5)
        ax.tick_params(axis="y", labelsize=5.5)
        ax.set_ylim(0, 1)

        # Node label bold upper-left
        ax.text(0.03, 0.97, lbl, transform=ax.transAxes,
                fontsize=6.5, fontweight="bold", ha="left", va="top")

    # Shared axis labels
    for ax in axes[:, 0]:
        ax.set_ylabel("Fraction of monthly FFEs", fontsize=6)

    # ── Save ─────────────────────────────────────────────────────────────────
    base = os.path.join(OUT_DIR, "figS06_ffe_monthly_histograms")
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.png", dpi=DPI_RASTER)
    fig.savefig(f"{base}.tiff", dpi=DPI_RASTER)
    print(f"Saved to {OUT_DIR}/")
    plt.close()


if __name__ == "__main__":
    main()
