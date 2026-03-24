"""
Supplementary Figure S6 — evSOM to All-Days SOM Node Mapping

4-row × 1-column layout, one panel per evSOM node (A1, A2, B1, B2),
matching the row order of Figure 2.
Each panel shows a 5×4 heatmap of what fraction of the node's FFEs
fall into each all-days SOM node (thetae, 5×4).
Event counts are annotated inside each cell.

Usage:
    python -m figure_scripts.figS06_evsom_to_alldays_mapping
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

EVSOM_CSV = os.path.join(DATA_DIR, "som_2x2_evsom_24h_bmus_thetae.csv")
ALLDAYS_CSV = os.path.join(DATA_DIR, "som_5x4_alldays_ffe_bmus_thetae.csv")

# ── Figure / SOM parameters ───────────────────────────────────────────────────
XFFE, YFFE = 2, 2  # evSOM dimensions
XALL, YALL = 5, 4  # all-days SOM dimensions

FIG_WIDTH = 3.5
FIG_HEIGHT = 8.0
DPI_RASTER = 300

# Node traversal order: A1, A2, B1, B2
NODE_ORDER = [(i, j) for j in range(YFFE) for i in range(XFFE)]


def _load_and_merge(evsom_csv, alldays_csv):
    """Inner-join evSOM and all-days BMU CSVs on date."""
    ev = pd.read_csv(evsom_csv, parse_dates=["timestamp"])
    ad = pd.read_csv(alldays_csv, parse_dates=["timestamp"])

    ev["date"] = ev["timestamp"].dt.normalize()
    ad["date"] = ad["timestamp"].dt.normalize()

    ev = ev.rename(columns={"node_i": "node_ev_i", "node_j": "node_ev_j"})
    ad = ad.rename(columns={"node_i": "node_all_i", "node_j": "node_all_j"})

    return ev.merge(ad[["date", "node_all_i", "node_all_j"]], on="date", how="inner")


def main():
    setup_plotting()

    print("Loading BMU CSVs ...")
    merged = _load_and_merge(EVSOM_CSV, ALLDAYS_CSV)
    print(f"Matched {len(merged)} events")

    # ── Build figure: 4 rows × 1 col ─────────────────────────────────────────
    fig, axes = plt.subplots(
        XFFE * YFFE,
        1,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        constrained_layout=True,
        dpi=DPI_RASTER,
    )

    last_im = None
    for row, (i, j) in enumerate(NODE_ORDER):
        ax = axes[row]
        lbl = node_label(i, j)
        subset = merged[(merged["node_ev_i"] == i) & (merged["node_ev_j"] == j)]
        n_events = len(subset)

        # Count hits in each all-days node
        counts = np.zeros((XALL, YALL), dtype=int)
        for _, r in subset.iterrows():
            counts[int(r["node_all_i"]), int(r["node_all_j"])] += 1

        pcts = 100.0 * counts / n_events if n_events > 0 else counts.astype(float)

        # imshow: rows = all-days row (j), cols = all-days col (i)
        last_im = ax.imshow(
            pcts.T,
            cmap="YlOrRd",
            origin="upper",
            vmin=0,
            vmax=50,
            aspect="auto",
        )

        # Annotate each cell with count
        for bi in range(XALL):
            for bj in range(YALL):
                if counts[bi, bj] > 0:
                    text_color = "white" if pcts[bi, bj] > 28 else "black"
                    ax.text(
                        bi,
                        bj,
                        str(counts[bi, bj]),
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color=text_color,
                    )

        # Axis ticks — label all-days nodes
        ax.set_xticks(np.arange(XALL))
        ax.set_yticks(np.arange(YALL))
        ax.set_xticklabels([str(bi + 1) for bi in range(XALL)], fontsize=5)
        ax.set_yticklabels([node_label(0, bj)[0] for bj in range(YALL)], fontsize=5)
        ax.tick_params(length=2)

        # Axis labels only on outer edges
        if row == XFFE * YFFE - 1:
            ax.set_xlabel("All-days SOM column", fontsize=5.5)
        ax.set_ylabel("All-days SOM row", fontsize=5.5)

        ax.text(
            0.0,
            1.02,
            lbl,
            transform=ax.transAxes,
            fontsize=6.5,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        ax.text(
            1.0,
            1.02,
            f"$n$={n_events}",
            transform=ax.transAxes,
            fontsize=5.5,
            ha="right",
            va="bottom",
        )

    # ── Shared colorbar ───────────────────────────────────────────────────────
    cbar = fig.colorbar(
        last_im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02, aspect=25
    )
    cbar.set_label(r"Fraction of node events (\%)", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    # ── Save ─────────────────────────────────────────────────────────────────
    base = os.path.join(OUT_DIR, "figS06_evsom_to_alldays_mapping")
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.png", dpi=DPI_RASTER)
    fig.savefig(f"{base}.tiff", dpi=DPI_RASTER)
    print(f"Saved to {OUT_DIR}/")
    plt.close()


if __name__ == "__main__":
    main()
