"""
Supplementary Figure S10 — evSOM TC Tracks and Node Association Histogram

Two-panel figure:
  Left  : TC tracks during FFEs, colored by evSOM node. Full track (faint) and
           ±48 h window (bold) shown; dot marks position at event time.
  Right : Stacked bar chart of TC-associated vs non-TC FFEs per node, with
           dashed line at overall TC fraction.

Usage:
    python -m figure_scripts.figS09_evsom_tc_tracks_and_histogram
"""

import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from som_analysis.config import DATA_DIR, setup_plotting
from som_analysis.helpers import node_label

IBTRACS_PATH = (
    "/home/janoski/nyc_flash_flood/precip_data_and_tc_association_code/"
    "ibtracs.NA.list.v04r01.processed_6hrly.statslp3.csv"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures", "figS09")
os.makedirs(OUT_DIR, exist_ok=True)

TC_CSV = os.path.join(DATA_DIR, "som_2x2_evsom_24h_tc_associations.csv")

# ── Figure / SOM parameters ───────────────────────────────────────────────────
XDIM, YDIM = 2, 2
FIG_WIDTH = 7.0
FIG_HEIGHT = 3.2
DPI_RASTER = 300

# Node colors — consistent across figures
NODE_COLORS = {
    (0, 0): "tab:blue",
    (0, 1): "tab:orange",
    (1, 0): "tab:green",
    (1, 1): "tab:red",
}

# Map extent
LAT_MIN, LAT_MAX = 20.0, 55.0
LON_MIN, LON_MAX = -105.0, -55.0


def main():
    setup_plotting()

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading TC association data ...")
    tc_df = pd.read_csv(TC_CSV, parse_dates=["timestamp"])
    tc_df["tc_present"] = tc_df["tc_present"].astype(bool)

    print(f"Loading IBTrACS from {IBTRACS_PATH} ...")
    ibtracs = pd.read_csv(IBTRACS_PATH, parse_dates=["ISO_TIME"])

    tc_events = tc_df[tc_df["tc_present"]].sort_values("timestamp")

    # ── Build figure with gridspec (map wider than bar chart) ─────────────────
    fig = plt.figure(
        figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI_RASTER, constrained_layout=True
    )
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1])
    fig.get_layout_engine().set(wspace=0.02)

    ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
    ax_bar = fig.add_subplot(gs[1])

    # ── Left: TC track map ────────────────────────────────────────────────────
    ax_map.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#ebebeb", zorder=0)
    ax_map.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#dde8f2", zorder=0)
    ax_map.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.25, zorder=2)
    ax_map.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.4, zorder=2)
    ax_map.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.3, zorder=2)

    plotted_nodes = set()
    for _, row in tc_events.iterrows():
        event_time = row["timestamp"]
        ni, nj = int(row["node_i"]), int(row["node_j"])
        color = NODE_COLORS[(ni, nj)]
        lbl = node_label(ni, nj)

        for sid in row["storm_ids"].split(", "):
            track = ibtracs[ibtracs["SID"] == sid.strip()].sort_values("ISO_TIME")
            if len(track) < 2:
                continue

            # Full track (faint)
            ax_map.plot(
                track["LON"],
                track["LAT"],
                color=color,
                lw=0.5,
                alpha=0.25,
                transform=ccrs.PlateCarree(),
            )

            # ±48 h window (bold)
            win = (track["ISO_TIME"] >= event_time - pd.Timedelta(hours=48)) & (
                track["ISO_TIME"] <= event_time + pd.Timedelta(hours=48)
            )
            wt = track[win]
            if len(wt) > 1:
                ax_map.plot(
                    wt["LON"],
                    wt["LAT"],
                    color=color,
                    lw=1.0,
                    alpha=0.85,
                    transform=ccrs.PlateCarree(),
                )

            # Dot at event time
            ci = (track["ISO_TIME"] - event_time).abs().idxmin()
            cp = track.loc[ci]
            label_kw = dict(label=lbl) if (ni, nj) not in plotted_nodes else {}
            ax_map.scatter(
                cp["LON"],
                cp["LAT"],
                color=color,
                s=10,
                marker="o",
                edgecolors="k",
                linewidths=0.3,
                zorder=5,
                transform=ccrs.PlateCarree(),
                **label_kw,
            )
            plotted_nodes.add((ni, nj))

    # TC-association domain box
    dom_lon = [-100.0, -68.0]
    dom_lat = [32.0, 54.0]
    ax_map.plot(
        [dom_lon[0], dom_lon[1], dom_lon[1], dom_lon[0], dom_lon[0]],
        [dom_lat[0], dom_lat[0], dom_lat[1], dom_lat[1], dom_lat[0]],
        "k--",
        lw=0.8,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )

    # NYC star
    ax_map.scatter(
        -74.0,
        40.7,
        color="black",
        s=30,
        marker="*",
        zorder=8,
        transform=ccrs.PlateCarree(),
    )

    # Legend
    legend_handles = [
        plt.Line2D([0], [0], color=NODE_COLORS[(i, j)], lw=1.5, label=node_label(i, j))
        for j in range(YDIM)
        for i in range(XDIM)
    ] + [
        plt.Line2D([0], [0], color="gray", lw=0.5, alpha=0.4, label="Full track"),
        plt.Line2D([0], [0], color="gray", lw=1.0, label=r"$\pm$48 h"),
    ]
    ax_map.legend(
        handles=legend_handles, fontsize=6.5, loc="lower left", framealpha=0.8, ncol=2
    )

    ax_map.text(
        -0.02,
        1.01,
        "(a)",
        transform=ax_map.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="right",
        va="bottom",
    )

    # ── Right: stacked bar chart ──────────────────────────────────────────────
    node_labels_flat = [node_label(i, j) for j in range(YDIM) for i in range(XDIM)]
    tc_counts = np.zeros(XDIM * YDIM)
    non_tc_counts = np.zeros(XDIM * YDIM)

    for k, (i, j) in enumerate((i, j) for j in range(YDIM) for i in range(XDIM)):
        nd = tc_df[(tc_df["node_i"] == i) & (tc_df["node_j"] == j)]
        tc_counts[k] = nd["tc_present"].sum()
        non_tc_counts[k] = (~nd["tc_present"]).sum()

    x = np.arange(XDIM * YDIM)
    totals = tc_counts + non_tc_counts
    colors = [NODE_COLORS[(i, j)] for j in range(YDIM) for i in range(XDIM)]

    tc_pct = 100 * tc_counts / totals
    overall_pct = 100 * tc_counts.sum() / totals.sum()

    ax_bar.bar(x, tc_pct, 0.6, color=colors, alpha=0.85, linewidth=0)

    # Label each bar with n_TC / n_total
    for xi, (pct, tc, tot) in enumerate(zip(tc_pct, tc_counts, totals, strict=False)):
        ax_bar.text(
            xi,
            pct + 0.5,
            f"{int(tc)}/{int(tot)}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )

    # Overall TC % dashed line
    ax_bar.axhline(
        overall_pct, color="k", lw=0.8, ls="--", label=f"Overall: {overall_pct:.1f}\\%"
    )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(node_labels_flat, fontsize=7.5)
    ax_bar.set_ylabel(r"TC-associated FFEs (\%)", fontsize=8)
    ax_bar.set_ylim(0, max(tc_pct) + 12)
    ax_bar.tick_params(axis="y", labelsize=7.5)
    ax_bar.legend(fontsize=6.5, loc="upper left", framealpha=0.8)

    ax_bar.text(
        -0.18,
        1.01,
        "(b)",
        transform=ax_bar.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    base = os.path.join(OUT_DIR, "figS09_evsom_tc_tracks_and_histogram")
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.png", dpi=DPI_RASTER)
    fig.savefig(f"{base}.tiff", dpi=DPI_RASTER)
    print(f"Saved to {OUT_DIR}/")
    plt.close()


if __name__ == "__main__":
    main()
