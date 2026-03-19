"""Cross-mapping analysis between FFE-only SOM nodes and all-days SOM nodes.

For each flash-flood event the script finds its Best-Matching Unit in both
the FFE SOM and the all-days SOM, then visualises how the two grids relate.

Usage
-----
    python -m som_analysis.map_som_nodes --ffe-var IVT --alldays-var IVT
    python -m som_analysis.map_som_nodes --ffe-var thetae --alldays-var thetae
    python -m som_analysis.map_som_nodes --ffe-var IVT --alldays-var thetae
"""

import argparse
import os

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyBboxPatch
from scipy.stats import entropy

from som_analysis.config import (
    FIGS_DIR,
    MOISTURE_CONFIGS,
    get_alldays_paths,
    get_paths,
    setup_plotting,
)
from som_analysis.helpers import node_label

# ── Fixed colour palette for FFE nodes ───────────────────────────────────────

_FFE_COLORS_FIXED = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _ffe_color_map(xdim, ydim):
    """Return a dict mapping (i, j) → colour for FFE nodes."""
    nodes = [(i, j) for i in range(xdim) for j in range(ydim)]
    return {
        node: _FFE_COLORS_FIXED[k % len(_FFE_COLORS_FIXED)]
        for k, node in enumerate(nodes)
    }


# ── 1. Data loading ──────────────────────────────────────────────────────────


def load_and_merge(ffe_csv, alldays_csv):
    """Load both BMU CSVs and inner-join on date.

    The FFE CSV uses event timestamps (multiple per day possible); the all-days
    CSV uses daily timestamps.  Both are floored to date before merging so that
    days with multiple flash-flood events are each matched to the same all-days
    BMU, preserving the full 117-event count.

    Returns
    -------
    pandas.DataFrame with columns:
        date, node_ffe_i, node_ffe_j, node_all_i, node_all_j,
        node_ffe (str label), node_all (str label)
    """
    ffe = pd.read_csv(ffe_csv, parse_dates=["timestamp"])
    alldays = pd.read_csv(alldays_csv, parse_dates=["timestamp"])

    ffe["date"] = ffe["timestamp"].dt.normalize()
    alldays["date"] = alldays["timestamp"].dt.normalize()

    ffe = ffe.rename(columns={"node_i": "node_ffe_i", "node_j": "node_ffe_j"})
    alldays = alldays.rename(columns={"node_i": "node_all_i", "node_j": "node_all_j"})

    merged = ffe.merge(
        alldays[["date", "node_all_i", "node_all_j"]], on="date", how="inner"
    )

    merged["node_ffe"] = merged.apply(
        lambda r: node_label(int(r["node_ffe_i"]), int(r["node_ffe_j"])), axis=1
    )
    merged["node_all"] = merged.apply(
        lambda r: node_label(int(r["node_all_i"]), int(r["node_all_j"])), axis=1
    )

    return merged


# ── 2. Console output ────────────────────────────────────────────────────────


def print_crosstab(merged, xffe, yffe, xall, yall):
    """Print cross-tabulation and per-node summary statistics."""
    crosstab = pd.crosstab(merged["node_ffe"], merged["node_all"])
    print("\nCross-tabulation of FFE SOM nodes to all-days SOM nodes:")
    print(crosstab.to_string())

    max_ent = np.log(xall * yall)
    print(f"\n\nMapping Summary by FFE SOM Node (entropy max = {max_ent:.2f}):")
    print("=" * 60)

    for node_ffe in sorted(merged["node_ffe"].unique()):
        subset = merged[merged["node_ffe"] == node_ffe]
        n_events = len(subset)
        dist = subset["node_all"].value_counts()
        ent = entropy(dist / dist.sum())
        top_node = dist.idxmax()
        top_pct = 100 * dist.max() / n_events

        print(f"\nFFE Node {node_ffe} ({n_events} events):")
        print(
            f"  Spread across {len(dist)} of {xall * yall} possible all-days nodes"
        )
        print(f"  Entropy: {ent:.2f}")
        print(f"  Top destination: all-days {top_node} ({top_pct:.1f}%)")
        for node_all, count in dist.items():
            print(f"    -> all-days {node_all}: {count} ({100 * count / n_events:.1f}%)")

    print("\n\nComposition of All-Days SOM Nodes by FFE Source:")
    print("=" * 60)
    for bi in range(xall):
        for bj in range(yall):
            node_all = node_label(bi, bj)
            subset = merged[merged["node_all"] == node_all]
            n_events = len(subset)
            if n_events == 0:
                continue
            composition = subset.groupby("node_ffe").size()
            comp_str = ", ".join(
                f"{nd}: {count} ({100 * count / n_events:.0f}%)"
                for nd, count in composition.items()
            )
            print(f"all-days {node_all} (n={n_events}): {comp_str}")


# ── 3. Heatmaps figure ───────────────────────────────────────────────────────


def plot_heatmaps(merged, xffe, yffe, xall, yall, out_path):
    """One subplot per FFE node showing % of its events in each all-days node."""
    fig, axes = plt.subplots(
        yffe, xffe,
        figsize=(4 * xffe, 3.5 * yffe),
        dpi=600,
        layout="constrained",
    )

    # Normalise axes to always be 2-D
    axes = np.atleast_2d(axes)
    if axes.shape == (yffe, xffe):
        pass
    elif axes.shape == (xffe, yffe):
        axes = axes.T

    last_im = None
    for i in range(xffe):
        for j in range(yffe):
            ax = axes[j, i]
            node_ffe = node_label(i, j)
            subset = merged[merged["node_ffe"] == node_ffe]
            n_events = len(subset)

            counts = np.zeros((xall, yall))
            for _, row in subset.iterrows():
                counts[int(row["node_all_i"]), int(row["node_all_j"])] += 1

            pcts = 100 * counts / n_events if n_events > 0 else counts

            im = ax.imshow(pcts.T, cmap="YlOrRd", origin="lower", vmin=0, vmax=50)
            last_im = im

            for bi in range(xall):
                for bj in range(yall):
                    count = int(counts[bi, bj])
                    if count > 0:
                        text_color = "white" if pcts[bi, bj] > 25 else "black"
                        ax.text(
                            bi, bj, str(count),
                            ha="center", va="center",
                            fontsize=6, color=text_color,
                        )

            ax.set_title(f"FF-Only SOM Node {node_label(i, j)}\nn={n_events}", fontsize=8)
            ax.set_xlabel("All-Days SOM col", fontsize=6)
            ax.set_ylabel("All-Days SOM row", fontsize=6)
            ax.set_xticks(np.arange(xall))
            ax.set_yticks(np.arange(yall))
            ax.tick_params(labelsize=5)
            ax.invert_yaxis()

    if last_im is not None:
        cbar = fig.colorbar(
            last_im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02
        )
        cbar.set_label(r"Percentage of Events (\%)", fontsize=7)

    plt.suptitle(
        "Distribution of FF-Only SOM Events Across All-Days SOM Nodes",
        fontsize=10,
        y=1.02,
    )
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved heatmaps -> {out_path}")


# ── 4. Sankey-style flow diagram ──────────────────────────────────────────────


def plot_flow(merged, xffe, yffe, xall, yall, out_path):
    """Sankey-style flow diagram: FFE nodes (left) to all-days nodes (right)."""
    color_map = _ffe_color_map(xffe, yffe)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=600)

    left_x, right_x = 0.1, 0.9
    small_nodes = [(i, j) for i in range(xffe) for j in range(yffe)]
    big_nodes = [(i, j) for i in range(xall) for j in range(yall)]
    small_y = np.linspace(0.85, 0.15, len(small_nodes))
    big_y = np.linspace(0.95, 0.05, len(big_nodes))

    # Draw FFE nodes (left column)
    small_centers = {}
    for idx, (i, j) in enumerate(small_nodes):
        y = small_y[idx]
        small_centers[(i, j)] = (left_x, y)
        n = len(merged[merged["node_ffe"] == node_label(i, j)])
        box = FancyBboxPatch(
            (left_x - 0.04, y - 0.03), 0.08, 0.06,
            boxstyle="round,pad=0.01",
            facecolor=color_map[(i, j)],
            edgecolor="black", linewidth=1,
        )
        ax.add_patch(box)
        ax.text(
            left_x, y, f"{node_label(i, j)}\nn={n}",
            ha="center", va="center", fontsize=8, fontweight="bold",
        )

    # Draw all-days nodes (right column)
    big_centers = {}
    for idx, (i, j) in enumerate(big_nodes):
        y = big_y[idx]
        big_centers[(i, j)] = (right_x, y)
        n = len(merged[merged["node_all"] == node_label(i, j)])
        box = FancyBboxPatch(
            (right_x - 0.03, y - 0.018), 0.06, 0.036,
            boxstyle="round,pad=0.005",
            facecolor="lightgray",
            edgecolor="black", linewidth=0.5,
        )
        ax.add_patch(box)
        ax.text(
            right_x, y, f"{node_label(i, j)}\n{n}",
            ha="center", va="center", fontsize=5,
        )

    # Draw connecting flows
    for si, sj in small_nodes:
        small_label = node_label(si, sj)
        subset = merged[merged["node_ffe"] == small_label]
        n_total = len(subset)
        if n_total == 0:
            continue
        for bi, bj in big_nodes:
            n_flow = len(subset[subset["node_all"] == node_label(bi, bj)])
            if n_flow == 0:
                continue
            frac = n_flow / n_total
            x1, y1 = small_centers[(si, sj)]
            x2, y2 = big_centers[(bi, bj)]
            ax.annotate(
                "",
                xy=(x2 - 0.03, y2),
                xytext=(x1 + 0.04, y1),
                arrowprops=dict(
                    arrowstyle="->",
                    connectionstyle="arc3,rad=0.1",
                    color=to_rgba(color_map[(si, sj)], 0.3 + 0.5 * frac),
                    linewidth=0.5 + 3 * frac,
                ),
            )

    ax.text(
        left_x, 0.98, f"{xffe}x{yffe} SOM\n(FFE-only)",
        ha="center", va="bottom", fontsize=10, fontweight="bold",
    )
    ax.text(
        right_x, 0.98, f"{xall}x{yall} SOM\n(All Daily)",
        ha="center", va="bottom", fontsize=10, fontweight="bold",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.axis("off")
    ax.set_title(
        f"Flow of Flash Flood Events: {xffe}x{yffe} SOM to {xall}x{yall} SOM Mapping",
        fontsize=12, pad=20,
    )

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved flow diagram -> {out_path}")


# ── 5. Composition grid ───────────────────────────────────────────────────────


def plot_composition(merged, xffe, yffe, xall, yall, out_path):
    """5×4 grid with one stacked bar per all-days node coloured by FFE source."""
    color_map = _ffe_color_map(xffe, yffe)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=600)

    for i in range(xall):
        for j in range(yall):
            node_all = node_label(i, j)
            subset = merged[merged["node_all"] == node_all]
            n_total = len(subset)

            if n_total == 0:
                rect = plt.Rectangle(
                    (i - 0.45, j - 0.45), 0.9, 0.9,
                    facecolor="lightgray", edgecolor="black", linewidth=0.5,
                )
                ax.add_patch(rect)
                ax.text(i, j, "n=0", ha="center", va="center", fontsize=6)
                continue

            # Collect non-zero fractions (keyed by (i,j) tuple)
            fractions = {}
            for node_ffe in sorted(color_map.keys()):
                nl = node_label(node_ffe[0], node_ffe[1])
                count = len(subset[subset["node_ffe"] == nl])
                if count > 0:
                    fractions[node_ffe] = count / n_total

            # Draw stacked horizontal bar
            cumulative = 0.0
            for node_ffe, frac in sorted(fractions.items()):
                rect = plt.Rectangle(
                    (i - 0.45 + cumulative * 0.9, j - 0.45),
                    0.9 * frac, 0.9,
                    facecolor=color_map[node_ffe],
                    edgecolor="black", linewidth=0.3,
                )
                ax.add_patch(rect)
                cumulative += frac

            ax.text(
                i, j, f"n={n_total}",
                ha="center", va="center",
                fontsize=6, fontweight="bold", color="white",
                path_effects=[
                    patheffects.withStroke(linewidth=2, foreground="black")
                ],
            )

    # Legend
    legend_elements = [
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor=color_map[(i, j)],
            label=f"FFE {node_label(i, j)}",
        )
        for i in range(xffe)
        for j in range(yffe)
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left", bbox_to_anchor=(1.02, 1),
        fontsize=8, title="FFE Source Node",
    )

    ax.set_xlim(-0.6, xall - 0.4)
    ax.set_ylim(-0.6, yall - 0.4)
    ax.set_xticks(np.arange(xall))
    ax.set_yticks(np.arange(yall))
    ax.set_xlabel(f"{xall}x{yall} SOM col", fontsize=9)
    ax.set_ylabel(f"{xall}x{yall} SOM row", fontsize=9)
    ax.invert_yaxis()
    ax.set_title(
        f"{xall}x{yall} SOM Node Composition by {xffe}x{yffe} SOM Source",
        fontsize=10,
    )
    ax.set_aspect("equal")
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved composition grid -> {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Map FFE-only SOM nodes to all-days SOM nodes."
    )
    parser.add_argument(
        "--ffe-var", required=True,
        choices=list(MOISTURE_CONFIGS.keys()),
        help="Moisture variable of the FFE SOM",
    )
    parser.add_argument(
        "--alldays-var", default=None,
        choices=list(MOISTURE_CONFIGS.keys()),
        help="Moisture variable of the all-days SOM (default: same as --ffe-var)",
    )
    parser.add_argument(
        "--xdim-ffe", type=int, default=2,
        help="FFE SOM columns (default 2)",
    )
    parser.add_argument(
        "--ydim-ffe", type=int, default=2,
        help="FFE SOM rows (default 2)",
    )
    parser.add_argument(
        "--xdim-alldays", type=int, default=5,
        help="All-days SOM columns (default 5)",
    )
    parser.add_argument(
        "--ydim-alldays", type=int, default=4,
        help="All-days SOM rows (default 4)",
    )
    args = parser.parse_args()

    if args.alldays_var is None:
        args.alldays_var = args.ffe_var

    setup_plotting()

    ffe_paths = get_paths(args.ffe_var)
    alldays_paths = get_alldays_paths(
        args.alldays_var, args.xdim_alldays, args.ydim_alldays
    )

    ffe_csv = ffe_paths["bmu_csv_path"]
    alldays_csv = alldays_paths["bmu_csv_path"]
    ffe_lbl = MOISTURE_CONFIGS[args.ffe_var]["file_label"]
    all_lbl = MOISTURE_CONFIGS[args.alldays_var]["file_label"]
    xffe, yffe = args.xdim_ffe, args.ydim_ffe
    xall, yall = args.xdim_alldays, args.ydim_alldays

    print(f"FFE BMU CSV:      {ffe_csv}")
    print(f"All-days BMU CSV: {alldays_csv}")

    merged = load_and_merge(ffe_csv, alldays_csv)
    print(f"Matched {len(merged)} events between the two SOMs")

    print_crosstab(merged, xffe, yffe, xall, yall)

    os.makedirs(FIGS_DIR, exist_ok=True)
    stem = f"som_{xffe}x{yffe}_{ffe_lbl}_to_{xall}x{yall}_{all_lbl}"

    plot_heatmaps(
        merged, xffe, yffe, xall, yall,
        os.path.join(FIGS_DIR, f"{stem}_heatmaps.png"),
    )
    plot_flow(
        merged, xffe, yffe, xall, yall,
        os.path.join(FIGS_DIR, f"{stem}_flow.png"),
    )
    plot_composition(
        merged, xffe, yffe, xall, yall,
        os.path.join(
            FIGS_DIR,
            f"som_{xall}x{yall}_{all_lbl}_composition_by_{xffe}x{yffe}_{ffe_lbl}.png",
        ),
    )


if __name__ == "__main__":
    main()
