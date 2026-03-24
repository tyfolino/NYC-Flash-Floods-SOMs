"""
SOM Diagnostic Plots for NYC Flash Flood Analysis
By: Ty Janoski

Generates all diagnostic plots for a trained SOM: U-matrix, Sammon map,
node weights, composites, representativeness, trough/maximum locations,
individual event maps, and monthly histograms.

Usage:
    python -m som_analysis.plot_som --moisture-var thetae
    python -m som_analysis.plot_som --moisture-var IVT --skip-indiv
"""

import argparse
import os
from itertools import combinations

import cartopy.crs as ccrs
import cmweather  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import chi2_contingency, pearsonr

from .config import MOISTURE_CONFIGS, SOM_INTERMEDIATE_PATH, get_paths, setup_plotting
from .helpers import (
    add_map_features,
    compute_composites,
    create_som_figure,
    get_node_indices,
    load_moist_var,
    node_label,
    plot_node_events,
    plot_single_events,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate diagnostic plots for a trained SOM."
    )
    parser.add_argument(
        "--moisture-var",
        required=True,
        choices=list(MOISTURE_CONFIGS.keys()),
        help="Moisture variable (IVT, tcwv, thetae).",
    )
    parser.add_argument(
        "--moisture-weight",
        type=float,
        default=1,
        help="Moisture weight used during training (default: 1).",
    )
    parser.add_argument(
        "--skip-indiv",
        action="store_true",
        help="Skip individual node event plots (slow).",
    )
    parser.add_argument(
        "--single-events",
        action="store_true",
        help="Save each event as its own figure (one file per event per node).",
    )
    parser.add_argument(
        "--daily",
        action="store_true",
        help="Use daily-mean FFE fields (must match --daily used in train_som).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_plotting()

    cfg = MOISTURE_CONFIGS[args.moisture_var]
    paths = get_paths(args.moisture_var, args.moisture_weight, daily=args.daily)
    fig_dir = paths["fig_dir"]
    _lbl = paths["file_label"]
    ffe_suffix = "_ffe_daily" if args.daily else "_ffe"
    moist_time_dim = cfg["time_dim"]
    moist_label = cfg["label"]
    moist_label_short = cfg["label_short"]
    moist_units_raw = cfg["units_raw"]
    levels_moist_weights = cfg["levels_weights"]
    levels_moist_anom = cfg["levels_anom"]
    levels_moist_raw = cfg["levels_raw"]
    levels_moist_indiv = cfg["levels_indiv"]
    cmap_moist_raw = cfg["cmap_raw"]

    xdim, ydim = 2, 2
    n_nodes = xdim * ydim
    pfx = cfg["file_prefix"]
    var_name = cfg["var_name"]

    os.makedirs(f"{fig_dir}/indiv-nodes", exist_ok=True)

    # ── Load cached SOM results ───────────────────────────────────────────────
    cache_name = "som_results_daily.npz" if args.daily else "som_results.npz"
    cache_path = os.path.join(fig_dir, ".cache", cache_name)
    print(f"Loading SOM results from {cache_path} ...")
    cached = np.load(cache_path)
    z500_nodes = cached["z500_nodes"]
    moist_nodes = cached["moist_nodes"]
    bmus = cached["bmus"]
    u_matrix = cached["u_matrix"]
    hit_map = cached["hit_map"]
    coords = cached["coords"]

    # ── Load data for composites ──────────────────────────────────────────────
    print("Loading data files ...")
    moist_norm_weighted_ffe = load_moist_var(
        f"{SOM_INTERMEDIATE_PATH}{pfx}_norm_weighted{ffe_suffix}.nc", var_name
    )
    moist_norm_ffe = load_moist_var(
        f"{SOM_INTERMEDIATE_PATH}{pfx}_norm{ffe_suffix}.nc", var_name
    )
    moist_ffe = load_moist_var(f"{SOM_INTERMEDIATE_PATH}{pfx}{ffe_suffix}.nc", var_name)

    z500_norm_weighted_ffe = xr.load_dataarray(
        f"{SOM_INTERMEDIATE_PATH}era5_Z500_norm_weighted{ffe_suffix}.nc"
    )
    z500_norm_ffe = xr.load_dataarray(
        f"{SOM_INTERMEDIATE_PATH}era5_Z500_norm{ffe_suffix}.nc"
    )
    z500_ffe = xr.load_dataarray(f"{SOM_INTERMEDIATE_PATH}era5_Z500{ffe_suffix}.nc")
    z500_time_dim = z500_norm_ffe.dims[0]
    tp_ffe = xr.load_dataarray(f"{SOM_INTERMEDIATE_PATH}era5_tp{ffe_suffix}.nc")
    mslp_ffe = xr.load_dataarray(f"{SOM_INTERMEDIATE_PATH}era5_mslp{ffe_suffix}.nc")

    lat = moist_norm_ffe[cfg["lat_dim"]]
    lon = moist_norm_ffe[cfg["lon_dim"]]
    # ── U-matrix and hit map ──────────────────────────────────────────────────
    print("Plotting U-matrix and hit map ...")
    fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(6, 3), dpi=600)

    im0 = axes[0].imshow(u_matrix, cmap="viridis", origin="lower")
    axes[0].set_title("U-Matrix (Mean Inter-Node Distance)", fontsize=7)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, shrink=0.7)

    im1 = axes[1].imshow(hit_map, cmap="plasma", origin="lower")
    axes[1].set_title("Hit Map (Samples per Node)", fontsize=7)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, shrink=0.7)

    for ax in axes:
        ax.set_xticks(np.arange(xdim))
        ax.set_yticks(np.arange(ydim))
        ax.set_xlabel("X-index", fontsize=6)
        ax.set_ylabel("Y-index", fontsize=6)

    plt.savefig(f"{fig_dir}/Z500_{_lbl}_som_u_matrix_hit_map.png")
    plt.close()

    # ── Sammon / MDS map ──────────────────────────────────────────────────────
    print("Plotting Sammon/MDS map ...")
    U_flat = u_matrix.T.reshape(-1)
    hits_flat = hit_map.T.reshape(-1)
    hits_scaled = 30 + 250 * (hits_flat / hits_flat.max())

    plt.figure(figsize=(7, 7))
    sc = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=U_flat,
        s=hits_scaled,
        cmap="balance",
        edgecolor="k",
        linewidth=0.5,
        zorder=3,
    )

    for i in range(xdim):
        for j in range(ydim):
            node = i * ydim + j
            if j + 1 < ydim:
                nbr = i * ydim + (j + 1)
                plt.plot(
                    [coords[node, 0], coords[nbr, 0]],
                    [coords[node, 1], coords[nbr, 1]],
                    "k-",
                    lw=0.6,
                    alpha=0.4,
                )
            if i + 1 < xdim:
                nbr = (i + 1) * ydim + j
                plt.plot(
                    [coords[node, 0], coords[nbr, 0]],
                    [coords[node, 1], coords[nbr, 1]],
                    "k-",
                    lw=0.6,
                    alpha=0.4,
                )

    for idx, (x, y) in enumerate(coords):
        ix, iy = divmod(idx, ydim)
        plt.text(
            x, y, node_label(ix, iy), fontsize=8, ha="center", va="center", zorder=5
        )

    plt.title(
        "Sammon / MDS Distortion Grid\nU-Matrix (Color) \\& Node Frequency (Size)"
    )
    plt.axis("off")
    plt.colorbar(sc, label="U-Matrix (Avg. Neighbor Distance)")
    plt.savefig(f"{fig_dir}/Z500_{_lbl}_som_sammon_mds.png", bbox_inches="tight")
    plt.close()

    # ── Node weights map ──────────────────────────────────────────────────────
    print("Plotting node weights ...")
    levels_Z = np.arange(-1.4, 1.41, 0.2)
    fig, axes = create_som_figure(xdim, ydim)

    for i in range(xdim):
        for j in range(ydim):
            ax = axes[j, i]
            im = ax.contourf(
                lon,
                lat,
                moist_nodes[i, j],
                cmap="balance",
                levels=levels_moist_weights,
                transform=ccrs.PlateCarree(),
            )
            cn = ax.contour(
                lon,
                lat,
                z500_nodes[i, j],
                colors="black",
                linewidths=0.5,
                levels=levels_Z,
                transform=ccrs.PlateCarree(),
            )
            ax.clabel(cn, inline=True, fontsize=5, fmt="%.1f")
            add_map_features(ax)
            ax.set_title(node_label(i, j), fontsize=6)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label(f"Standardized {moist_label} Anomaly (Shaded)", fontsize=6)
    plt.suptitle(
        f"Flash Flood Only SOM: Node Weight Patterns\n"
        f"Z500 (contoured) + {moist_label_short} (shaded)",
        fontsize=8,
    )
    plt.savefig(
        f"{fig_dir}/combined_node_weights_{_lbl}_shaded.png", bbox_inches="tight"
    )
    plt.close()

    # ── Anomaly composite map ─────────────────────────────────────────────────
    print("Plotting anomaly composites ...")
    z500_patterns, counts = compute_composites(
        z500_norm_ffe, bmus, xdim, ydim, time_dim=z500_time_dim
    )
    moist_patterns, _ = compute_composites(
        moist_norm_ffe, bmus, xdim, ydim, time_dim=moist_time_dim
    )
    levels_Z_anom = np.arange(-2.0, 2.1, 0.25)

    fig, axes = create_som_figure(xdim, ydim)
    for i in range(xdim):
        for j in range(ydim):
            ax = axes[j, i]
            im = ax.contourf(
                lon,
                lat,
                moist_patterns[i, j],
                cmap="balance",
                levels=levels_moist_anom,
                transform=ccrs.PlateCarree(),
                extend="both",
            )
            ax.contour(
                lon,
                lat,
                z500_patterns[i, j],
                colors="black",
                linewidths=0.5,
                levels=levels_Z_anom,
                transform=ccrs.PlateCarree(),
            )
            add_map_features(ax)
            ax.set_title(f"{node_label(i, j)}  N={counts[i, j]}", fontsize=6)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label("Standardized Anomaly", fontsize=6)
    plt.suptitle(
        f"Flash Flood Only SOM Composite Anomalies: "
        f"Z500 (contoured) + {moist_label_short} (shaded)",
        fontsize=8,
        y=1.04,
    )
    plt.savefig(
        f"{fig_dir}/Z500_and_{_lbl}_SOM_composite_anomalies_{_lbl}_shaded.png",
        bbox_inches="tight",
    )
    plt.close()

    # ── Representativeness ────────────────────────────────────────────────────
    print("Computing representativeness ...")
    z500_patterns_weighted, node_counts = compute_composites(
        z500_norm_weighted_ffe, bmus, xdim, ydim, time_dim=z500_time_dim
    )
    moist_patterns_weighted, _ = compute_composites(
        moist_norm_weighted_ffe, bmus, xdim, ydim, time_dim=moist_time_dim
    )

    representativeness = []
    print("=" * 75)
    print("NODE REPRESENTATIVENESS: Pattern Correlations (Weights vs. Composites)")
    print("=" * 75)
    print(
        f"\n{'Node':<8} {'N':>4} {'r(Z500)':>10} "
        f"{'r(' + moist_label_short + ')':>10} {'r(Combined)':>12} {'Interpretation':<20}"
    )
    print("-" * 75)

    for i in range(xdim):
        for j in range(ydim):
            n_events = int(node_counts[i, j])
            z500_weight = z500_nodes[i, j].flatten()
            moist_weight = moist_nodes[i, j].flatten()
            z500_composite = z500_patterns_weighted[i, j].flatten()
            moist_composite = moist_patterns_weighted[i, j].flatten()

            if n_events > 0 and not np.any(np.isnan(z500_composite)):
                r_z500, _ = pearsonr(z500_weight, z500_composite)
                r_moist, _ = pearsonr(moist_weight, moist_composite)
                combined_weight = np.concatenate([z500_weight, moist_weight])
                combined_composite = np.concatenate([z500_composite, moist_composite])
                r_combined, _ = pearsonr(combined_weight, combined_composite)
            else:
                r_z500 = r_moist = r_combined = np.nan

            if r_combined >= 0.9:
                interp = "Excellent"
            elif r_combined >= 0.8:
                interp = "Good"
            elif r_combined >= 0.7:
                interp = "Moderate"
            else:
                interp = "Poor"

            representativeness.append(
                {
                    "node": node_label(i, j),
                    "n": n_events,
                    "r_z500": r_z500,
                    "r_moist": r_moist,
                    "r_combined": r_combined,
                }
            )
            print(
                f"{node_label(i, j)}{'':<4} {n_events:>4} {r_z500:>10.3f} "
                f"{r_moist:>10.3f} {r_combined:>12.3f} {interp:<20}"
            )

    print("-" * 75)
    r_combined_values = [
        r["r_combined"] for r in representativeness if not np.isnan(r["r_combined"])
    ]
    print(
        f"\nSummary: Mean r(Combined) = {np.mean(r_combined_values):.3f}, "
        f"Min = {np.min(r_combined_values):.3f}, Max = {np.max(r_combined_values):.3f}"
    )

    # Representativeness figure
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), dpi=600, constrained_layout=True)

    ax = axes[0]
    r_grid = np.zeros((xdim, ydim))
    for i in range(xdim):
        for j in range(ydim):
            idx = i * ydim + j
            r_grid[i, j] = representativeness[idx]["r_combined"]

    im = ax.imshow(r_grid.T, cmap="RdYlGn", vmin=0.3, vmax=1.0, origin="lower")
    ax.set_xticks(np.arange(xdim))
    ax.set_yticks(np.arange(ydim))
    ax.set_xlabel("X-index", fontsize=7)
    ax.set_ylabel("Y-index", fontsize=7)
    ax.set_title("Combined Pattern Correlation\n(Weights vs. Composites)", fontsize=8)

    for i in range(xdim):
        for j in range(ydim):
            r_val = r_grid[i, j]
            color = "white" if r_val < 0.75 else "black"
            ax.text(
                i,
                j,
                f"{r_val:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson r", fontsize=7)

    ax = axes[1]
    node_labels = [r["node"] for r in representativeness]
    x = np.arange(len(node_labels))
    width = 0.25

    ax.bar(
        x - width,
        [r["r_z500"] for r in representativeness],
        width,
        label="Z500",
        color="steelblue",
        alpha=0.9,
    )
    ax.bar(
        x,
        [r["r_moist"] for r in representativeness],
        width,
        label=moist_label_short,
        color="coral",
        alpha=0.9,
    )
    ax.bar(
        x + width,
        [r["r_combined"] for r in representativeness],
        width,
        label="Combined",
        color="seagreen",
        alpha=0.9,
    )
    ax.axhline(
        0.8, color="gray", linestyle="--", linewidth=0.8, label="r = 0.8 threshold"
    )
    ax.set_xlabel("SOM Node", fontsize=7)
    ax.set_ylabel("Pattern Correlation (r)", fontsize=7)
    ax.set_title("Representativeness by Variable", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(node_labels, fontsize=6)
    ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=6, loc="lower right")
    ax.grid(True, linewidth=0.3, alpha=0.5, axis="y")

    plt.savefig(
        f"{fig_dir}/Z500_and_{_lbl}_som_representativeness.png", bbox_inches="tight"
    )
    plt.close()
    print("Saved representativeness figure.")

    # ── Composite mean map ────────────────────────────────────────────────────
    print("Plotting composite means ...")
    z500_patterns_raw, _ = compute_composites(
        z500_ffe, bmus, xdim, ydim, time_dim=z500_time_dim
    )
    moist_patterns_raw, _ = compute_composites(
        moist_ffe, bmus, xdim, ydim, time_dim=moist_time_dim
    )

    levels_Z_raw = range(552, 595, 3)
    fig, axes = create_som_figure(xdim, ydim)
    for i in range(xdim):
        for j in range(ydim):
            ax = axes[j, i]
            im = ax.contourf(
                lon,
                lat,
                moist_patterns_raw[i, j],
                cmap=cmap_moist_raw,
                levels=levels_moist_raw,
                transform=ccrs.PlateCarree(),
                extend="max",
            )
            cn = ax.contour(
                lon,
                lat,
                z500_patterns_raw[i, j] / 98.1,
                colors="black",
                linewidths=0.5,
                levels=levels_Z_raw,
                transform=ccrs.PlateCarree(),
            )
            ax.clabel(cn, inline=True, fontsize=5, fmt="%.0f")
            add_map_features(ax)
            ax.set_title(f"{node_label(i, j)}  N={counts[i, j]}", fontsize=6)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label(f"{moist_label_short} ({moist_units_raw})", fontsize=6)
    plt.suptitle(
        f"Flash Flood Only SOM Composite: "
        f"{moist_label_short} (shaded) + Z500 (contoured)",
        fontsize=8,
        y=1.04,
    )
    plt.savefig(
        f"{fig_dir}/Z500_and_{_lbl}_SOM_composite_mean_{_lbl}_shaded.png",
        bbox_inches="tight",
    )
    plt.close()

    # ── Cross-variable composite mean maps ────────────────────────────────────
    # For each moisture variable NOT used in training, composite its raw field
    # using the same BMU assignments.  This shows how synoptic patterns
    # identified by one variable project onto another.
    other_vars = [v for v in MOISTURE_CONFIGS if v != args.moisture_var]
    if other_vars:
        print("Plotting cross-variable composites ...")
    for other_var in other_vars:
        other_cfg = MOISTURE_CONFIGS[other_var]
        other_file_label = other_cfg["file_label"]
        other_pfx = other_cfg["file_prefix"]
        other_var_name = other_cfg["var_name"]
        other_time_dim = other_cfg["time_dim"]

        other_ffe = load_moist_var(
            f"{SOM_INTERMEDIATE_PATH}{other_pfx}{ffe_suffix}.nc", other_var_name
        )
        other_patterns_raw, _ = compute_composites(
            other_ffe, bmus, xdim, ydim, time_dim=other_time_dim
        )

        fig, axes = create_som_figure(xdim, ydim)
        for i in range(xdim):
            for j in range(ydim):
                ax = axes[j, i]
                im = ax.contourf(
                    lon,
                    lat,
                    other_patterns_raw[i, j],
                    cmap=other_cfg["cmap_raw"],
                    levels=other_cfg["levels_raw"],
                    transform=ccrs.PlateCarree(),
                    extend="max",
                )
                cn = ax.contour(
                    lon,
                    lat,
                    z500_patterns_raw[i, j] / 98.1,
                    colors="black",
                    linewidths=0.5,
                    levels=levels_Z_raw,
                    transform=ccrs.PlateCarree(),
                )
                ax.clabel(cn, inline=True, fontsize=5, fmt="%.0f")
                add_map_features(ax)
                ax.set_title(f"{node_label(i, j)}  N={counts[i, j]}", fontsize=6)

        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
        cbar.set_label(
            f"{other_cfg['label_short']} ({other_cfg['units_raw']})", fontsize=6
        )
        plt.suptitle(
            f"Flash Flood Only SOM Composite: "
            f"{other_cfg['label_short']} (shaded) + Z500 (contoured)\n"
            f"(SOM trained on Z500 + {moist_label_short})",
            fontsize=8,
            y=1.04,
        )
        out_fname = f"Z500_and_{_lbl}_SOM_composite_mean_{other_file_label}_shaded.png"
        plt.savefig(f"{fig_dir}/{out_fname}", bbox_inches="tight")
        plt.close()
        print(f"  Saved {out_fname}")

    # ── Z500 trough locations ─────────────────────────────────────────────────
    print("Plotting Z500 trough locations ...")
    fig, axes = create_som_figure(xdim, ydim)
    for i in range(xdim):
        for j in range(ydim):
            ax = axes[j, i]
            idx = get_node_indices(bmus, i, j)
            min_lats, min_lons = [], []
            for k in idx:
                field = z500_norm_ffe.isel({z500_time_dim: k})
                min_idx = field.values.argmin()
                min_lat_idx, min_lon_idx = np.unravel_index(min_idx, field.shape)
                min_lats.append(float(lat[min_lat_idx]))
                min_lons.append(float(lon[min_lon_idx]))

            ax.scatter(
                min_lons,
                min_lats,
                c="blue",
                s=15,
                alpha=0.7,
                edgecolors="black",
                linewidths=0.3,
                transform=ccrs.PlateCarree(),
            )
            add_map_features(ax)
            ax.set_title(f"{node_label(i, j)}  N={counts[i, j]}", fontsize=6)

    plt.suptitle("Z500 Anomaly Minimum Locations by SOM Node", fontsize=8, y=1.04)
    plt.savefig(
        f"{fig_dir}/Z500_and_{_lbl}_SOM_trough_locations.png", bbox_inches="tight"
    )
    plt.close()

    # ── Moisture anomaly max locations ────────────────────────────────────────
    print("Plotting moisture anomaly max locations ...")
    fig, axes = create_som_figure(xdim, ydim)
    for i in range(xdim):
        for j in range(ydim):
            ax = axes[j, i]
            idx = get_node_indices(bmus, i, j)
            max_lats, max_lons = [], []
            for k in idx:
                field = moist_norm_ffe.isel({moist_time_dim: k})
                max_idx = field.values.argmax()
                max_lat_idx, max_lon_idx = np.unravel_index(max_idx, field.shape)
                max_lats.append(float(lat[max_lat_idx]))
                max_lons.append(float(lon[max_lon_idx]))

            ax.scatter(
                max_lons,
                max_lats,
                c="red",
                s=15,
                alpha=0.7,
                edgecolors="black",
                linewidths=0.3,
                transform=ccrs.PlateCarree(),
            )
            add_map_features(ax)
            ax.set_title(f"{node_label(i, j)}  N={counts[i, j]}", fontsize=6)

    plt.suptitle(
        f"{moist_label_short} Anomaly Maximum Locations by SOM Node",
        fontsize=8,
        y=1.04,
    )
    plt.savefig(
        f"{fig_dir}/Z500_and_{_lbl}_SOM_{_lbl}_max_locations.png", bbox_inches="tight"
    )
    plt.close()

    # ── Individual node event maps ────────────────────────────────────────────
    if not args.skip_indiv:
        print("Plotting individual node events (this may take a while) ...")
        plot_node_events(
            moist_ffe,
            bmus,
            xdim,
            ydim,
            lon,
            lat,
            time_dim=moist_time_dim,
            levels=levels_moist_indiv,
            cmap=cmap_moist_raw,
            save_pattern=f"{fig_dir}/indiv-nodes/node_{{i}}_{{j}}.png",
            cbar_label=f"{moist_label_short} ({moist_units_raw})",
            z500_data=z500_ffe,
            z500_levels=range(552, 595, 3),
            z500_scale=1 / 98.1,
            z500_time_dim=z500_time_dim,
        )
        plot_node_events(
            tp_ffe,
            bmus,
            xdim,
            ydim,
            lon,
            lat,
            levels=np.arange(0, 28, 3),
            cmap="HomeyerRainbow",
            save_pattern=f"{fig_dir}/indiv-nodes/node_{{i}}_{{j}}_precip.png",
            scale=1000,
            cbar_label="Total Precipitation (mm)",
        )
        plot_node_events(
            mslp_ffe,
            bmus,
            xdim,
            ydim,
            lon,
            lat,
            levels=np.arange(976, 1041, 4),
            cmap=None,
            save_pattern=f"{fig_dir}/indiv-nodes/node_{{i}}_{{j}}_mslp.png",
            scale=0.01,
            contour=True,
        )
    else:
        print("Skipping individual node event plots (--skip-indiv).")

    # ── Single-event figures ──────────────────────────────────────────────────
    if args.single_events:
        singles_dir = f"{fig_dir}/single-events"
        print(f"Saving single-event figures to {singles_dir}/ ...")
        plot_single_events(
            moist_ffe,
            bmus,
            xdim,
            ydim,
            lon,
            lat,
            time_dim=moist_time_dim,
            levels=levels_moist_indiv,
            cmap=cmap_moist_raw,
            save_dir=singles_dir,
            cbar_label=f"{moist_label_short} ({moist_units_raw})",
            z500_data=z500_ffe,
            z500_levels=range(552, 595, 3),
            z500_scale=1 / 98.1,
            z500_time_dim=z500_time_dim,
        )
        plot_single_events(
            tp_ffe,
            bmus,
            xdim,
            ydim,
            lon,
            lat,
            levels=np.arange(0, 28, 3),
            cmap="HomeyerRainbow",
            save_dir=singles_dir,
            suffix="_precip",
            scale=1000,
            cbar_label="Total Precipitation (mm)",
        )
        plot_single_events(
            mslp_ffe,
            bmus,
            xdim,
            ydim,
            lon,
            lat,
            levels=np.arange(976, 1041, 4),
            cmap=None,
            save_dir=singles_dir,
            suffix="_mslp",
            scale=0.01,
            contour=True,
        )
        print("Done saving single-event figures.")

    # ── Monthly histograms ────────────────────────────────────────────────────
    print("Plotting monthly histograms ...")
    months = pd.to_datetime(moist_ffe[moist_time_dim].values).month.to_numpy()
    month_counts = {}
    for i in range(xdim):
        for j in range(ydim):
            idx = get_node_indices(bmus, i, j)
            node_months = months[idx]
            month_counts[(i, j)] = np.bincount(node_months, minlength=13)[1:]

    month_labels = ["May", "Jun", "Jul", "Aug", "Sep", "Oct"]
    fig, axes = plt.subplots(
        ydim, xdim, figsize=(6, 3.7), constrained_layout=True, dpi=600
    )

    # Total FF events per month (denominator for normalization)
    all_month_counts_warm = np.bincount(months, minlength=13)[1:][4:10].astype(float)
    uniform_frac = 1.0 / n_nodes  # expected fraction if all nodes equally likely

    for i in range(xdim):
        for j in range(ydim):
            ax = axes[j, i]
            monthly = month_counts[(i, j)][4:10]
            frac = np.where(
                all_month_counts_warm > 0, monthly / all_month_counts_warm, 0.0
            )
            ax.bar(month_labels, frac, color="teal", alpha=0.9, width=0.8)
            ax.axhline(
                uniform_frac, color="gray", linewidth=0.7, linestyle="--", alpha=0.8
            )
            ax.set_title(f"{node_label(i, j)}  N={monthly.sum()}", fontsize=6)
            ax.tick_params(axis="x", bottom=False, labelsize=5)
            ax.set_ylim(0, 0.65)
            ax.yaxis.set_major_formatter(
                plt.matplotlib.ticker.FormatStrFormatter("%.1f")
            )
            ax.grid(True, linewidth=0.3, alpha=0.5, axis="y")

    plt.suptitle(
        "Warm-Season (May\u2013Oct) Event Distribution per SOM Node",
        fontsize=8,
        y=1.04,
    )
    plt.savefig(
        f"{fig_dir}/Z500_and_{_lbl}_som_monthly_counts.png", bbox_inches="tight"
    )
    plt.close()

    # ── Chi-square test ───────────────────────────────────────────────────────
    print("\nMonthly distribution chi-square test:")
    node_labels_chi = [node_label(i, j) for i in range(xdim) for j in range(ydim)]
    contingency = np.array(
        [month_counts[(i, j)][4:10] for i in range(xdim) for j in range(ydim)]
    )

    print("Contingency Table (Node x Month):")
    print("-" * 55)
    print(f"{'Node':<10}", end="")
    for m in month_labels:
        print(f"{m:>7}", end="")
    print(f"{'Total':>8}")
    print("-" * 55)
    for k, label in enumerate(node_labels_chi):
        print(f"{label:<10}", end="")
        for val in contingency[k]:
            print(f"{val:>7}", end="")
        print(f"{contingency[k].sum():>8}")
    print("-" * 55)
    print(f"{'Total':<10}", end="")
    for val in contingency.sum(axis=0):
        print(f"{val:>7}", end="")
    print(f"{contingency.sum():>8}")

    chi2, p_value, dof, expected = chi2_contingency(contingency)
    print(f"\nChi-square statistic: {chi2:.3f}")
    print(f"Degrees of freedom:   {dof}")
    print(f"p-value:              {p_value:.4f}")

    min_expected = expected.min()
    low_expected = (expected < 5).sum()
    print(f"Minimum expected count: {min_expected:.2f}")
    print(f"Cells with expected < 5: {low_expected}/{expected.size}")

    if p_value < 0.05:
        print("-> REJECT H0 at alpha=0.05. Monthly distributions differ across nodes.")
    else:
        print(
            "-> FAIL TO REJECT H0. No significant difference in monthly distributions."
        )

    # Pairwise comparisons
    pairs = list(combinations(range(n_nodes), 2))
    n_comparisons = len(pairs)
    alpha_corrected = 0.05 / n_comparisons
    print(f"\nPairwise chi-square tests (Bonferroni alpha={alpha_corrected:.4f}):")
    for idx1, idx2 in pairs:
        pair_table = contingency[[idx1, idx2], :]
        chi2_pair, p_pair, _, _ = chi2_contingency(pair_table)
        sig = "***" if p_pair < alpha_corrected else ""
        print(
            f"  {node_labels_chi[idx1]} vs {node_labels_chi[idx2]}: "
            f"chi2={chi2_pair:.2f}, p={p_pair:.4f} {sig}"
        )

    # ── Monthly relative likelihood heatmap ───────────────────────────────────
    print("\nPlotting monthly relative likelihood heatmap ...")
    node_totals = np.array(
        [month_counts[(i, j)].sum() for i in range(xdim) for j in range(ydim)]
    )
    total_events = node_totals.sum()
    P_node = node_totals / total_events
    all_month_counts = np.bincount(months, minlength=13)[1:]
    month_idx = np.arange(4, 10)

    heatmap = np.zeros((n_nodes, len(month_idx)))
    k = 0
    for i in range(xdim):
        for j in range(ydim):
            counts_mo = month_counts[(i, j)][month_idx]
            totals = all_month_counts[month_idx]
            heatmap[k, :] = counts_mo / totals
            k += 1

    relative_heatmap = np.zeros_like(heatmap)
    for k in range(n_nodes):
        relative_heatmap[k, :] = heatmap[k, :] / P_node[k]

    fig, ax = plt.subplots(figsize=(6, 3.7), dpi=600)
    im = ax.imshow(relative_heatmap, aspect="auto", cmap="RdBu_r", vmin=0, vmax=2)
    ax.set_xticks(np.arange(len(month_idx)))
    ax.set_xticklabels(["May", "Jun", "Jul", "Aug", "Sep", "Oct"], fontsize=7)
    ax.set_yticks(np.arange(n_nodes))
    ax.set_yticklabels(node_labels_chi, fontsize=6)
    ax.set_xlabel("Month")
    ax.set_ylabel("SOM Node")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Relative Likelihood", fontsize=7)
    plt.title(
        "Monthly Relative Likelihood of SOM Nodes\n"
        "(Normalized by Seasonal Event Frequency)",
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(
        f"{fig_dir}/Z500_and_{_lbl}_som_monthly_relative_heatmap.png",
        bbox_inches="tight",
    )
    plt.close()

    print(f"\nAll plots saved to {fig_dir}/")


if __name__ == "__main__":
    main()
