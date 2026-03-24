"""
All-Days SOM Diagnostic Plots for NYC Flash Flood Analysis
By: Ty Janoski

Generates diagnostic plots for a trained all-days SOM: U-matrix, Sammon map,
node weights, composites, flash-flood risk heatmap, cutoff-low frequency,
QE ranking, residual analysis, and individual FF-day event maps.

Usage:
    python -m som_analysis.plot_alldays_som --moisture-var IVT
    python -m som_analysis.plot_alldays_som --moisture-var thetae --skip-indiv
    python -m som_analysis.plot_alldays_som --moisture-var IVT --xdim 5 --ydim 4
"""

import argparse
import os

import cartopy.crs as ccrs
import cmweather  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from .config import (
    ALLDAYS_MOISTURE_VARS,
    MOISTURE_CONFIGS,
    SOM_INTERMEDIATE_PATH,
    get_alldays_paths,
    setup_plotting,
)
from .helpers import (
    add_map_features,
    compute_composites,
    create_som_figure,
    get_node_indices,
    has_cutoff_low,
    load_moist_var,
    node_label,
    plot_node_events,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate diagnostic plots for a trained all-days SOM."
    )
    parser.add_argument(
        "--moisture-var",
        required=True,
        choices=ALLDAYS_MOISTURE_VARS,
        help="Moisture variable (IVT or thetae).",
    )
    parser.add_argument(
        "--moisture-weight",
        type=float,
        default=1,
        help="Moisture weight used during training (default: 1).",
    )
    parser.add_argument(
        "--xdim",
        type=int,
        default=5,
        help="SOM columns (must match trained SOM, default: 5).",
    )
    parser.add_argument(
        "--ydim",
        type=int,
        default=4,
        help="SOM rows (must match trained SOM, default: 4).",
    )
    parser.add_argument(
        "--skip-indiv",
        action="store_true",
        help="Skip individual FF-day event plots (slow).",
    )
    parser.add_argument(
        "--n-qe-plot",
        type=int,
        default=15,
        help="Number of events to show in top/bottom QE plots (default: 15).",
    )
    parser.add_argument(
        "--snapshot-hour",
        type=int,
        default=None,
        help="Use hourly snapshot files (e.g. 20 for 2000 UTC) instead of daily means.",
    )
    return parser.parse_args()


def _node_title(i, j, counts, totals, risk):
    """Build a node panel title with FF risk."""
    n = counts[i, j]
    tot = totals[i, j]
    r = risk[i, j]
    if np.isnan(r):
        return f"{node_label(i, j)}  FF=0/{tot}"
    return f"{node_label(i, j)}  FF={n}/{tot} ({r * 100:.1f}\\%)"


def main():
    args = parse_args()
    setup_plotting()

    cfg = MOISTURE_CONFIGS[args.moisture_var]
    paths = get_alldays_paths(
        args.moisture_var, args.xdim, args.ydim, snapshot_hour=args.snapshot_hour
    )
    fig_dir = paths["fig_dir"]
    _lbl = paths["file_label"]
    moist_time_dim = cfg["time_dim"]
    moist_label = cfg["label"]
    moist_label_short = cfg["label_short"]
    moist_units_raw = cfg["units_raw"]
    levels_moist_weights = cfg["levels_weights"]
    levels_moist_anom = cfg["levels_anom"]
    levels_moist_raw = cfg["levels_raw"]
    levels_moist_indiv = cfg["levels_indiv"]
    cmap_moist_raw = cfg["cmap_raw"]

    xdim, ydim = args.xdim, args.ydim
    pfx = cfg["file_prefix"]
    var_name = cfg["var_name"]

    os.makedirs(f"{fig_dir}/indiv-nodes", exist_ok=True)

    # ── Load cached SOM results ───────────────────────────────────────────────
    cache_path = os.path.join(fig_dir, ".cache", "som_results.npz")
    print(f"Loading SOM results from {cache_path} ...")
    cached = np.load(cache_path)
    weights = cached["weights"]
    z500_nodes = cached["z500_nodes"]  # shape (xdim, ydim, n_lat, n_lon)
    moist_nodes = cached["moist_nodes"]
    bmus = cached["bmus"]  # shape (n_days, 2)
    u_matrix = cached["u_matrix"]
    hit_map = cached["hit_map"]
    coords = cached["coords"]
    event_indices = cached["event_indices"]
    counts = cached["counts"]  # shape (xdim, ydim) — FF days per node
    totals = cached["totals"]  # shape (xdim, ydim) — all days per node
    risk = cached["risk"]  # shape (xdim, ydim)
    lat = cached["lat"]
    lon = cached["lon"]

    n_ff = len(event_indices)

    # ── Load data files ───────────────────────────────────────────────────────
    if args.snapshot_hour is not None:
        alldays_suffix = f"_snapshot_{args.snapshot_hour:02d}utc"
        print(f"Loading snapshot data files (hour={args.snapshot_hour:02d}UTC) ...")
    else:
        alldays_suffix = "_daily"
        print("Loading daily data files ...")

    moist_norm_weighted_daily = load_moist_var(
        f"{SOM_INTERMEDIATE_PATH}{pfx}_norm_weighted{alldays_suffix}.nc", var_name
    )
    moist_norm_daily = load_moist_var(
        f"{SOM_INTERMEDIATE_PATH}{pfx}_norm{alldays_suffix}.nc", var_name
    )
    moist_daily = load_moist_var(
        f"{SOM_INTERMEDIATE_PATH}{pfx}{alldays_suffix}.nc", var_name
    )
    z500_norm_weighted_daily = xr.load_dataarray(
        f"{SOM_INTERMEDIATE_PATH}era5_Z500_norm_weighted{alldays_suffix}.nc"
    )
    z500_norm_daily = xr.load_dataarray(
        f"{SOM_INTERMEDIATE_PATH}era5_Z500_norm{alldays_suffix}.nc"
    )
    z500_daily = xr.load_dataarray(
        f"{SOM_INTERMEDIATE_PATH}era5_Z500{alldays_suffix}.nc"
    )

    # ── Compute QE for FF days ────────────────────────────────────────────────
    # Reconstruct feature matrix X for QE calculation (matches training)
    print("Computing quantization errors ...")
    z500_lat_dim = "latitude" if "latitude" in z500_norm_weighted_daily.dims else "lat"
    z500_lon_dim = (
        "longitude" if "longitude" in z500_norm_weighted_daily.dims else "lon"
    )
    z500_time_dim = (
        "valid_time" if "valid_time" in z500_norm_weighted_daily.dims else "time"
    )
    z500_flat = z500_norm_weighted_daily.stack(
        features=[z500_lat_dim, z500_lon_dim]
    ).values
    moist_flat = moist_norm_weighted_daily.stack(
        features=[cfg["lat_dim"], cfg["lon_dim"]]
    ).values
    X = np.concatenate((z500_flat, moist_flat * args.moisture_weight), axis=1)

    qe_ff = np.array(
        [
            np.linalg.norm(X[k] - weights[bmus[k, 0] * ydim + bmus[k, 1]])
            for k in event_indices
        ]
    )
    ff_timestamps = pd.to_datetime(z500_norm_daily[z500_time_dim].values[event_indices])
    ff_node_i = bmus[event_indices, 0]
    ff_node_j = bmus[event_indices, 1]

    # Save QE ranking CSV
    qe_df = pd.DataFrame(
        {
            "timestamp": ff_timestamps,
            "node_i": ff_node_i,
            "node_j": ff_node_j,
            "node": [
                node_label(int(i), int(j))
                for i, j in zip(ff_node_i, ff_node_j, strict=False)
            ],
            "qe": qe_ff,
        }
    ).sort_values("qe")
    qe_df.to_csv(os.path.join(fig_dir, "qe_ranking.csv"), index=False)
    print(f"  Saved QE ranking to {fig_dir}/qe_ranking.csv")

    # ── U-matrix and hit map ──────────────────────────────────────────────────
    print("Plotting U-matrix and hit map ...")
    fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(8, 4), dpi=600)

    im0 = axes[0].imshow(u_matrix, cmap="viridis", origin="lower")
    axes[0].set_title("U-Matrix (Mean Inter-Node Distance)", fontsize=7)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, shrink=0.7)

    im1 = axes[1].imshow(hit_map, cmap="plasma", origin="lower")
    axes[1].set_title("Hit Map (Days per Node)", fontsize=7)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, shrink=0.7)

    for ax in axes:
        ax.set_xticks(np.arange(xdim))
        ax.set_yticks(np.arange(ydim))
        ax.set_xlabel("X-index", fontsize=6)
        ax.set_ylabel("Y-index", fontsize=6)

    plt.savefig(f"{fig_dir}/umatrix_hitmap.png")
    plt.close()

    # ── Sammon / MDS map ──────────────────────────────────────────────────────
    print("Plotting Sammon/MDS map ...")
    U_flat = u_matrix.T.reshape(-1)
    hits_flat = hit_map.T.reshape(-1)
    hits_scaled = 30 + 250 * (hits_flat / hits_flat.max())

    plt.figure(figsize=(9, 7))
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
            x, y, node_label(ix, iy), fontsize=7, ha="center", va="center", zorder=5
        )

    plt.title(
        "Sammon / MDS Distortion Grid\nU-Matrix (Color) \\& Node Frequency (Size)"
    )
    plt.axis("off")
    plt.colorbar(sc, label="U-Matrix (Avg. Neighbor Distance)")
    plt.savefig(f"{fig_dir}/sammon_map.png", bbox_inches="tight")
    plt.close()

    # ── Node weight patterns ──────────────────────────────────────────────────
    print("Plotting node weight patterns ...")
    levels_Z = np.arange(-1.4, 1.41, 0.2)
    proj = ccrs.PlateCarree()

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
                transform=proj,
            )
            cn = ax.contour(
                lon,
                lat,
                z500_nodes[i, j],
                colors="black",
                linewidths=0.5,
                levels=levels_Z,
                transform=proj,
            )
            ax.clabel(cn, inline=True, fontsize=4, fmt="%.1f")
            add_map_features(ax)
            ax.set_title(_node_title(i, j, counts, totals, risk), fontsize=5)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label(f"Standardized {moist_label} Anomaly (Shaded)", fontsize=6)
    plt.suptitle(
        f"All-Days SOM: Node Weight Patterns\n"
        f"Z500 (contoured) + {moist_label_short} (shaded)",
        fontsize=8,
    )
    plt.savefig(
        f"{fig_dir}/combined_node_weights_{_lbl}_shaded.png", bbox_inches="tight"
    )
    plt.close()

    # ── Anomaly composites (all days) ─────────────────────────────────────────
    print("Plotting anomaly composites ...")
    z500_anom_comp, comp_counts = compute_composites(
        z500_norm_daily, bmus, xdim, ydim, time_dim=z500_time_dim
    )
    moist_anom_comp, _ = compute_composites(
        moist_norm_daily, bmus, xdim, ydim, time_dim=moist_time_dim
    )
    levels_Z_anom = np.arange(-2.0, 2.1, 0.25)

    fig, axes = create_som_figure(xdim, ydim)
    for i in range(xdim):
        for j in range(ydim):
            ax = axes[j, i]
            im = ax.contourf(
                lon,
                lat,
                moist_anom_comp[i, j],
                cmap="balance",
                levels=levels_moist_anom,
                transform=proj,
                extend="both",
            )
            ax.contour(
                lon,
                lat,
                z500_anom_comp[i, j],
                colors="black",
                linewidths=0.5,
                levels=levels_Z_anom,
                transform=proj,
            )
            add_map_features(ax)
            ax.set_title(_node_title(i, j, counts, totals, risk), fontsize=5)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label("Standardized Anomaly", fontsize=6)
    plt.suptitle(
        f"All-Days SOM Anomaly Composites (All Days): "
        f"Z500 (contoured) + {moist_label_short} (shaded)",
        fontsize=8,
        y=1.04,
    )
    plt.savefig(
        f"{fig_dir}/Z500_and_{_lbl}_SOM_anomaly_composites.png",
        bbox_inches="tight",
    )
    plt.close()

    # ── Raw composites — all days ─────────────────────────────────────────────
    print("Plotting raw composites (all days) ...")
    z500_raw_comp, _ = compute_composites(
        z500_daily, bmus, xdim, ydim, time_dim=z500_time_dim
    )
    moist_raw_comp, _ = compute_composites(
        moist_daily, bmus, xdim, ydim, time_dim=moist_time_dim
    )
    levels_Z_raw = range(552, 595, 3)

    fig, axes = create_som_figure(xdim, ydim)
    for i in range(xdim):
        for j in range(ydim):
            ax = axes[j, i]
            im = ax.contourf(
                lon,
                lat,
                moist_raw_comp[i, j],
                cmap=cmap_moist_raw,
                levels=levels_moist_raw,
                transform=proj,
                extend="max",
            )
            cn = ax.contour(
                lon,
                lat,
                z500_raw_comp[i, j] / 98.1,
                colors="black",
                linewidths=0.5,
                levels=levels_Z_raw,
                transform=proj,
            )
            ax.clabel(cn, inline=True, fontsize=4, fmt="%.0f")
            add_map_features(ax)
            ax.set_title(_node_title(i, j, counts, totals, risk), fontsize=5)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label(f"{moist_label_short} ({moist_units_raw})", fontsize=6)
    plt.suptitle(
        f"All-Days SOM Composites (All Days): "
        f"{moist_label_short} (shaded) + Z500 (contoured)",
        fontsize=8,
        y=1.04,
    )
    plt.savefig(
        f"{fig_dir}/Z500_and_{_lbl}_SOM_raw_composites_alldays.png",
        bbox_inches="tight",
    )
    plt.close()

    # ── Raw composites — FF days only ─────────────────────────────────────────
    # Create a fake BMU array indexed 0..n_ff-1 so compute_composites works
    # on the FF-day subset
    print("Plotting raw composites (FF days only) ...")
    ff_bmus = bmus[event_indices]

    # Subset data arrays to FF days
    z500_ff = z500_daily.isel(
        {z500_time_dim: xr.DataArray(event_indices, dims="valid_time")}
    )
    moist_ff = moist_daily.isel(
        {moist_time_dim: xr.DataArray(event_indices, dims="valid_time")}
    )

    # Assign a simple integer index dimension for compute_composites
    z500_ff = z500_ff.assign_coords({z500_time_dim: np.arange(n_ff)})
    moist_ff = moist_ff.assign_coords(valid_time=np.arange(n_ff))

    z500_ff_comp, ff_node_counts = compute_composites(
        z500_ff, ff_bmus, xdim, ydim, time_dim=z500_time_dim
    )
    moist_ff_comp, _ = compute_composites(
        moist_ff, ff_bmus, xdim, ydim, time_dim="valid_time"
    )

    fig, axes = create_som_figure(xdim, ydim)
    for i in range(xdim):
        for j in range(ydim):
            ax = axes[j, i]
            im = ax.contourf(
                lon,
                lat,
                moist_ff_comp[i, j],
                cmap=cmap_moist_raw,
                levels=levels_moist_raw,
                transform=proj,
                extend="max",
            )
            cn = ax.contour(
                lon,
                lat,
                z500_ff_comp[i, j] / 98.1,
                colors="black",
                linewidths=0.5,
                levels=levels_Z_raw,
                transform=proj,
            )
            ax.clabel(cn, inline=True, fontsize=4, fmt="%.0f")
            add_map_features(ax)
            ax.set_title(_node_title(i, j, counts, totals, risk), fontsize=5)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label(f"{moist_label_short} ({moist_units_raw})", fontsize=6)
    plt.suptitle(
        f"All-Days SOM Composites (FF Days Only): "
        f"{moist_label_short} (shaded) + Z500 (contoured)",
        fontsize=8,
        y=1.04,
    )
    plt.savefig(
        f"{fig_dir}/Z500_and_{_lbl}_SOM_raw_composites_ff.png",
        bbox_inches="tight",
    )
    plt.close()

    # ── Cross-variable composites (all days) ──────────────────────────────────
    other_vars = [v for v in ALLDAYS_MOISTURE_VARS if v != args.moisture_var]
    if other_vars:
        print("Plotting cross-variable composites ...")
    for other_var in other_vars:
        other_cfg = MOISTURE_CONFIGS[other_var]
        other_file_label = other_cfg["file_label"]
        other_pfx = other_cfg["file_prefix"]
        other_var_name = other_cfg["var_name"]
        other_time_dim = other_cfg["time_dim"]

        other_daily = load_moist_var(
            f"{SOM_INTERMEDIATE_PATH}{other_pfx}{alldays_suffix}.nc", other_var_name
        )
        other_comp, _ = compute_composites(
            other_daily, bmus, xdim, ydim, time_dim=other_time_dim
        )

        fig, axes = create_som_figure(xdim, ydim)
        for i in range(xdim):
            for j in range(ydim):
                ax = axes[j, i]
                im = ax.contourf(
                    lon,
                    lat,
                    other_comp[i, j],
                    cmap=other_cfg["cmap_raw"],
                    levels=other_cfg["levels_raw"],
                    transform=proj,
                    extend="max",
                )
                cn = ax.contour(
                    lon,
                    lat,
                    z500_raw_comp[i, j] / 98.1,
                    colors="black",
                    linewidths=0.5,
                    levels=levels_Z_raw,
                    transform=proj,
                )
                ax.clabel(cn, inline=True, fontsize=4, fmt="%.0f")
                add_map_features(ax)
                ax.set_title(_node_title(i, j, counts, totals, risk), fontsize=5)

        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
        cbar.set_label(
            f"{other_cfg['label_short']} ({other_cfg['units_raw']})", fontsize=6
        )
        plt.suptitle(
            f"All-Days SOM Composites (All Days): "
            f"{other_cfg['label_short']} (shaded) + Z500 (contoured)\n"
            f"(SOM trained on Z500 + {moist_label_short})",
            fontsize=8,
            y=1.04,
        )
        out_fname = f"Z500_and_{_lbl}_SOM_composite_mean_{other_file_label}_shaded.png"
        plt.savefig(f"{fig_dir}/{out_fname}", bbox_inches="tight")
        plt.close()
        print(f"  Saved {out_fname}")

    # ── Trained-variable mean composite (all days) ────────────────────────────
    # Always generate composite_mean_<var>_shaded for the trained variable itself.
    print(f"Plotting composite_mean_{_lbl}_shaded ...")
    fig, axes = create_som_figure(xdim, ydim)
    for i in range(xdim):
        for j in range(ydim):
            ax = axes[j, i]
            im = ax.contourf(
                lon,
                lat,
                moist_raw_comp[i, j],
                cmap=cmap_moist_raw,
                levels=levels_moist_raw,
                transform=proj,
                extend="max",
            )
            cn = ax.contour(
                lon,
                lat,
                z500_raw_comp[i, j] / 98.1,
                colors="black",
                linewidths=0.5,
                levels=levels_Z_raw,
                transform=proj,
            )
            ax.clabel(cn, inline=True, fontsize=4, fmt="%.0f")
            add_map_features(ax)
            ax.set_title(_node_title(i, j, counts, totals, risk), fontsize=5)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label(f"{moist_label_short} ({moist_units_raw})", fontsize=6)
    plt.suptitle(
        f"All-Days SOM Composites (All Days): "
        f"{moist_label_short} (shaded) + Z500 (contoured)",
        fontsize=8,
        y=1.04,
    )
    out_fname = f"Z500_and_{_lbl}_SOM_composite_mean_{_lbl}_shaded.png"
    plt.savefig(f"{fig_dir}/{out_fname}", bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_fname}")

    # ── FF risk heatmap ───────────────────────────────────────────────────────
    print("Plotting FF risk heatmap ...")
    fig, ax = plt.subplots(
        figsize=(0.8 * xdim + 1, 0.9 * ydim + 1), dpi=300, constrained_layout=True
    )
    im = ax.imshow(
        risk.T,
        cmap="YlOrRd",
        vmin=0,
        vmax=risk[~np.isnan(risk)].max(),
        origin="lower",
    )
    for i in range(xdim):
        for j in range(ydim):
            r = risk[i, j]
            txt = f"{r:.1%}" if not np.isnan(r) else "N/A"
            ax.text(
                i,
                j,
                f"{counts[i, j]}/{totals[i, j]}\n{txt}",
                ha="center",
                va="center",
                fontsize=6,
                color=(
                    "white"
                    if (not np.isnan(r) and r > 0.6 * risk[~np.isnan(risk)].max())
                    else "black"
                ),
            )
    ax.set_xticks(np.arange(xdim))
    ax.set_yticks(np.arange(ydim))
    ax.set_xlabel("X-index")
    ax.set_ylabel("Y-index")
    ax.set_title("Flash Flood Risk by Node (FF days / total days)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("FF Risk", fontsize=7)
    plt.savefig(f"{fig_dir}/ff_risk_heatmap.png", bbox_inches="tight")
    plt.close()

    # ── Cutoff-low frequency heatmap (FF days per node only) ─────────────────
    print("Computing cutoff-low frequency per node (FF days only) ...")
    cutoff_count = np.zeros((xdim, ydim), dtype=int)
    cutoff_freq = np.zeros((xdim, ydim))
    # Z500 in dam for cutoff-low threshold (540 dam ≈ 5400 m)
    for k in event_indices:
        ni, nj = bmus[k]
        if has_cutoff_low(z500_daily.isel({z500_time_dim: int(k)}).values / 98.1):
            cutoff_count[ni, nj] += 1
    for i in range(xdim):
        for j in range(ydim):
            if counts[i, j] > 0:
                cutoff_freq[i, j] = cutoff_count[i, j] / counts[i, j]

    fig, ax = plt.subplots(
        figsize=(0.8 * xdim + 1, 0.9 * ydim + 1), dpi=300, constrained_layout=True
    )
    im = ax.imshow(cutoff_freq.T * 100, cmap="YlOrRd", vmin=0, vmax=100, origin="lower")
    for i in range(xdim):
        for j in range(ydim):
            frac = cutoff_freq[i, j]
            cnt = cutoff_count[i, j]
            n = counts[i, j]
            txt = f"{frac:.0%}\n({cnt}/{n})" if n > 0 else "n=0"
            ax.text(
                i,
                j,
                txt,
                ha="center",
                va="center",
                fontsize=6,
                color="white" if frac > 0.5 else "black",
            )
    ax.set_xticks(np.arange(xdim))
    ax.set_yticks(np.arange(ydim))
    ax.set_xlabel("X-index")
    ax.set_ylabel("Y-index")
    ax.set_title("Cutoff Low Frequency by Node\n(FF days only, Z500 $<$ 540 dam)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(r"Fraction of FF Days (\%)", fontsize=7)
    plt.savefig(f"{fig_dir}/cutoff_low_freq_heatmap.png", bbox_inches="tight")
    plt.close()

    # ── QE top/bottom event maps ──────────────────────────────────────────────
    n_qe = min(args.n_qe_plot, n_ff)
    print(f"Plotting top/bottom {n_qe} QE event maps ...")

    for label, selection in [("best", qe_df.head(n_qe)), ("worst", qe_df.tail(n_qe))]:
        # Get original indices into event_indices
        sel_timestamps = pd.to_datetime(selection["timestamp"].values)
        all_timestamps = pd.to_datetime(z500_norm_daily[z500_time_dim].values)
        sel_global_idx = np.array(
            [np.where(all_timestamps == ts)[0][0] for ts in sel_timestamps]
        )
        n_sel = len(sel_global_idx)
        cols = 5
        rows = int(np.ceil(n_sel / cols))

        fig, axes_grid = plt.subplots(
            rows,
            cols,
            figsize=(3 * cols, 2.5 * rows),
            subplot_kw={"projection": proj},
            layout="constrained",
            dpi=200,
        )

        for k, ax in enumerate(axes_grid.flat):
            if k < n_sel:
                gidx = sel_global_idx[k]
                ts = sel_timestamps[k]
                ni, nj = (
                    int(selection["node_i"].iloc[k]),
                    int(selection["node_j"].iloc[k]),
                )
                field_moist = moist_daily.isel({moist_time_dim: gidx})
                field_z500 = z500_daily.isel({z500_time_dim: gidx})

                im = ax.contourf(
                    lon,
                    lat,
                    field_moist.values,
                    cmap=cmap_moist_raw,
                    levels=levels_moist_indiv,
                    transform=proj,
                    extend="max",
                )
                cn = ax.contour(
                    lon,
                    lat,
                    field_z500.values / 98.1,
                    colors="black",
                    linewidths=0.5,
                    levels=levels_Z_raw,
                    transform=proj,
                )
                ax.clabel(cn, inline=True, fontsize=4, fmt="%.0f")
                add_map_features(ax)
                ax.set_title(
                    f"{str(ts)[:10]}  ({ni},{nj})\nQE={selection['qe'].iloc[k]:.2f}",
                    fontsize=5,
                )
            else:
                ax.axis("off")

        fig.colorbar(im, ax=axes_grid.ravel().tolist(), shrink=0.6, pad=0.02)
        qe_label = "Lowest" if label == "best" else "Highest"
        fig.suptitle(
            f"FF Days — {qe_label} QE (Top {n_sel}): "
            f"{moist_label_short} (shaded) + Z500 (contoured)",
            fontsize=8,
            y=1.02,
        )
        plt.savefig(f"{fig_dir}/qe_{label}_{n_sel}.png", bbox_inches="tight")
        plt.close()

    # ── Residual composite per node (FF days) ────────────────────────────────
    # For each node: mean(obs_norm - node_centroid) over FF days in that node.
    # Also compute mean spatial correlation r(obs, centroid) per node.
    print("Plotting residual composites (FF days, per node) ...")
    z500_residuals = np.full((xdim, ydim, len(lat), len(lon)), np.nan)
    moist_residuals = np.full((xdim, ydim, len(lat), len(lon)), np.nan)
    mean_corr_per_node = np.full((xdim, ydim), np.nan)

    for i in range(xdim):
        for j in range(ydim):
            idx_ff_node = np.intersect1d(get_node_indices(bmus, i, j), event_indices)
            if len(idx_ff_node) == 0:
                continue
            node_wt = weights[i * ydim + j]

            z500_obs = np.stack(
                [
                    z500_norm_daily.isel({z500_time_dim: int(k)}).values
                    for k in idx_ff_node
                ]
            )
            moist_obs = np.stack(
                [
                    moist_norm_daily.isel({moist_time_dim: int(k)}).values
                    for k in idx_ff_node
                ]
            )

            z500_residuals[i, j] = np.mean(z500_obs - z500_nodes[i, j], axis=0)
            moist_residuals[i, j] = np.mean(moist_obs - moist_nodes[i, j], axis=0)

            corr_vals = [np.corrcoef(X[k], node_wt)[0, 1] for k in idx_ff_node]
            mean_corr_per_node[i, j] = np.mean(corr_vals)

    levels_resid = np.arange(-2.0, 2.05, 0.2)
    fig, axes = create_som_figure(xdim, ydim)
    for i in range(xdim):
        for j in range(ydim):
            ax = axes[j, i]
            if np.isnan(z500_residuals[i, j]).all():
                ax.set_title(f"{node_label(i, j)}  n=0", fontsize=5)
                add_map_features(ax)
                continue
            im = ax.contourf(
                lon,
                lat,
                moist_residuals[i, j],
                cmap="balance",
                levels=levels_resid,
                transform=proj,
                extend="both",
            )
            ax.contour(
                lon,
                lat,
                z500_residuals[i, j],
                colors="black",
                linewidths=0.5,
                levels=levels_resid,
                transform=proj,
            )
            add_map_features(ax)
            ax.set_title(
                f"{node_label(i, j)} n={counts[i, j]} r={mean_corr_per_node[i, j]:.2f}",
                fontsize=5,
            )

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label("Standardized Residual", fontsize=6)
    plt.suptitle(
        f"Mean Residuals (FF days): Z500 (contoured) + {moist_label_short} (shaded)",
        fontsize=8,
        y=1.04,
    )
    plt.savefig(f"{fig_dir}/residuals_ff.png", bbox_inches="tight")
    plt.close()

    # ── Individual FF-day event maps ──────────────────────────────────────────
    if not args.skip_indiv:
        print("Plotting individual FF-day event maps (this may take a while) ...")
        # Build a DataArray containing only FF-day slices with a simple integer index
        moist_ff_indexed = moist_daily.isel(
            {moist_time_dim: xr.DataArray(event_indices, dims="event")}
        ).assign_coords(event=np.arange(n_ff))
        z500_ff_indexed = z500_daily.isel(
            {z500_time_dim: xr.DataArray(event_indices, dims="event")}
        ).assign_coords(event=np.arange(n_ff))

        # Drop leftover scalar coordinates that would conflict with the rename
        if moist_time_dim in moist_ff_indexed.coords:
            moist_ff_indexed = moist_ff_indexed.drop_vars(moist_time_dim)
        if z500_time_dim in z500_ff_indexed.coords:
            z500_ff_indexed = z500_ff_indexed.drop_vars(z500_time_dim)

        # Rename event dim to the expected time_dim for plot_node_events
        moist_ff_indexed = moist_ff_indexed.rename({"event": moist_time_dim})
        z500_ff_indexed = z500_ff_indexed.rename({"event": z500_time_dim})

        # Assign original timestamps as coordinate so titles show real dates
        moist_ff_indexed[moist_time_dim] = ff_timestamps.values
        z500_ff_indexed[z500_time_dim] = ff_timestamps.values

        plot_node_events(
            moist_ff_indexed,
            ff_bmus,
            xdim,
            ydim,
            lon,
            lat,
            time_dim=moist_time_dim,
            levels=levels_moist_indiv,
            cmap=cmap_moist_raw,
            save_pattern=f"{fig_dir}/indiv-nodes/node_{{i}}_{{j}}_ff.png",
            cbar_label=f"{moist_label_short} ({moist_units_raw})",
            z500_data=z500_ff_indexed,
            z500_levels=levels_Z_raw,
            z500_scale=1 / 98.1,
            z500_time_dim=z500_time_dim,
        )
    else:
        print("Skipping individual FF-day event plots (--skip-indiv).")

    print(f"\nAll plots saved to {fig_dir}/")


if __name__ == "__main__":
    main()
