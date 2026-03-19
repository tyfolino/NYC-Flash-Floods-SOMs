"""
Node Statistics for NYC Flash Flood SOM Analysis
By: Ty Janoski

Per-node precipitation analysis (ASOS, StageIV, ERA5) and tropical cyclone
association analysis. Reads BMU assignments produced by train_som.py.

Usage:
    python -m som_analysis.node_statistics --moisture-var thetae
    python -m som_analysis.node_statistics --moisture-var IVT --moisture-weight 2
"""

import argparse
import os
from itertools import combinations

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import xarray as xr
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
from metpy.plots import ctables
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu

from .config import (
    DATA_DIR,
    ERA5_TP_DIR,
    IBTRACS_PATH,
    MOISTURE_CONFIGS,
    STAGEIV_NC,
    STATION_COORDS,
    STATION_NPY_FILES,
    get_paths,
    setup_plotting,
)
from .helpers import node_label


def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-node precipitation and TC association analysis."
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
        "--stageiv-agg",
        choices=["max", "spatial-median"],
        default="max",
        help=(
            "Spatial aggregation for StageIV precipitation. "
            "'max' (default): single highest value across all NYC cells. "
            "'spatial-median': median of per-cell time-maxes."
        ),
    )
    parser.add_argument(
        "--daily",
        action="store_true",
        help="Use daily-based BMU assignments (must match --daily used in train_som).",
    )
    return parser.parse_args()


# ── ASOS Precipitation ───────────────────────────────────────────────────────


def load_asos_precip():
    """Load ASOS precipitation data from .npy files."""
    precip_dfs = {}
    for name, fpath in STATION_NPY_FILES.items():
        arr = np.load(fpath, allow_pickle=True)
        df = pd.DataFrame(arr, columns=["precip", "time"])
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        df["precip"] = pd.to_numeric(df["precip"], errors="coerce")
        precip_dfs[name] = df
    return precip_dfs


def compute_asos_max_precip(bmu_df, precip_dfs, window_hours=6):
    """Compute max hourly ASOS precip within a window for each event."""
    max_precip = []
    for _, row in bmu_df.iterrows():
        t = row["timestamp_local"]
        lo = t - pd.Timedelta(hours=window_hours)
        hi = t + pd.Timedelta(hours=window_hours)
        site_maxes = [df.loc[lo:hi, "precip"].max() for df in precip_dfs.values()]
        valid = [v for v in site_maxes if not np.isnan(v)]
        max_precip.append(max(valid) if valid else 0.0)
    return max_precip


def plot_precip_histograms(
    bmu_df,
    col,
    xdim,
    ydim,
    fig_dir,
    _lbl,
    color,
    source_label,
    bins=None,
    xlim=None,
    xlabel="Max Hourly Precip (in)",
):
    """Plot per-node precipitation histograms."""
    if bins is None:
        bins = np.arange(0, 3.76, 0.25)
    if xlim is None:
        xlim = (0, 3.75)

    fig, axes = plt.subplots(
        ydim, xdim, figsize=(6, 4), constrained_layout=True, dpi=600
    )

    for i in range(xdim):
        for j in range(ydim):
            ax = axes[j, i]
            node_data = bmu_df[(bmu_df["node_i"] == i) & (bmu_df["node_j"] == j)][
                col
            ].dropna()

            ax.hist(
                node_data,
                bins=bins,
                color=color,
                alpha=0.9,
                edgecolor="white",
                linewidth=0.5,
            )
            n = len(node_data)
            if n > 0:
                median = node_data.median()
                ax.axvline(
                    median,
                    color="red",
                    linestyle="--",
                    linewidth=1,
                    label=f'Median: {median:.2f}"',
                )
                ax.legend(fontsize=4, loc="upper right")

            ax.set_title(f"{source_label}  {node_label(i, j)}  N={n}", fontsize=6)
            ax.set_xlim(*xlim)
            ax.set_ylim(0, 12)
            ax.set_yticks(np.arange(0, 13, 2))
            ax.tick_params(axis="both", labelsize=5)
            ax.grid(True, linewidth=0.3, alpha=0.5, axis="y")
            if j == ydim - 1:
                ax.set_xlabel(xlabel, fontsize=5)
            if i == 0:
                ax.set_ylabel("Count", fontsize=5)

    plt.suptitle(
        rf"Max Hourly {source_label} Precip ($\pm$6 hr window) by SOM Node",
        fontsize=8,
        y=1.02,
    )
    fname = f"Z500_and_{_lbl}_som_max_precip_histograms_{source_label.lower().replace(' ', '_')}.png"
    plt.savefig(f"{fig_dir}/{fname}", bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


# ── StageIV Precipitation ────────────────────────────────────────────────────


def compute_stageiv_max_precip(bmu_df, window_hours=6, agg="max"):
    """Compute max hourly StageIV precip over NYC for each event."""
    MM_TO_IN = 1.0 / 25.4
    nyc_lat_min, nyc_lat_max = 40.5, 40.9
    nyc_lon_min, nyc_lon_max = -74.3, -73.7

    ds_s4 = xr.open_dataset(STAGEIV_NC, chunks={"time": 168})
    s4_times = pd.DatetimeIndex(ds_s4["time"].values)
    lat2d = ds_s4["latitude"].values
    lon2d = ds_s4["longitude"].values

    spatial_mask = (
        (lat2d >= nyc_lat_min)
        & (lat2d <= nyc_lat_max)
        & (lon2d >= nyc_lon_min)
        & (lon2d <= nyc_lon_max)
    )

    print(f"StageIV time range : {s4_times.min()} -- {s4_times.max()}")
    print(
        f"NYC mask           : {int(spatial_mask.sum())} cells "
        f"({nyc_lat_min}--{nyc_lat_max} N, {nyc_lon_min}--{nyc_lon_max} W)"
    )

    # Collect needed time indices
    all_needed = set()
    event_windows = []
    for _, row in bmu_df.iterrows():
        t0 = row["timestamp"]
        mask = (s4_times >= t0 - pd.Timedelta(hours=window_hours)) & (
            s4_times <= t0 + pd.Timedelta(hours=window_hours)
        )
        idxs = np.where(mask)[0].tolist()
        event_windows.append(idxs)
        all_needed.update(idxs)

    needed_idxs = sorted(all_needed)
    print(f"Loading {len(needed_idxs)} unique StageIV time steps ...")

    precip_raw = ds_s4["precipitation"].isel(time=needed_idxs).values
    precip_raw = np.where(precip_raw < 0, np.nan, precip_raw)
    precip_nyc = precip_raw[:, spatial_mask]
    idx_to_sub = {orig: new for new, orig in enumerate(needed_idxs)}

    max_precip_s4 = []
    for idxs in event_windows:
        if idxs:
            sub = [idx_to_sub[i] for i in idxs]
            cell_maxes = np.nanmax(precip_nyc[sub, :], axis=0)  # max over time, per cell
            if agg == "spatial-median":
                agg_mm = np.nanmedian(cell_maxes)
            else:
                agg_mm = np.nanmax(cell_maxes)
            val = float(agg_mm) * MM_TO_IN if not np.isnan(agg_mm) else np.nan
            max_precip_s4.append(val)
        else:
            max_precip_s4.append(np.nan)

    # Return extra data needed by Ida case study
    return max_precip_s4, ds_s4, s4_times, spatial_mask, lat2d, lon2d


# ── StageIV Composite Maps ────────────────────────────────────────────────────


def probability_matched_mean(fields):
    """
    Probability-matched mean (PMM) composite precipitation.

    fields: ndarray (n_events, nlat, nlon), NaN where no data.
    Returns PMM field (nlat, nlon).

    Spatial pattern from arithmetic mean; intensities drawn from the pooled
    distribution of all individual event fields, matched by rank so the
    highest mean-field cell gets the highest pooled value.
    """
    n_events, nlat, nlon = fields.shape
    mean_field = np.nanmean(fields, axis=0)

    pooled = fields.flatten()
    pooled = pooled[np.isfinite(pooled) & (pooled > 0)]
    if len(pooled) == 0 or np.nanmax(mean_field) == 0:
        return mean_field

    pooled_sorted = np.sort(pooled)[::-1]  # descending

    mean_flat = mean_field.flatten()
    pos_mask = np.isfinite(mean_flat) & (mean_flat > 0)
    pos_indices = np.where(pos_mask)[0]
    rank_order = pos_indices[np.argsort(mean_flat[pos_indices])[::-1]]  # desc rank

    pmm_flat = mean_flat.copy()
    for rank_idx, grid_idx in enumerate(rank_order):
        pool_idx = rank_idx * n_events
        pmm_flat[grid_idx] = (
            pooled_sorted[pool_idx] if pool_idx < len(pooled_sorted) else 0.0
        )

    return pmm_flat.reshape(nlat, nlon)


def compute_stageiv_node_composites(bmu_df, xdim, ydim, window_hours=6):
    """
    Collect per-event StageIV max-over-window precip fields for each node.

    Returns
    -------
    node_fields : dict {(i, j): ndarray (n_events, nlat_reg, nlon_reg)} in inches
    lat2d_reg, lon2d_reg : 2-D coordinate arrays for the regional domain
    """
    reg_lat_min, reg_lat_max = 38.5, 44.0
    reg_lon_min, reg_lon_max = -78.0, -70.0
    MM_TO_IN = 1.0 / 25.4

    ds_s4 = xr.open_dataset(STAGEIV_NC, chunks={"time": 168})
    s4_times = pd.DatetimeIndex(ds_s4["time"].values)
    lat2d = ds_s4["latitude"].values
    lon2d = ds_s4["longitude"].values

    reg_mask = (
        (lat2d >= reg_lat_min)
        & (lat2d <= reg_lat_max)
        & (lon2d >= reg_lon_min)
        & (lon2d <= reg_lon_max)
    )
    row_idxs, col_idxs = np.where(reg_mask)
    r_min, r_max = row_idxs.min(), row_idxs.max() + 1
    c_min, c_max = col_idxs.min(), col_idxs.max() + 1

    lat2d_reg = lat2d[r_min:r_max, c_min:c_max]
    lon2d_reg = lon2d[r_min:r_max, c_min:c_max]
    reg_mask_sub = reg_mask[r_min:r_max, c_min:c_max]

    all_needed = set()
    event_windows = []
    for _, row in bmu_df.iterrows():
        t0 = row["timestamp"]
        tmask = (s4_times >= t0 - pd.Timedelta(hours=window_hours)) & (
            s4_times <= t0 + pd.Timedelta(hours=window_hours)
        )
        idxs = np.where(tmask)[0].tolist()
        event_windows.append(idxs)
        all_needed.update(idxs)

    needed_idxs = sorted(all_needed)
    print(f"Loading {len(needed_idxs)} StageIV time steps for composite maps ...")
    precip_raw = (
        ds_s4["precipitation"]
        .isel(time=needed_idxs)
        .values[:, r_min:r_max, c_min:c_max]
        .astype(float)
    )
    precip_raw = np.where(precip_raw < 0, np.nan, precip_raw)
    precip_raw[:, ~reg_mask_sub] = np.nan
    idx_to_sub = {orig: new for new, orig in enumerate(needed_idxs)}
    ds_s4.close()

    node_fields = {}
    for i in range(xdim):
        for j in range(ydim):
            positions = np.where(
                (bmu_df["node_i"] == i) & (bmu_df["node_j"] == j)
            )[0]
            event_list = []
            for pos in positions:
                idxs = event_windows[pos]
                if idxs:
                    sub = [idx_to_sub[k] for k in idxs]
                    field_in = np.nanmax(precip_raw[sub], axis=0) * MM_TO_IN
                else:
                    field_in = np.full(lat2d_reg.shape, np.nan)
                event_list.append(field_in)
            node_fields[(i, j)] = np.stack(event_list, axis=0)

    return node_fields, lat2d_reg, lon2d_reg


def plot_stageiv_composite_maps(
    node_fields, xdim, ydim, lat2d, lon2d, fig_dir, _lbl, map_extent, fname_suffix
):
    """
    Plot PMM StageIV composite precipitation maps for each SOM node.

    Parameters
    ----------
    map_extent : [lon_min, lon_max, lat_min, lat_max]
    fname_suffix : appended to filename before .png (e.g. "_regional", "_nyc")
    """
    nws_levels = [0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00]
    _n_bins = len(nws_levels) - 1
    _mrms_colors = [
        "#0099ff",  # 0.10–0.25"  light blue
        "#0000ff",  # 0.25–0.50"  blue
        "#00ff00",  # 0.50–0.75"  green
        "#009900",  # 0.75–1.00"  dark green
        "#ffff00",  # 1.00–1.25"  yellow
        "#ffaa00",  # 1.25–1.50"  orange
        "#ff4400",  # 1.50–2.00"  red-orange
        "#cc0000",  # 2.00–2.50"  red
        "#ff00ff",  # 2.50–3.00"  magenta
        "#9900cc",  # 3.00–4.00"  purple
    ]
    cmap_nws = ListedColormap(_mrms_colors)
    norm_nws = BoundaryNorm(nws_levels, ncolors=_n_bins, clip=False)

    # Use 10m features for zoomed-in NYC domain, 50m for regional
    lon_span = map_extent[1] - map_extent[0]
    scale = "10m" if lon_span < 4 else "50m"

    fig, axes = plt.subplots(
        ydim,
        xdim,
        figsize=(6, 4.5),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
        dpi=600,
    )

    for i in range(xdim):
        for j in range(ydim):
            ax = axes[j, i]
            fields = node_fields[(i, j)]
            n_events = int(np.sum(~np.all(np.isnan(fields), axis=(1, 2))))
            pmm = probability_matched_mean(fields)
            pmm_plot = np.where(pmm >= nws_levels[0], pmm, np.nan)

            ax.set_extent(map_extent)
            ax.add_feature(
                cfeature.LAND.with_scale(scale), facecolor="#ebebeb", zorder=0
            )
            ax.add_feature(
                cfeature.OCEAN.with_scale(scale), facecolor="#dde8f2", zorder=0
            )
            ax.add_feature(cfeature.STATES.with_scale(scale), linewidth=0.3, zorder=4)
            ax.add_feature(
                cfeature.COASTLINE.with_scale(scale), linewidth=0.4, zorder=4
            )
            ax.pcolormesh(
                lon2d,
                lat2d,
                pmm_plot,
                cmap=cmap_nws,
                norm=norm_nws,
                shading="auto",
                zorder=2,
                transform=ccrs.PlateCarree(),
            )
            ax.scatter(
                -74.0,
                40.7,
                color="black",
                s=10,
                marker="*",
                zorder=5,
                transform=ccrs.PlateCarree(),
            )
            ax.set_title(f"{node_label(i, j)}  N={n_events}", fontsize=6)

    sm = ScalarMappable(cmap=cmap_nws, norm=norm_nws)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=list(axes.flat),
        orientation="vertical",
        pad=0.02,
        shrink=0.85,
        extend="max",
    )
    cbar.set_label("PMM Precip (in)", fontsize=7)
    cbar.set_ticks(nws_levels)
    cbar.ax.tick_params(labelsize=5)

    fig.suptitle(
        "Stage IV Composite Precipitation by SOM Node\n"
        r"PMM of hourly max rainfall ($\pm$6 hr window)",
        fontsize=8,
    )
    fname = f"Z500_and_{_lbl}_som_stageiv_composite_maps{fname_suffix}.png"
    plt.savefig(f"{fig_dir}/{fname}", bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


# ── ERA5 Precipitation ───────────────────────────────────────────────────────


def compute_era5_max_precip(bmu_df, window_hours=6):
    """Compute max hourly ERA5 precip over NYC for each event."""
    M_TO_IN = 1000.0 / 25.4
    era5_lat_max, era5_lat_min = 40.75, 40.5
    era5_lon_min, era5_lon_max = -74.25, -73.75

    max_precip_era5 = pd.Series(np.nan, index=bmu_df.index)

    for year, grp in bmu_df.groupby(bmu_df["timestamp"].dt.year):
        fpath = f"{ERA5_TP_DIR}/era5_tp_NWHem_{year}.nc"
        if not os.path.exists(fpath):
            print(f"  {year}: file not found, skipping")
            continue

        ds_e5 = xr.open_dataset(fpath, engine="netcdf4")
        era5_times = pd.DatetimeIndex(ds_e5["valid_time"].values)
        tp_nyc = (
            ds_e5["tp"]
            .sel(
                latitude=slice(era5_lat_max, era5_lat_min),
                longitude=slice(era5_lon_min, era5_lon_max),
            )
            .load()
            .values
        )
        ds_e5.close()
        print(f"  {year}: {tp_nyc.shape} loaded, {len(grp)} event(s)")

        for idx, row in grp.iterrows():
            t0 = row["timestamp"]
            tmask = (era5_times >= t0 - pd.Timedelta(hours=window_hours)) & (
                era5_times <= t0 + pd.Timedelta(hours=window_hours)
            )
            t_idxs = np.where(tmask)[0]
            if len(t_idxs) == 0:
                continue
            max_m = float(np.nanmax(tp_nyc[t_idxs]))
            if not np.isnan(max_m):
                max_precip_era5.loc[idx] = max_m * M_TO_IN

    return max_precip_era5


# ── Tropical Cyclone Association ──────────────────────────────────────────────


def compute_tc_associations(bmu_df, xdim, ydim, time_window_hours=6):
    """Cross-reference flash flood events with IBTrACS."""
    ibtracs = pd.read_csv(IBTRACS_PATH)
    ibtracs["ISO_TIME"] = pd.to_datetime(ibtracs["ISO_TIME"])

    lat_min, lat_max = 30.0, 54.0
    lon_min, lon_max = -100.0, -60.0

    ibtracs_domain = ibtracs[
        (ibtracs["LAT"] >= lat_min)
        & (ibtracs["LAT"] <= lat_max)
        & (ibtracs["LON"] >= lon_min)
        & (ibtracs["LON"] <= lon_max)
    ].copy()

    print(f"Total IBTrACS records : {len(ibtracs):,}")
    print(f"Records in domain     : {len(ibtracs_domain):,}")
    print(f"Unique storms         : {ibtracs_domain['SID'].nunique()}")

    tc_associations = []
    for _, row in bmu_df.iterrows():
        event_time = row["timestamp"]
        lo = event_time - pd.Timedelta(hours=time_window_hours)
        hi = event_time + pd.Timedelta(hours=time_window_hours)
        matching_tcs = ibtracs_domain[
            (ibtracs_domain["ISO_TIME"] >= lo) & (ibtracs_domain["ISO_TIME"] <= hi)
        ]

        if len(matching_tcs) > 0:
            storm_ids = matching_tcs["SID"].unique()
            tc_associations.append(
                {
                    "timestamp": event_time,
                    "node_i": row["node_i"],
                    "node_j": row["node_j"],
                    "tc_present": True,
                    "n_storms": len(storm_ids),
                    "storm_ids": ", ".join(storm_ids),
                    "storm_status": (
                        matching_tcs["STAT"].mode().iloc[0]
                        if len(matching_tcs["STAT"].dropna()) > 0
                        else "Unknown"
                    ),
                }
            )
        else:
            tc_associations.append(
                {
                    "timestamp": event_time,
                    "node_i": row["node_i"],
                    "node_j": row["node_j"],
                    "tc_present": False,
                    "n_storms": 0,
                    "storm_ids": "",
                    "storm_status": "",
                }
            )

    tc_df = pd.DataFrame(tc_associations)
    return tc_df, ibtracs, ibtracs_domain


def plot_tc_association(tc_df, xdim, ydim, fig_dir, _lbl):
    """Plot TC association bar charts."""
    node_labels = [node_label(i, j) for i in range(xdim) for j in range(ydim)]
    n_nodes = xdim * ydim

    tc_counts = np.zeros((xdim, ydim))
    non_tc_counts = np.zeros((xdim, ydim))
    for i in range(xdim):
        for j in range(ydim):
            nd = tc_df[(tc_df["node_i"] == i) & (tc_df["node_j"] == j)]
            tc_counts[i, j] = nd["tc_present"].sum()
            non_tc_counts[i, j] = (~nd["tc_present"]).sum()

    x = np.arange(len(node_labels))
    tc_flat = tc_counts.flatten()
    non_tc_flat = non_tc_counts.flatten()
    totals = tc_flat + non_tc_flat
    tc_pct = 100 * tc_flat / totals

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), dpi=600, constrained_layout=True)

    ax = axes[0]
    ax.bar(x, non_tc_flat, 0.6, label="Non-TC", color="steelblue", alpha=0.9)
    ax.bar(
        x,
        tc_flat,
        0.6,
        bottom=non_tc_flat,
        label="TC-Associated",
        color="coral",
        alpha=0.9,
    )
    ax.set_xlabel("SOM Node", fontsize=7)
    ax.set_ylabel("Number of Events", fontsize=7)
    ax.set_title("Flash Flood Events by TC Association", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(node_labels, fontsize=6)
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, linewidth=0.3, alpha=0.5, axis="y")

    ax = axes[1]
    bars = ax.bar(x, tc_pct, 0.6, color="coral", alpha=0.9, edgecolor="white")
    for bar, pct, tc, total in zip(bars, tc_pct, tc_flat, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{int(tc)}/{int(total)}",
            ha="center",
            va="bottom",
            fontsize=6,
        )
    overall_pct = 100 * tc_flat.sum() / totals.sum()
    ax.axhline(
        overall_pct,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"Overall: {overall_pct:.1f}\\%",
    )
    ax.set_xlabel("SOM Node", fontsize=7)
    ax.set_ylabel(r"TC-Associated Events (\%)", fontsize=7)
    ax.set_title("Percentage of Events with TC Influence", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(node_labels, fontsize=6)
    ax.set_ylim(0, max(tc_pct) + 15)
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, linewidth=0.3, alpha=0.5, axis="y")

    plt.savefig(
        f"{fig_dir}/Z500_and_{_lbl}_som_tc_association.png", bbox_inches="tight"
    )
    plt.close()
    print(f"Saved TC association figure.")

    return tc_flat, non_tc_flat


def print_tc_stats(tc_df, tc_flat, non_tc_flat, xdim, ydim):
    """Print chi-square and Fisher's exact test results."""
    node_labels = [node_label(i, j) for i in range(xdim) for j in range(ydim)]
    n_nodes = xdim * ydim

    contingency_tc = np.array([tc_flat, non_tc_flat])
    print("\n" + "=" * 60)
    print("CHI-SQUARE TEST: TC Association vs SOM Node")
    print("=" * 60)

    contingency_df = pd.DataFrame(
        {
            lbl: [int(tc), int(ntc)]
            for lbl, tc, ntc in zip(node_labels, tc_flat, non_tc_flat)
        },
        index=["TC", "Non-TC"],
    )
    contingency_df["Total"] = contingency_df.sum(axis=1)
    print(contingency_df)

    chi2, p_value, dof, expected = chi2_contingency(contingency_tc)
    print(f"\nChi2={chi2:.3f}  dof={dof}  p={p_value:.4f}")
    if expected.min() < 5:
        print("Warning: min expected count < 5")
    if p_value < 0.05:
        print("-> REJECT H0: TC association differs across nodes.")
    else:
        print("-> FAIL TO REJECT H0: no significant difference across nodes.")

    print("\n" + "=" * 60)
    print("PAIRWISE FISHER'S EXACT TESTS (Bonferroni-corrected)")
    print("=" * 60)
    pairs = list(combinations(range(n_nodes), 2))
    alpha_corrected = 0.05 / len(pairs)
    print(f"Bonferroni alpha: {alpha_corrected:.4f}\n")
    for idx1, idx2 in pairs:
        table = np.array(
            [[tc_flat[idx1], tc_flat[idx2]], [non_tc_flat[idx1], non_tc_flat[idx2]]]
        )
        odds_ratio, p = fisher_exact(table)
        sig = "***" if p < alpha_corrected else ""
        print(
            f"{node_labels[idx1]} vs {node_labels[idx2]}: "
            f"OR={odds_ratio:.2f}, p={p:.4f} {sig}"
        )


def plot_tc_tracks(
    tc_df,
    ibtracs,
    xdim,
    ydim,
    fig_dir,
    _lbl,
):
    """Plot TC tracks during flash flood events."""
    node_colors = {
        (0, 0): "tab:blue",
        (0, 1): "tab:orange",
        (1, 0): "tab:green",
        (1, 1): "tab:red",
    }
    lat_min, lat_max = 30.0, 54.0
    lon_min, lon_max = -100.0, -60.0

    tc_events = tc_df[tc_df["tc_present"]].sort_values("timestamp")

    fig, ax = plt.subplots(
        figsize=(8, 5), subplot_kw={"projection": ccrs.PlateCarree()}, dpi=600
    )

    for _, row in tc_events.iterrows():
        event_time = row["timestamp"]
        node_i, node_j = int(row["node_i"]), int(row["node_j"])
        color = node_colors[(node_i, node_j)]

        for sid in row["storm_ids"].split(", "):
            full_track = ibtracs[ibtracs["SID"] == sid].sort_values("ISO_TIME")
            if len(full_track) < 2:
                continue

            ax.plot(
                full_track["LON"],
                full_track["LAT"],
                color=color,
                linewidth=0.8,
                alpha=0.3,
                transform=ccrs.PlateCarree(),
            )

            wm = (full_track["ISO_TIME"] >= event_time - pd.Timedelta(hours=48)) & (
                full_track["ISO_TIME"] <= event_time + pd.Timedelta(hours=48)
            )
            wt = full_track[wm]
            if len(wt) > 1:
                ax.plot(
                    wt["LON"],
                    wt["LAT"],
                    color=color,
                    linewidth=1.2,
                    alpha=0.85,
                    transform=ccrs.PlateCarree(),
                )

            ci = (full_track["ISO_TIME"] - event_time).abs().idxmin()
            cp = full_track.loc[ci]
            ax.scatter(
                cp["LON"],
                cp["LAT"],
                color=color,
                s=20,
                marker="o",
                edgecolor="black",
                linewidth=0.5,
                zorder=7,
                transform=ccrs.PlateCarree(),
            )

    ax.plot(
        [lon_min, lon_max, lon_max, lon_min, lon_min],
        [lat_min, lat_min, lat_max, lat_max, lat_min],
        "k--",
        linewidth=1,
        transform=ccrs.PlateCarree(),
    )
    ax.scatter(
        -74.0,
        40.7,
        color="black",
        s=100,
        marker="*",
        zorder=10,
        transform=ccrs.PlateCarree(),
    )
    ax.text(-73.5, 40.7, "NYC", fontsize=7, transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.set_extent([lon_min - 5, lon_max + 5, lat_min - 5, lat_max + 5])

    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            color=node_colors[(i, j)],
            linewidth=2,
            label=f"Node {node_label(i, j)}",
        )
        for i in range(xdim)
        for j in range(ydim)
    ] + [
        plt.Line2D(
            [0], [0], color="gray", linewidth=0.8, alpha=0.5, label="Full track"
        ),
        plt.Line2D([0], [0], color="gray", linewidth=1.2, label=r"$\pm$48 hr window"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=6)
    ax.set_title(
        "TC Tracks During Flash Flood Events\n"
        r"(bold = $\pm$48 hr window, $\bullet$ = event time)",
        fontsize=8,
    )
    plt.savefig(f"{fig_dir}/Z500_and_{_lbl}_som_tc_tracks.png", bbox_inches="tight")
    plt.close()
    print(f"Saved TC tracks figure.")


# ── Ida Case Study ───────────────────────────────────────────────────────────


def plot_ida_case_study(
    precip_dfs,
    ds_s4,
    s4_times,
    spatial_mask,
    lat2d,
    lon2d,
    fig_dir,
    _lbl,
):
    """Three-panel comparison of max 1-hour precip during Hurricane Ida."""
    M_TO_IN = 1000.0 / 25.4

    nws_levels = [
        0.01,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.00,
        1.25,
        1.50,
        1.75,
        2.00,
        2.50,
        3.00,
        3.50,
        4.00,
        5.00,
        6.00,
        8.00,
    ]
    _n_bins = len(nws_levels) - 1
    cmap_nws = ListedColormap(ctables.registry["precipitation"]).resampled(_n_bins)
    norm_nws = BoundaryNorm(nws_levels, ncolors=_n_bins, clip=False)

    ida_start = pd.Timestamp("2021-09-01 00:00:00")
    ida_end = pd.Timestamp("2021-09-02 23:59:00")

    map_lon_min, map_lon_max = -74.8, -73.4
    map_lat_min, map_lat_max = 40.3, 41.2

    # Station labels
    station_labels = {
        "JFK": ("JFK", "left", 0.05, -0.05),
        "LGA": ("LGA", "left", 0.05, 0.02),
        "Central Park": ("CP", "right", -0.05, 0.03),
        "EWR": ("EWR", "left", 0.05, 0.02),
    }

    station_ida_max = {}
    for name, df in precip_dfs.items():
        w = df.loc[ida_start:ida_end, "precip"]
        station_ida_max[name] = float(w.max()) if len(w) > 0 else np.nan

    print("\nASOS max hourly precip during Ida:")
    for name, val in station_ida_max.items():
        print(f"  {name:<15}: {val:.2f} in")

    # StageIV
    ida_s4_mask = (s4_times >= ida_start) & (s4_times <= ida_end)
    ida_s4_raw = ds_s4["precipitation"].isel(time=np.where(ida_s4_mask)[0]).values
    ida_s4_raw = np.where(ida_s4_raw < 0, np.nan, ida_s4_raw)
    ida_s4_max_in = np.nanmax(ida_s4_raw, axis=0) / 25.4

    # ERA5
    era5_lat_max, era5_lat_min = 40.75, 40.5
    era5_lon_min, era5_lon_max = -74.25, -73.75

    ds_e5_ida = xr.open_dataset(
        f"{ERA5_TP_DIR}/era5_tp_NWHem_2021.nc", engine="netcdf4"
    )
    era5_times_2021 = pd.DatetimeIndex(ds_e5_ida["valid_time"].values)
    ida_e5_idxs = np.where(
        (era5_times_2021 >= ida_start) & (era5_times_2021 <= ida_end)
    )[0]

    e5_da = (
        ds_e5_ida["tp"]
        .sel(
            latitude=slice(era5_lat_max, era5_lat_min),
            longitude=slice(era5_lon_min, era5_lon_max),
        )
        .isel(valid_time=ida_e5_idxs)
        .load()
    )
    e5_lats = e5_da.latitude.values
    e5_lons = e5_da.longitude.values
    ds_e5_ida.close()

    ida_e5_max_in = np.nanmax(e5_da.values, axis=0) * M_TO_IN
    e5_lon2d, e5_lat2d = np.meshgrid(e5_lons, e5_lats)

    # Build figure
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10, 3.5),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
        dpi=300,
    )
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.88))

    for ax in axes:
        ax.set_extent([map_lon_min, map_lon_max, map_lat_min, map_lat_max])
        ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#ebebeb", zorder=0)
        ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#dde8f2", zorder=0)
        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.5, zorder=4)
        ax.add_feature(cfeature.STATES.with_scale("10m"), linewidth=0.4, zorder=4)
        ax.scatter(
            -74.0,
            40.7,
            color="black",
            s=20,
            marker="*",
            zorder=5,
            transform=ccrs.PlateCarree(),
        )

    for ax, title in zip(
        axes,
        [
            "(a) ASOS Stations",
            "(b) Stage IV (4-km)",
            r"(c) ERA5 ($0.25^{\circ}$)",
        ],
    ):
        ax.set_title(title, fontsize=7)

    # (a) ASOS
    ax = axes[0]
    for name, (slat, slon) in STATION_COORDS.items():
        val = station_ida_max.get(name, np.nan)
        ax.scatter(
            [slon],
            [slat],
            c=[val],
            cmap=cmap_nws,
            norm=norm_nws,
            s=70,
            marker="o",
            edgecolors="k",
            linewidths=0.5,
            zorder=5,
            transform=ccrs.PlateCarree(),
        )
        label, ha, dlon, dlat = station_labels[name]
        ax.text(
            slon + dlon,
            slat + dlat,
            label,
            fontsize=5,
            ha=ha,
            transform=ccrs.PlateCarree(),
            zorder=6,
        )

    # (b) StageIV
    ax = axes[1]
    plot_s4 = np.where(spatial_mask, ida_s4_max_in, np.nan)
    plot_s4 = np.where(plot_s4 >= nws_levels[0], plot_s4, np.nan)
    ax.pcolormesh(
        lon2d,
        lat2d,
        plot_s4,
        cmap=cmap_nws,
        norm=norm_nws,
        shading="auto",
        zorder=2,
        transform=ccrs.PlateCarree(),
    )

    # (c) ERA5
    ax = axes[2]
    plot_e5 = np.where(ida_e5_max_in >= nws_levels[0], ida_e5_max_in, np.nan)
    ax.pcolormesh(
        e5_lon2d,
        e5_lat2d,
        plot_e5,
        cmap=cmap_nws,
        norm=norm_nws,
        shading="auto",
        zorder=2,
        edgecolors="gray",
        linewidths=0.4,
        transform=ccrs.PlateCarree(),
    )

    # Shared colorbar
    sm = ScalarMappable(cmap=cmap_nws, norm=norm_nws)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=list(axes.flat),
        orientation="vertical",
        pad=0.02,
        shrink=0.85,
        extend="max",
    )
    cbar.set_label("Max Hourly Precip (in)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.suptitle(
        "Hurricane Ida: Maximum Hourly Precipitation by Source\n1--2 September 2021",
        fontsize=8,
        y=0.9,
    )
    plt.savefig(f"{fig_dir}/ida_precip_three_sources.png", bbox_inches="tight")
    plt.close()
    print(f"Saved Ida case study figure.")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    setup_plotting()

    cfg = MOISTURE_CONFIGS[args.moisture_var]
    paths = get_paths(args.moisture_var, args.moisture_weight, daily=args.daily)
    fig_dir = paths["fig_dir"]
    _lbl = paths["file_label"]
    xdim, ydim = 2, 2
    n_nodes = xdim * ydim

    os.makedirs(fig_dir, exist_ok=True)

    # ── Load BMU assignments ──────────────────────────────────────────────────
    bmu_csv = paths["bmu_csv_path"]
    print(f"Loading BMU assignments from {bmu_csv} ...")
    bmu_df = pd.read_csv(bmu_csv)
    bmu_df["timestamp"] = pd.to_datetime(bmu_df["timestamp"])
    bmu_df["timestamp_local"] = (
        bmu_df["timestamp"]
        .dt.tz_localize("UTC")
        .dt.tz_convert("EST")
        .dt.tz_localize(None)
    )

    # ── ASOS Precipitation ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ASOS PRECIPITATION ANALYSIS")
    print("=" * 60)
    precip_dfs = load_asos_precip()
    bmu_df["max_precip_in"] = compute_asos_max_precip(bmu_df, precip_dfs)
    print(
        f"Matched {len(bmu_df)} events | "
        f"range: {bmu_df['max_precip_in'].min():.2f}--"
        f"{bmu_df['max_precip_in'].max():.2f} in"
    )

    node_stats = (
        bmu_df.groupby(["node_i", "node_j"])["max_precip_in"]
        .agg(count="count", mean="mean", median="median", std="std")
        .round(2)
    )
    print(node_stats)

    plot_precip_histograms(
        bmu_df,
        "max_precip_in",
        xdim,
        ydim,
        fig_dir,
        _lbl,
        color="steelblue",
        source_label="ASOS",
    )

    # ── StageIV Precipitation ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGEIV PRECIPITATION ANALYSIS")
    print("=" * 60)
    max_precip_s4, ds_s4, s4_times, spatial_mask, lat2d, lon2d = (
        compute_stageiv_max_precip(bmu_df, agg=args.stageiv_agg)
    )
    bmu_df["max_precip_s4_in"] = max_precip_s4

    n_cov = bmu_df["max_precip_s4_in"].notna().sum()
    print(f"Events with StageIV coverage : {n_cov}/{len(bmu_df)}")

    s4_node_stats = (
        bmu_df.dropna(subset=["max_precip_s4_in"])
        .groupby(["node_i", "node_j"])["max_precip_s4_in"]
        .agg(count="count", mean="mean", median="median", std="std")
        .round(2)
    )
    print(s4_node_stats)

    if args.stageiv_agg == "spatial-median":
        s4_source_label = "Stage IV (spatial median)"
        s4_xlabel = "Median of Max Hourly Precip (in)"
    else:
        s4_source_label = "Stage IV"
        s4_xlabel = "Max Hourly Precip (in)"

    plot_precip_histograms(
        bmu_df,
        "max_precip_s4_in",
        xdim,
        ydim,
        fig_dir,
        _lbl,
        color="darkorange",
        source_label=s4_source_label,
        bins=np.arange(0, 4.01, 0.25),
        xlim=(0, 4.0),
        xlabel=s4_xlabel,
    )

    # ── StageIV Composite Maps ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGEIV COMPOSITE MAPS")
    print("=" * 60)
    node_fields, lat2d_reg, lon2d_reg = compute_stageiv_node_composites(
        bmu_df, xdim, ydim
    )
    # Regional domain
    plot_stageiv_composite_maps(
        node_fields, xdim, ydim, lat2d_reg, lon2d_reg, fig_dir, _lbl,
        map_extent=[-78.0, -70.0, 38.5, 44.0],
        fname_suffix="_regional",
    )
    # NYC zoom (matches Ida case study domain)
    plot_stageiv_composite_maps(
        node_fields, xdim, ydim, lat2d_reg, lon2d_reg, fig_dir, _lbl,
        map_extent=[-74.8, -73.4, 40.3, 41.2],
        fname_suffix="_nyc",
    )

    # ── ERA5 Precipitation ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ERA5 PRECIPITATION ANALYSIS")
    print("=" * 60)
    bmu_df["max_precip_era5_in"] = compute_era5_max_precip(bmu_df)

    n_cov = bmu_df["max_precip_era5_in"].notna().sum()
    print(f"\nEvents with ERA5 coverage : {n_cov}/{len(bmu_df)}")

    era5_node_stats = (
        bmu_df.dropna(subset=["max_precip_era5_in"])
        .groupby(["node_i", "node_j"])["max_precip_era5_in"]
        .agg(count="count", mean="mean", median="median", std="std")
        .round(3)
    )
    print(era5_node_stats)

    era5_max = bmu_df["max_precip_era5_in"].max()
    plot_precip_histograms(
        bmu_df,
        "max_precip_era5_in",
        xdim,
        ydim,
        fig_dir,
        _lbl,
        color="mediumpurple",
        source_label="ERA5",
        bins=np.arange(0, max(era5_max + 0.25, 2.01), 0.1),
        xlim=(0, max(era5_max + 0.1, 2.0)),
    )

    # ── Tropical Cyclone Association ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TROPICAL CYCLONE ASSOCIATION")
    print("=" * 60)
    tc_df, ibtracs, ibtracs_domain = compute_tc_associations(bmu_df, xdim, ydim)

    n_tc = tc_df["tc_present"].sum()
    print(
        f"\nTC-associated events: {n_tc}/{len(tc_df)} ({100 * n_tc / len(tc_df):.1f}%)"
    )
    for i in range(xdim):
        for j in range(ydim):
            nd = tc_df[(tc_df["node_i"] == i) & (tc_df["node_j"] == j)]
            tc_c = nd["tc_present"].sum()
            print(f"  Node {node_label(i, j)}: {tc_c}/{len(nd)} ({100 * tc_c / len(nd):.1f}%)")

    tc_events = tc_df[tc_df["tc_present"]].sort_values("timestamp")
    print("\nTC-Associated Flash Flood Events:")
    print("-" * 80)
    for _, row in tc_events.iterrows():
        print(
            f"{row['timestamp'].strftime('%Y-%m-%d %H:%M')} | "
            f"Node {node_label(int(row['node_i']), int(row['node_j']))} | "
            f"Status: {row['storm_status']:>3} | {row['storm_ids']}"
        )

    tc_flat, non_tc_flat = plot_tc_association(tc_df, xdim, ydim, fig_dir, _lbl)
    print_tc_stats(tc_df, tc_flat, non_tc_flat, xdim, ydim)
    plot_tc_tracks(tc_df, ibtracs, xdim, ydim, fig_dir, _lbl)

    # ── TC vs non-TC precipitation comparison ─────────────────────────────────
    bmu_df_with_tc = bmu_df.merge(
        tc_df[["timestamp", "tc_present", "storm_ids"]], on="timestamp", how="left"
    )
    tc_precip = bmu_df_with_tc[bmu_df_with_tc["tc_present"]]["max_precip_in"].dropna()
    non_tc_precip = bmu_df_with_tc[~bmu_df_with_tc["tc_present"]][
        "max_precip_in"
    ].dropna()

    print("\n" + "=" * 60)
    print("TC vs NON-TC PRECIPITATION COMPARISON")
    print("=" * 60)
    hdr = f"{'Category':<20} {'N':>6} {'Mean':>8} {'Median':>8} {'Std':>8}"
    print(hdr)
    print("-" * len(hdr))
    for label, s in [("TC-Associated", tc_precip), ("Non-TC", non_tc_precip)]:
        print(
            f"{label:<20} {len(s):>6} {s.mean():>8.2f} "
            f"{s.median():>8.2f} {s.std():>8.2f}"
        )

    stat, p = mannwhitneyu(tc_precip, non_tc_precip, alternative="two-sided")
    print(f"\nMann-Whitney U={stat:.1f}, p={p:.4f}")
    if p < 0.05:
        print("-> Significant difference in precip intensity (TC vs non-TC).")
    else:
        print("-> No significant difference in precip intensity.")

    tc_csv_suffix = "_daily" if args.daily else ""
    tc_df.to_csv(
        os.path.join(DATA_DIR, f"som_2x2_tc_associations{tc_csv_suffix}.csv"),
        index=False,
    )
    print("Saved TC association data.")

    # ── Ida Case Study ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("IDA CASE STUDY")
    print("=" * 60)
    plot_ida_case_study(
        precip_dfs,
        ds_s4,
        s4_times,
        spatial_mask,
        lat2d,
        lon2d,
        fig_dir,
        _lbl,
    )

    print(f"\nAll outputs saved to {fig_dir}/")


if __name__ == "__main__":
    main()
