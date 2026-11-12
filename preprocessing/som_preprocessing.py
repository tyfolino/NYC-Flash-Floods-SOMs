"""
SOM Data Preprocessing
By: Ty Janoski

Preprocesses ERA5 variables for the NYC flash flood SOM analysis.
Usage (from the project root):
    NYCFF_INTERMEDIATE=v2 python -m preprocessing.som_preprocessing VARIABLE \
        [--climatology monthday|doy] [--no-plots] [--no-snapshots]
Variables: Z500, mslp, tcwv, ivt, tp, theta_e

Writes to som_analysis.config.SOM_INTERMEDIATE_PATH (select with
NYCFF_INTERMEDIATE); refuses to write into the published v1 directory.
"""

import argparse
import os

import cartopy.crs as ccrs
import cmweather  # noqa: F401
import dask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import xarray as xr
from dask.diagnostics.progress import ProgressBar

from preprocessing.climatology import climatology, lat_weight, standardize
from som_analysis.config import (
    ERA5_SLICED_DIR,
    FIGS_DIR,
    SOM_INTERMEDIATE_PATH,
    check_intermediate_writable,
)
from som_analysis.timeutils import ff_event_hours_utc

# Cap dask's threaded scheduler (default = all cores) on the shared server.
dask.config.set(
    scheduler="threads", num_workers=int(os.environ.get("NYCFF_DASK_WORKERS", "4"))
)

plt.style.use(["science", "nature", "grid"])
plt.rcParams["text.usetex"] = True

# ── Config ────────────────────────────────────────────────────────────────────

CONFIGS = {
    "Z500": {
        "file_path": f"{ERA5_SLICED_DIR}/era5_z500_NWHem_*.nc",
        "load_func": "mfdataset",
        "load_kwargs": {
            "combine": "by_coords",
            "decode_times": True,
            "chunks": {"valid_time": 8760},
        },
        "var_name": "z",
        "squeeze_dims": ["pressure_level"],
        "time_dim": "valid_time",
        "lat_dim": "latitude",
        "lon_dim": "longitude",
        "compute_anomalies": True,
        "compute_magnitude": None,
        "output_prefix": "era5_Z500",
        "plot_config": {
            "title": "Z$_{500}$",
            "cmap": "viridis",
            "raw_levels": np.arange(522, 595, 6),
            "raw_transform": lambda x: x / 10 / 9.81,
            "raw_unit": "dam",
            "std_levels": np.arange(-3, 3.1, 0.5),
            "extend": "neither",
            "fig_dir": os.path.join(FIGS_DIR, "Z500-SOM"),
        },
    },
    "mslp": {
        "file_path": f"{ERA5_SLICED_DIR}/era5_mslp_NWHem_*.nc",
        "load_func": "mfdataset",
        "load_kwargs": {
            "combine": "by_coords",
            "decode_times": True,
            "chunks": {"valid_time": 8760},
        },
        "var_name": "msl",
        "time_dim": "valid_time",
        "lat_dim": "latitude",
        "lon_dim": "longitude",
        "compute_anomalies": True,
        "compute_magnitude": None,
        "output_prefix": "era5_mslp",
        "plot_config": {
            "title": "MSLP",
            "cmap": "viridis",
            "raw_levels": np.arange(988, 1032, 4),
            "raw_transform": lambda x: x / 100,
            "raw_unit": "mb",
            "std_levels": np.arange(-3, 3.1, 0.5),
            "extend": "both",
            "fig_dir": os.path.join(FIGS_DIR, "mslp-SOM"),
        },
    },
    "tcwv": {
        "file_path": f"{ERA5_SLICED_DIR}/era5_tcwv_NWHem_*.nc",
        "load_func": "mfdataset",
        "load_kwargs": {
            "combine": "by_coords",
            "decode_times": True,
            "chunks": {"valid_time": 8760},
        },
        "var_name": "tcwv",
        "time_dim": "valid_time",
        "lat_dim": "latitude",
        "lon_dim": "longitude",
        "compute_anomalies": True,
        "compute_magnitude": None,
        "output_prefix": "era5_tcwv",
        "plot_config": {
            "title": "TCWV",
            "cmap": "viridis",
            "raw_levels": np.arange(0, 51, 5),
            "raw_transform": None,
            "raw_unit": "kg m$^{-2}$",
            "std_levels": np.arange(-3, 3.1, 0.5),
            "extend": "max",
            "fig_dir": os.path.join(FIGS_DIR, "tcwv-SOM"),
        },
    },
    "ivt": {
        "file_path": f"{ERA5_SLICED_DIR}/era5_viwv_NWHem_*.nc",
        "load_func": "mfdataset",
        "load_kwargs": {
            "combine": "by_coords",
            "decode_times": True,
            "chunks": {"valid_time": 8760},
        },
        "time_dim": "valid_time",
        "lat_dim": "latitude",
        "lon_dim": "longitude",
        "compute_anomalies": True,
        "compute_magnitude": {"components": ["viwve", "viwvn"], "name": "ivt"},
        "output_prefix": "era5_ivt",
        "plot_config": {
            "title": "IVT",
            "cmap": "viridis",
            "raw_levels": np.arange(0, 701, 50),
            "raw_transform": None,
            "raw_unit": "kg m$^{-1}$ s$^{-1}$",
            "std_levels": np.arange(-6, 6.1, 1.0),
            "extend": "both",
            "fig_dir": os.path.join(FIGS_DIR, "Z500-and-ivtxy-SOM"),
            "components": [
                ("ivt", "IVT$_{mag}$", np.arange(0, 701, 50), "max"),
                ("viwve", "IVT$_{x}$", np.arange(-700, 701, 100), "both"),
                ("viwvn", "IVT$_{y}$", np.arange(-700, 701, 100), "both"),
            ],
        },
    },
    "tp": {
        "file_path": f"{ERA5_SLICED_DIR}/era5_tp_*.nc",
        "load_func": "mfdataset",
        "load_kwargs": {
            "combine": "by_coords",
            "decode_times": True,
            "chunks": {"valid_time": 8760},
        },
        "var_name": "tp",
        "time_dim": "valid_time",
        "lat_dim": "latitude",
        "lon_dim": "longitude",
        "compute_anomalies": False,
        "compute_magnitude": None,
        "output_prefix": "era5_tp",
        "plot_config": None,
    },
    "theta_e": {
        "file_path": f"{ERA5_SLICED_DIR}/*thetae*.nc",
        "load_func": "mfdataset",
        "load_kwargs": {
            "combine": "by_coords",
            "decode_times": True,
            "chunks": {"valid_time": 8760},
        },
        "var_name": "theta_e",
        "time_dim": "valid_time",
        "lat_dim": "latitude",
        "lon_dim": "longitude",
        "compute_anomalies": True,
        "compute_magnitude": None,
        "output_prefix": "era5_thetae",
        "plot_config": {
            "title": r"$\theta_e$",
            "cmap": "viridis",
            "raw_levels": np.arange(280, 336, 5),
            "raw_transform": None,
            "raw_unit": "K",
            "std_levels": np.arange(-3, 3.1, 0.5),
            "extend": "both",
            "fig_dir": os.path.join(FIGS_DIR, "thetae-SOM"),
        },
    },
}

OUT_DIR = SOM_INTERMEDIATE_PATH.rstrip("/")

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("variable", choices=CONFIGS)
parser.add_argument("--no-plots", action="store_true")
parser.add_argument(
    "--climatology",
    choices=["monthday", "doy"],
    default="monthday",
    help="Calendar grouping of the climatology: monthday (v2, leap years "
    "aligned; default) or doy (as published).",
)
parser.add_argument(
    "--no-snapshots",
    action="store_true",
    help="Skip the 20 UTC warm-season snapshot files (only used by "
    "train_alldays_som --snapshot-hour).",
)
args = parser.parse_args()
check_intermediate_writable(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

cfg = CONFIGS[args.variable]
time_dim = cfg["time_dim"]
lat_dim = cfg["lat_dim"]
lon_dim = cfg["lon_dim"]
prefix = cfg["output_prefix"]

# ── Load ──────────────────────────────────────────────────────────────────────

print(f"Loading {args.variable}...")

if cfg["load_func"] == "dataarray":
    data = xr.load_dataarray(cfg["file_path"], decode_timedelta=True)
else:  # mfdataset
    data = xr.open_mfdataset(cfg["file_path"], **cfg["load_kwargs"])
    with ProgressBar():
        data = data.load()
    if "var_name" in cfg:
        data = data[cfg["var_name"]]
    if "squeeze_dims" in cfg:
        data = data.squeeze(cfg["squeeze_dims"])

if cfg["compute_magnitude"] is not None:
    mag = cfg["compute_magnitude"]
    data[mag["name"]] = np.sqrt(sum(data[c] ** 2 for c in mag["components"]))

# ── Flash flood event times ───────────────────────────────────────────────────

# SED onsets -> UTC (CZ_TIMEZONE), first episode per UTC day; see timeutils.
times_utc = ff_event_hours_utc()
intersect = pd.Index(times_utc).intersection(data.indexes[time_dim])
print(f"Found {len(intersect)} matching events out of {len(times_utc)} total.")

# ── Standardized anomalies ────────────────────────────────────────────────────

if cfg["compute_anomalies"]:
    clim_mean, clim_std = climatology(data, time_dim, mode=args.climatology)
    norm = standardize(data, clim_mean, clim_std, time_dim, mode=args.climatology)
    norm_weighted = norm * lat_weight(norm[lat_dim])
    # Saved so som_preprocessing_evolution.py uses the identical climatology.
    clim_mean.to_netcdf(f"{OUT_DIR}/{prefix}_clim_mean_{args.climatology}.nc")
    clim_std.to_netcdf(f"{OUT_DIR}/{prefix}_clim_std_smooth_{args.climatology}.nc")
else:
    norm = norm_weighted = None

# ── Example plot ──────────────────────────────────────────────────────────────

if not args.no_plots and cfg["plot_config"] is not None:
    pcfg = cfg["plot_config"]

    if "components" in pcfg:
        components = pcfg["components"]
        fig, axs = plt.subplots(
            len(components),
            3,
            figsize=(6.5, 2 * len(components)),
            subplot_kw={"projection": ccrs.PlateCarree()},
            dpi=600,
        )
        for row, (var, label, raw_levels, extend) in enumerate(components):
            for col, (dset, suffix) in enumerate(
                [
                    (data, ""),
                    (norm, " Std. Anomalies"),
                    (norm_weighted, " Std. Anomalies Weighted"),
                ]
            ):
                ax = axs[row, col]
                pd_ = dset[var].isel({time_dim: 0})
                levels = raw_levels if col == 0 else pcfg["std_levels"]
                cmap = "viridis" if (row == 0 and col == 0) else "balance"
                c = ax.contourf(
                    pd_[lon_dim],
                    pd_[lat_dim],
                    pd_,
                    cmap=cmap,
                    levels=levels,
                    extend=extend,
                )
                ax.set_title(f"{label}{suffix}", fontsize=8)
                ax.coastlines(resolution="50m", linewidth=0.5)
                cb = fig.colorbar(c, ax=ax, orientation="horizontal", pad=0.03)
                if col == 0:
                    cb.set_label(pcfg["raw_unit"], fontsize=6)
                cb.set_ticks(levels)
                cb.ax.tick_params(labelsize=6, rotation=45)
    else:
        fig, axs = plt.subplots(
            1,
            3,
            figsize=(6.5, 3.5),
            subplot_kw={"projection": ccrs.PlateCarree()},
            dpi=600,
        )
        raw = data.isel({time_dim: 0})
        if pcfg["raw_transform"] is not None:
            raw = pcfg["raw_transform"](raw)
        panels = [
            (raw, pcfg["title"], pcfg["cmap"], pcfg["raw_levels"]),
            (
                norm.isel({time_dim: 0}),
                f"{pcfg['title']} Std. Anomalies",
                "balance",
                pcfg["std_levels"],
            ),
            (
                norm_weighted.isel({time_dim: 0}),
                f"{pcfg['title']} Std. Anomalies Weighted",
                "balance",
                pcfg["std_levels"],
            ),
        ]
        for i, (pd_, title, cmap, levels) in enumerate(panels):
            c = axs[i].contourf(
                pd_[lon_dim],
                pd_[lat_dim],
                pd_,
                cmap=cmap,
                levels=levels,
                extend=pcfg["extend"],
            )
            axs[i].set_title(title, fontsize=8)
            axs[i].coastlines(resolution="50m", linewidth=0.5)
            cb = fig.colorbar(c, ax=axs[i], orientation="horizontal", pad=0.03)
            if i == 0:
                cb.set_label(pcfg["raw_unit"], fontsize=6)
            cb.set_ticks(c.levels)
            cb.ax.tick_params(labelsize=6, rotation=45)

    plt.tight_layout()
    os.makedirs(pcfg["fig_dir"], exist_ok=True)
    out_fig = f"{pcfg['fig_dir']}/{args.variable}_standardized_anomalies_example.png"
    plt.savefig(out_fig, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_fig}")

# ── Save FFE snapshots ────────────────────────────────────────────────────────

data.sel({time_dim: intersect}).to_netcdf(f"{OUT_DIR}/{prefix}_ffe.nc")
print(f"Saved: {OUT_DIR}/{prefix}_ffe.nc")

if cfg["compute_anomalies"]:
    norm.sel({time_dim: intersect}).to_netcdf(f"{OUT_DIR}/{prefix}_norm_ffe.nc")
    norm_weighted.sel({time_dim: intersect}).to_netcdf(
        f"{OUT_DIR}/{prefix}_norm_weighted_ffe.nc"
    )
    print(f"Saved: {OUT_DIR}/{prefix}_norm_ffe.nc")
    print(f"Saved: {OUT_DIR}/{prefix}_norm_weighted_ffe.nc")

# ── Save daily averages ───────────────────────────────────────────────────────


def daily_mean(da):
    return (
        da.resample({time_dim: "1D"}, label="left", closed="left")
        .mean()
        .dropna(dim=time_dim, how="all")
    )


data_daily = daily_mean(data)
data_daily.to_netcdf(f"{OUT_DIR}/{prefix}_daily.nc")
print(f"Saved: {OUT_DIR}/{prefix}_daily.nc")

if cfg["compute_anomalies"]:
    norm_daily = daily_mean(norm)
    norm_weighted_daily = daily_mean(norm_weighted)
    norm_daily.to_netcdf(f"{OUT_DIR}/{prefix}_norm_daily.nc")
    norm_weighted_daily.to_netcdf(f"{OUT_DIR}/{prefix}_norm_weighted_daily.nc")
    print(f"Saved: {OUT_DIR}/{prefix}_norm_daily.nc")
    print(f"Saved: {OUT_DIR}/{prefix}_norm_weighted_daily.nc")

# ── Save FFE daily averages ───────────────────────────────────────────────────
# Normalize FFE UTC times to midnight and drop duplicates (two events same day → one)
ffe_dates = pd.DatetimeIndex(intersect).normalize().unique()

data_daily.sel({time_dim: ffe_dates}, method="nearest").to_netcdf(
    f"{OUT_DIR}/{prefix}_ffe_daily.nc"
)
print(f"Saved: {OUT_DIR}/{prefix}_ffe_daily.nc")

if cfg["compute_anomalies"]:
    norm_daily.sel({time_dim: ffe_dates}, method="nearest").to_netcdf(
        f"{OUT_DIR}/{prefix}_norm_ffe_daily.nc"
    )
    norm_weighted_daily.sel({time_dim: ffe_dates}, method="nearest").to_netcdf(
        f"{OUT_DIR}/{prefix}_norm_weighted_ffe_daily.nc"
    )
    print(f"Saved: {OUT_DIR}/{prefix}_norm_ffe_daily.nc")
    print(f"Saved: {OUT_DIR}/{prefix}_norm_weighted_ffe_daily.nc")

# ── Save warm-season hourly snapshots ─────────────────────────────────────────
# Select one hour per day from the warm season (May–Oct) for all-days SOM use.
# Controlled by the SNAPSHOT_HOURS list; extend as needed.
SNAPSHOT_HOURS = [] if args.no_snapshots else [20]  # 2000 UTC

warm_mask = data[time_dim].dt.month.isin([5, 6, 7, 8, 9, 10])
for hr in SNAPSHOT_HOURS:
    hour_mask = (data[time_dim].dt.hour == hr) & warm_mask
    snapshot = data.isel({time_dim: hour_mask.values})
    tag = f"_{hr:02d}utc"
    snapshot.to_netcdf(f"{OUT_DIR}/{prefix}_snapshot{tag}.nc")
    print(f"Saved: {OUT_DIR}/{prefix}_snapshot{tag}.nc")
    if cfg["compute_anomalies"]:
        norm.isel({time_dim: hour_mask.values}).to_netcdf(
            f"{OUT_DIR}/{prefix}_norm_snapshot{tag}.nc"
        )
        norm_weighted.isel({time_dim: hour_mask.values}).to_netcdf(
            f"{OUT_DIR}/{prefix}_norm_weighted_snapshot{tag}.nc"
        )
        print(f"Saved: {OUT_DIR}/{prefix}_norm_snapshot{tag}.nc")
        print(f"Saved: {OUT_DIR}/{prefix}_norm_weighted_snapshot{tag}.nc")
