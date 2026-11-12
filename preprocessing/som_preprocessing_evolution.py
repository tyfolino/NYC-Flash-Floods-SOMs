"""
Evolution SOM Preprocessing
By: Ty Janoski

Extracts N consecutive hourly ERA5 snapshots per flash flood event
(window ending at the event hour) and saves them with dimensions
(event_time, hour_offset, lat, lon).

The last hour_offset slice (T+0h) matches the single-snapshot file
produced by som_preprocessing.py exactly.

The standardization uses the climatology saved by som_preprocessing.py
(``{prefix}_clim_mean_{mode}.nc`` / ``{prefix}_clim_std_smooth_{mode}.nc`` in
the same intermediate directory), so the T+0h slice matches its
``*_norm_weighted_ffe.nc`` exactly; if those files are missing the
climatology is recomputed with the same code.

Usage (from the project root; run som_preprocessing first):
    NYCFF_INTERMEDIATE=v2 python -m preprocessing.som_preprocessing_evolution \
        Z500 [--n-hours 24] [--climatology monthday|doy]
    NYCFF_INTERMEDIATE=v2 python -m preprocessing.som_preprocessing_evolution \
        theta_e [--n-hours 48]
Output suffix: _ffe_evsom.nc for 24 h (published name), _ffe_evsom{N}h.nc else.
"""

import argparse
import os
import warnings

import dask
import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics.progress import ProgressBar

from preprocessing.climatology import calendar_key, climatology, lat_weight
from som_analysis.config import (
    ERA5_SLICED_DIR,
    SOM_INTERMEDIATE_PATH,
    check_intermediate_writable,
    evsom_suffix,
)
from som_analysis.timeutils import ff_event_hours_utc

dask.config.set(
    scheduler="threads", num_workers=int(os.environ.get("NYCFF_DASK_WORKERS", "4"))
)

OUT_DIR = SOM_INTERMEDIATE_PATH.rstrip("/")

# ── Variable configuration ─────────────────────────────────────────────────────

CONFIGS = {
    "Z500": {
        "file_path": f"{ERA5_SLICED_DIR}/era5_z500_NWHem_*.nc",
        "load_func": "mfdataset",
        "load_kwargs": {
            "combine": "by_coords",
            "decode_times": True,
            "chunks": {"valid_time": 8760},
        },
        "time_dim": "valid_time",
        "lat_dim": "latitude",
        "lon_dim": "longitude",
        "var_name": "z",
        "squeeze_dims": ["pressure_level"],
        "output_prefix": "era5_Z500",
    },
    "theta_e": {
        "file_path": f"{ERA5_SLICED_DIR}/*thetae*.nc",
        "load_func": "mfdataset",
        "load_kwargs": {
            "combine": "by_coords",
            "decode_times": True,
            "chunks": {"valid_time": 8760},
        },
        "time_dim": "valid_time",
        "lat_dim": "latitude",
        "lon_dim": "longitude",
        "var_name": "theta_e",
        "output_prefix": "era5_thetae",
    },
}

# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("variable", choices=list(CONFIGS.keys()))
parser.add_argument(
    "--n-hours",
    type=int,
    default=24,
    help="Length of the hourly window ending at the event hour (default: 24).",
)
parser.add_argument(
    "--climatology",
    choices=["monthday", "doy"],
    default="monthday",
    help="Calendar grouping (must match som_preprocessing; default monthday).",
)
args = parser.parse_args()
check_intermediate_writable(OUT_DIR)

cfg = CONFIGS[args.variable]
n_hours = args.n_hours
time_dim = cfg["time_dim"]
lat_dim = cfg["lat_dim"]
lon_dim = cfg["lon_dim"]
prefix = cfg["output_prefix"]

# ── Load full hourly data ──────────────────────────────────────────────────────

print(f"Loading {args.variable} (this may take a moment with dask)...")

if cfg["load_func"] == "dataarray":
    data = xr.open_dataarray(cfg["file_path"], **cfg["load_kwargs"])
else:
    data = xr.open_mfdataset(cfg["file_path"], **cfg["load_kwargs"])
    if cfg["var_name"] is not None:
        data = data[cfg["var_name"]]
    if "squeeze_dims" in cfg:
        data = data.squeeze(cfg["squeeze_dims"])

print(f"  Data shape: {dict(data.sizes)}")

# ── Flash flood event times ────────────────────────────────────────────────────

# SED onsets -> UTC (CZ_TIMEZONE), first episode per UTC day; see timeutils.
times_utc = ff_event_hours_utc()
print(f"Flash flood events: {len(times_utc)} total")

# ── Calendar climatology (rolling-smoothed std) ───────────────────────────────

mode = args.climatology
key = "dayofyear" if mode == "doy" else "monthday"
clim_mean_path = f"{OUT_DIR}/{prefix}_clim_mean_{mode}.nc"
clim_std_path = f"{OUT_DIR}/{prefix}_clim_std_smooth_{mode}.nc"
if os.path.exists(clim_mean_path) and os.path.exists(clim_std_path):
    print(f"Using saved {mode} climatology from som_preprocessing.py")
    mean_clim = xr.load_dataarray(clim_mean_path)
    std_clim_smooth = xr.load_dataarray(clim_std_path)
else:
    print(f"Computing {mode} climatology...")
    with ProgressBar():
        mean_clim, std_clim_smooth = climatology(data, time_dim, mode=mode)
        mean_clim, std_clim_smooth = mean_clim.compute(), std_clim_smooth.compute()

# ── Extract N-hour windows per event ──────────────────────────────────────────

data_times = pd.DatetimeIndex(data[time_dim].values)

raw_windows = []
norm_windows = []
norm_weighted_windows = []
valid_event_times = []

for t_event in times_utc:
    # Build window: n_hours slots ending at t_event (inclusive)
    window_times = pd.date_range(end=t_event, periods=n_hours, freq="1h")

    # Check that all window times are present in the data
    missing = [t for t in window_times if t not in data_times]
    if missing:
        warnings.warn(
            f"Skipping event {t_event}: {len(missing)} hours missing from data "
            f"(first missing: {missing[0]})",
            stacklevel=2,
        )
        continue

    # Extract raw window: (n_hours, lat, lon)
    raw_win = data.sel({time_dim: window_times}).values  # numpy via .values

    # Standardized anomaly for each hour using that hour's calendar key
    norm_win = np.empty_like(raw_win)
    for h_idx, t in enumerate(window_times):
        k = calendar_key(t, mode)
        mu = mean_clim.sel({key: k}).values
        sigma = std_clim_smooth.sel({key: k}).values
        norm_win[h_idx] = (raw_win[h_idx] - mu) / sigma

    # Latitude weighting: sqrt(cos(lat)); latitude is the first spatial dim
    weight_broadcast = lat_weight(data[lat_dim].values)[:, np.newaxis]
    norm_weighted_win = norm_win * weight_broadcast

    raw_windows.append(raw_win)
    norm_windows.append(norm_win)
    norm_weighted_windows.append(norm_weighted_win)
    valid_event_times.append(t_event)

n_events = len(valid_event_times)
print(f"Extracted {n_events} complete windows (skipped {len(times_utc) - n_events})")

# ── Assemble xarray DataArrays ─────────────────────────────────────────────────

lat_coords = data[lat_dim].values
lon_coords = data[lon_dim].values
hour_offsets = np.arange(n_hours) - (n_hours - 1)  # T-(n-1), ..., T+0

# Stack arrays: (n_events, n_hours, lat, lon)
raw_arr = np.stack(raw_windows, axis=0)
norm_arr = np.stack(norm_windows, axis=0)
norm_w_arr = np.stack(norm_weighted_windows, axis=0)


def make_da(arr, event_times, hour_offsets, lat_coords, lon_coords):
    return xr.DataArray(
        arr,
        dims=["event_time", "hour_offset", lat_dim, lon_dim],
        coords={
            "event_time": pd.DatetimeIndex(event_times),
            "hour_offset": hour_offsets,
            lat_dim: lat_coords,
            lon_dim: lon_coords,
        },
    )


da_raw = make_da(raw_arr, valid_event_times, hour_offsets, lat_coords, lon_coords)
da_norm = make_da(norm_arr, valid_event_times, hour_offsets, lat_coords, lon_coords)
da_norm_w = make_da(norm_w_arr, valid_event_times, hour_offsets, lat_coords, lon_coords)

# ── Save ───────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

suffix = evsom_suffix(n_hours)

out_raw = f"{OUT_DIR}/{prefix}{suffix}"
out_norm = f"{OUT_DIR}/{prefix}_norm{suffix}"
out_normw = f"{OUT_DIR}/{prefix}_norm_weighted{suffix}"

print(f"Saving {out_raw} ...")
da_raw.to_netcdf(out_raw)

print(f"Saving {out_norm} ...")
da_norm.to_netcdf(out_norm)

print(f"Saving {out_normw} ...")
da_norm_w.to_netcdf(out_normw)

print("\nDone.")
print(f"  Shape: {da_raw.sizes}")
print(f"  Events: {n_events}")
print(f"  Window: T-{n_hours - 1}h to T+0h  (hour_offset={hour_offsets[0]} to 0)")
print(
    "\nVerification hint:"
    "\n  The last hour_offset slice (hour_offset=0) should match"
    f"\n  {OUT_DIR}/{prefix}_norm_weighted_ffe.nc exactly."
)
