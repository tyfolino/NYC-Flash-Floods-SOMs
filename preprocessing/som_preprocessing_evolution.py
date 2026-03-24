"""
Evolution SOM Preprocessing
By: Ty Janoski

Extracts N consecutive hourly ERA5 snapshots per flash flood event
(window ending at the event hour) and saves them with dimensions
(event_time, hour_offset, lat, lon).

The last hour_offset slice (T+0h) matches the single-snapshot file
produced by som_preprocessing.py exactly.

Usage:
    python som_preprocessing_evolution.py Z500 [--n-hours 24]
    python som_preprocessing_evolution.py theta_e [--n-hours 24]
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics.progress import ProgressBar

OUT_DIR = "/mnt/drive2/SOM_intermediate_files"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
STORM_DATA_CSV = os.path.join(_PROJECT_ROOT, "data", "storm_data_search_results.csv")

# ── Variable configuration ─────────────────────────────────────────────────────

CONFIGS = {
    "Z500": {
        "file_path": "/mnt/drive2/ERA5/NC_files/hourly_sliced/era5_z500_NWHem_*.nc",
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
        "file_path": "/mnt/drive2/ERA5/NC_files/hourly_sliced/*thetae*.nc",
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
args = parser.parse_args()

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

# NWS Storm Events Database uses EST (not EDT).
df = pd.read_csv(STORM_DATA_CSV)
df = df[df["EVENT_ID"].astype(str).str.isdigit()]
df["BEGIN_TIME"] = df["BEGIN_TIME"].fillna(0).astype(int).astype(str).str.zfill(4)
df["BEGIN_DATETIME"] = pd.to_datetime(
    df["BEGIN_DATE"] + " " + df["BEGIN_TIME"], format="%m/%d/%Y %H%M", errors="coerce"
)
event_hours = (
    df.drop_duplicates(subset=["EPISODE_ID"], keep="first")["BEGIN_DATETIME"]
    .dt.floor("h")
    .tolist()
)

times_utc = (
    pd.to_datetime(event_hours).tz_localize("EST").tz_convert("UTC").tz_convert(None)
)
print(f"Flash flood events: {len(times_utc)} total")

# ── DOY climatology (rolling-smoothed std) ─────────────────────────────────────

print("Computing DOY climatology...")
with ProgressBar():
    mean_doy = data.groupby(f"{time_dim}.dayofyear").mean(dim=time_dim).compute()
    std_doy = data.groupby(f"{time_dim}.dayofyear").std(dim=time_dim).compute()

std_doy_smooth = (
    std_doy.sortby("dayofyear").rolling(dayofyear=14, center=True, min_periods=7).mean()
)

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

    # Compute standardized anomaly for each hour using that hour's DOY
    norm_win = np.empty_like(raw_win)
    for h_idx, t in enumerate(window_times):
        doy = t.dayofyear
        mu = mean_doy.sel(dayofyear=doy).values
        sigma = std_doy_smooth.sel(dayofyear=doy).values
        norm_win[h_idx] = (raw_win[h_idx] - mu) / sigma

    # Latitude weighting: sqrt(cos(lat))
    lat_vals = data[lat_dim].values
    lat_weight = np.sqrt(np.cos(np.deg2rad(lat_vals)))
    # Broadcast to (n_hours, lat, lon)
    if lat_dim == "lat":
        # lat is the first spatial dim
        weight_broadcast = lat_weight[:, np.newaxis]
    else:
        # latitude dim (also first spatial dim for these files)
        weight_broadcast = lat_weight[:, np.newaxis]
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

suffix = "_ffe_evsom.nc"

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
