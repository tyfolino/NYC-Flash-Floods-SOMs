"""
UTC handling for the NYC flash flood analysis, in one place.

Conventions (all returned timestamps are timezone-naive UTC):

* Storm Events Database (SED): BEGIN_DATE/BEGIN_TIME are local *standard*
  time as given by CZ_TIMEZONE (e.g. "EST" or "EST-5"; NWS does not use
  daylight time in SED). Converted to UTC with the fixed offset.
* ASOS hourly precipitation (``load_asos_precip``): archive segment is
  hour-ending EST and is shifted +5 h; the IEM segment is already UTC.
* Stage IV, ASOS and ERA5 precipitation are hour-ending: the value labelled
  t covers (t - 1 h, t]. ``hour_ending_window`` selects the accumulation
  hours inside [t0 - before, t0 + after], i.e. labels in
  (t0 - before, t0 + after].
"""

import re

import numpy as np
import pandas as pd

from .config import STATION_NPY_FILES, STORM_DATA_CSV, filter_nyc_counties

# Named SED time zones -> UTC offset (hours). "XST-N" forms are parsed directly.
_SED_TZ_OFFSETS = {"EST": -5, "EDT": -4, "CST": -6, "CDT": -5, "AST": -4}


def _utc_offset_hours(tz):
    tz = str(tz).strip().upper()
    m = re.fullmatch(r"[A-Z]{3}-(\d{1,2})", tz)
    if m:
        return -int(m.group(1))
    if tz in _SED_TZ_OFFSETS:
        return _SED_TZ_OFFSETS[tz]
    raise ValueError(f"Unknown SED CZ_TIMEZONE: {tz!r}")


def load_sed(storm_csv=STORM_DATA_CSV, nyc_only=True):
    """Storm Events rows (numeric EVENT_ID; five NYC counties by default) with
    a ``BEGIN_UTC`` column (timezone-naive UTC, minute precision)."""
    df = pd.read_csv(storm_csv)
    df = df[df["EVENT_ID"].astype(str).str.isdigit()]
    if nyc_only:
        df = filter_nyc_counties(df)
    df = df.copy()
    hhmm = df["BEGIN_TIME"].fillna(0).astype(int).astype(str).str.zfill(4)
    local = pd.to_datetime(
        df["BEGIN_DATE"] + " " + hhmm, format="%m/%d/%Y %H%M", errors="coerce"
    )
    offset = pd.to_timedelta(df["CZ_TIMEZONE"].map(_utc_offset_hours), unit="h")
    df["BEGIN_UTC"] = local - offset
    return df


def sed_episode_onsets_utc(storm_csv=STORM_DATA_CSV, floor="h"):
    """One onset per SED episode: the first row of each EPISODE_ID in file
    order (as in the published analysis), in UTC, floored to ``floor``
    (None = no flooring). Returns a DatetimeIndex in file order."""
    df = load_sed(storm_csv).drop_duplicates(subset=["EPISODE_ID"], keep="first")
    t = pd.DatetimeIndex(df["BEGIN_UTC"])
    return t.floor(floor) if floor else t


def ff_event_hours_utc(storm_csv=STORM_DATA_CSV):
    """Flash-flood event hours used by the SOMs: episode onsets floored to the
    hour, keeping only the first episode per UTC calendar day (file order).

    NB: this does not guarantee non-overlapping windows; episodes on
    consecutive UTC days can still overlap (e.g., 2019-07-22/23 by 11 h).
    """
    t = sed_episode_onsets_utc(storm_csv, floor="h")
    return t[~t.normalize().duplicated(keep="first")]


def ff_days_utc(storm_csv=STORM_DATA_CSV):
    """Sorted unique UTC calendar days with at least one flash-flood episode."""
    t = sed_episode_onsets_utc(storm_csv, floor=None)
    return sorted(t.dropna().floor("D").unique())


def hour_ending_window(times, t0, hours_before, hours_after=None):
    """Boolean mask of hour-ending ``times`` whose accumulation hour lies
    inside [t0 - hours_before, t0 + hours_after] (labels in the half-open
    interval (t0 - before, t0 + after])."""
    if hours_after is None:
        hours_after = hours_before
    times = pd.DatetimeIndex(times)
    return (times > t0 - pd.Timedelta(hours=hours_before)) & (
        times <= t0 + pd.Timedelta(hours=hours_after)
    )


def load_asos_precip():
    """Load ASOS hourly precipitation, with every timestamp in UTC.

    The .npy files have two segments with different time conventions:
      * archive segment (timestamps on the hour, through 2025-07-14 22:00):
        hour-ending local standard time (EST, no DST). Lag correlation with
        Stage IV peaks at +5 h (r ~ 0.93), so these are shifted +5 h to UTC.
      * IEM segment appended by fetch_asos_update.py (timestamps at :51,
        from 2025-07-15 00:51): already UTC (requested with tz=UTC).
    Where the two segments overlap after conversion, the archive values are
    kept.
    """
    precip_dfs = {}
    for name, fpath in STATION_NPY_FILES.items():
        arr = np.load(fpath, allow_pickle=True)
        df = pd.DataFrame(arr, columns=["precip", "time"])
        df["time"] = pd.to_datetime(df["time"])
        df["precip"] = pd.to_numeric(df["precip"], errors="coerce")
        archive = df["time"].dt.minute == 0
        old = df[archive].copy()
        old["time"] = old["time"] + pd.Timedelta(hours=5)  # EST -> UTC
        new = df[~archive]
        new = new[new["time"] > old["time"].max()]
        df = pd.concat([old, new]).set_index("time").sort_index()
        precip_dfs[name] = df
    return precip_dfs
