"""
Calendar climatology and standardized anomalies for the ERA5 SOM inputs.

Two groupings:
  * "doy"      : day of year (as published, v1 intermediates). In leap years
                 every date from March on gets a different DOY than in other
                 years, so e.g. 31 Oct pools with 30 Oct (audit A20).
  * "monthday" : calendar month-day (v2 intermediates; leap years aligned).

The standard deviation is smoothed with a centred 14-day running mean (min 7
days at the season edges, not circular); in "monthday" mode the mean is
smoothed the same way. Anomalies are (x - mean) / smoothed std.

NB (audit A19): the climatology pools all hours of a calendar day, so hourly
anomalies retain the mean diurnal cycle (<= 0.05 sigma over NYC). A
day-of-year x hour climatology would remove it; not done here.
"""

import numpy as np

KEYS = {"doy": "dayofyear", "monthday": "monthday"}


def _with_key(data, time_dim, mode):
    if mode == "doy":
        return data, f"{time_dim}.dayofyear"
    t = data[time_dim]
    md = (t.dt.month * 100 + t.dt.day).values
    return data.assign_coords(monthday=(time_dim, md)), "monthday"


def climatology(data, time_dim, mode="monthday", window=14, min_periods=7):
    """Return (mean, smoothed std) indexed by the calendar key of ``mode``.

    "doy" (published v1): only the std is smoothed. "monthday" (v2): the mean
    is smoothed with the same centred 14-day window and edge handling as the
    std (non-circular, min 7 days at the season edges), so the anomalies do not
    inherit the ~sigma/sqrt(30) sampling noise of raw 30-year daily means.
    """
    d, grp = _with_key(data, time_dim, mode)
    key = KEYS[mode]

    def smooth(x):
        return (
            x.sortby(key)
            .rolling({key: window}, center=True, min_periods=min_periods)
            .mean()
        )

    mean = d.groupby(grp).mean(dim=time_dim)
    std = d.groupby(grp).std(dim=time_dim)
    if mode == "monthday":
        mean = smooth(mean)
    return mean, smooth(std)


def standardize(data, mean, std_smooth, time_dim, mode="monthday"):
    """Standardized anomalies of ``data`` against a calendar climatology."""
    d, grp = _with_key(data, time_dim, mode)
    anoms = d.groupby(grp) - mean
    norm = anoms.groupby(grp) / std_smooth
    if mode == "monthday":
        norm = norm.drop_vars("monthday", errors="ignore")
    return norm


def calendar_key(t, mode="monthday"):
    """Calendar key of a single pandas Timestamp for ``mode``."""
    return t.dayofyear if mode == "doy" else t.month * 100 + t.day


def lat_weight(lat):
    """sqrt(cos(latitude)) area weighting."""
    return np.sqrt(np.cos(np.deg2rad(lat)))
