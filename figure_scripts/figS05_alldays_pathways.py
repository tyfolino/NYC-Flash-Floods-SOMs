"""
Figure S5 — evSOM events traced through the all-days SOM

For each evSOM node (panels), the all-days SOM node of every event on the UTC
days d−2, d−1, d0 (48-h window), where d0 is the event's flash-flood day
(timeutils.ff_day: the UTC date of onset − 12 h). Arrows: day-to-day moves
shared by >= 2 events (width ∝ number of events, light → dark = earlier →
later step); faint lines: one-event moves;
rings: number of events in each all-days node on d0 (the counts of the former
evSOM → all-days mapping figure). Shading: all-days node flash-flood risk
relative to climatology; outlined nodes are >= 2×.

Reads the trained SOMs under OUTPUT_ROOT (all-days cache, evSOM BMU CSV for
NYCFF_EVSOM_HOURS) and the daily time axis of the all-days training data.

Usage:
    python -m figure_scripts.figS05_alldays_pathways
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

from som_analysis.config import (
    ALLDAYS_CACHE,
    DATA_DIR,
    EVSOM_HOURS,
    FIGURES_DIR,
    SOM_INTERMEDIATE_PATH,
    alldays_grid,
)
from som_analysis.evsom_phase2 import plot_pathways
from som_analysis.timeutils import ff_day

OUT_DIR = os.path.join(FIGURES_DIR, "figS05")
os.makedirs(OUT_DIR, exist_ok=True)

EVSOM_CSV = os.path.join(DATA_DIR, f"som_2x2_evsom_{EVSOM_HOURS}h_bmus_thetae.csv")
# Same file (and so the same day order) the all-days SOM was trained on.
DAILY_NC = f"{SOM_INTERMEDIATE_PATH}era5_Z500_norm_weighted_daily.nc"


def main():
    xd, yd = alldays_grid()
    cz = np.load(ALLDAYS_CACHE)
    bmus = cz["bmus"]  # (n_days, 2), oriented
    a_lab = bmus[:, 0] * yd + bmus[:, 1]
    is_ff = cz["is_ff"].astype(bool)
    risk = (cz["risk"] / is_ff.mean()).reshape(-1)  # node = i * yd + j

    with xr.open_dataarray(DAILY_NC) as da:
        days = pd.DatetimeIndex(da[da.dims[0]].values).normalize()
    if len(days) != len(a_lab):
        raise SystemExit(f"{DAILY_NC}: {len(days)} days, cache has {len(a_lab)}")
    day_idx = {d: k for k, d in enumerate(days)}

    ev = pd.read_csv(EVSOM_CSV, parse_dates=["timestamp"])
    e_lab = (ev["node_i"] * 2 + ev["node_j"]).to_numpy()
    lags = list(range(-(EVSOM_HOURS // 24), 1))
    path = np.full((len(ev), len(lags)), -1)
    for e, t in enumerate(ff_day(ev["timestamp"])):  # d0 = flash-flood day
        for c, lag in enumerate(lags):
            k = day_idx.get(t + pd.Timedelta(days=lag))
            if k is not None:
                path[e, c] = a_lab[k]
    print(f"{len(ev)} events, {(path >= 0).all(1).sum()} with every window day")

    plot_pathways(
        path,
        e_lab,
        risk,
        xd,
        yd,
        lags,
        os.path.join(OUT_DIR, "figS05_alldays_pathways"),
        EVSOM_HOURS,
        suptitle=False,
        exts=("pdf", "png", "tiff"),
    )
    print(f"Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
