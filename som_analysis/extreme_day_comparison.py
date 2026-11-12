"""
D4: are NYC flash-flood (FF) days daily precipitation extremes, or sub-daily
(hourly) extremes?

Agel-style definition (Agel et al. 2015, 2019; see docs/lit_checks.md):
  * wet day / wet hour: >= 0.254 mm (0.01 in)
  * threshold: 99th percentile of wet-day daily totals (or wet-hour amounts),
    computed separately for each ASOS station / Stage IV cell
  * an FF event is a "daily extreme" if any station/cell exceeds its own daily
    threshold on the day containing onset; an "hourly extreme" if any
    station/cell exceeds its own hourly threshold within (t0 - 6 h, t0 + 6 h]
    (hour-ending labels).

Variants: threshold season (year-round vs May-Oct; Stage IV is May-Oct only,
because the Stage IV subset holds the warm season only) and day definition
(UTC day vs local EST day containing onset). Also reported: the hit rate (the
share of extreme days that had an FF report), the climatological frequency of
"extreme days" under each definition, and empirical quantiles of FF rates and
totals (R2 #4: no fitted distribution).

Usage (from the project root):
    OMP_NUM_THREADS=1 python -m som_analysis.extreme_day_comparison

Outputs: results/d4/ (d4_metrics.json, d4_summary.md, two PNGs).
Cache: /mnt/drive2/SOM_d4_cache/stageiv_nyc_box.npz
"""

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from .config import (  # noqa: E402
    DATA_DIR,
    STAGEIV_NC,
    STORM_DATA_CSV,
    filter_nyc_counties,
)
from .node_statistics import load_asos_precip  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "d4")
CACHE = "/mnt/drive2/SOM_d4_cache/stageiv_nyc_box.npz"
EVENTS_CSV = os.path.join(DATA_DIR, "som_2x2_evsom_24h_bmus_thetae.csv")

MM_PER_IN = 25.4
WET_MM = 0.254
PCTL = 99.0
WINDOW_H = 6
# NYC box used for Stage IV in node_statistics.compute_stageiv_max_precip / Fig. S8
NYC_BOX = dict(lat_min=40.5, lat_max=40.9, lon_min=-74.3, lon_max=-73.7)
EST = pd.Timedelta(hours=5)
MIN_VALID_HOURS = 22  # Stage IV: a cell-day needs >= 22 of 24 valid hours


# ── Helpers ───────────────────────────────────────────────────────────────────


def wilson(k, n, z=1.96):
    """Wilson score interval for k successes in n trials (as percentages)."""
    if n == 0:
        return [float("nan")] * 3
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return [100 * p, 100 * (c - h), 100 * (c + h)]


def day_of(labels, local):
    """Calendar day of hour-ending labels (UTC or EST day)."""
    t = labels - pd.Timedelta(minutes=1)
    if local:
        t = t - EST
    return t.floor("D")


def onset_day(t0, local):
    return (t0 - EST).floor("D") if local else t0.floor("D")


def quantiles(x, qs=(10, 50, 75, 95, 99)):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return {f"Q{q}": float(np.percentile(x, q)) for q in qs} | {"n": int(len(x))}


# ── Events ────────────────────────────────────────────────────────────────────


def load_events():
    """The 118 warm-season FF onsets (UTC), cross-checked against the SED."""
    ev = pd.read_csv(EVENTS_CSV, parse_dates=["timestamp"])["timestamp"]
    ev = pd.DatetimeIndex(ev).sort_values()
    sed = pd.read_csv(STORM_DATA_CSV)
    sed = filter_nyc_counties(sed[sed["EVENT_TYPE"].str.contains("Flash", case=False)])
    sed_days = set(pd.to_datetime(sed["BEGIN_DATE"]).dt.normalize())
    # SED dates are local; onset in EST may fall on the SED date or (EDT) next to it
    ok = [
        any((onset_day(t, True) + pd.Timedelta(days=d)) in sed_days for d in (-1, 0, 1))
        for t in ev
    ]
    return ev, int(np.sum(ok))


# ── ASOS ──────────────────────────────────────────────────────────────────────


def asos_hourly_mm(shift_archive=True):
    """Hourly ASOS precipitation (mm), hour-ending, UTC labels.

    The archive segment has a complete hourly grid with NaN for hours without
    precipitation, so NaN -> 0 (dry and missing cannot be told apart). IEM
    special (non-:51) observations are dropped to avoid double counting.
    ``shift_archive=False`` reproduces the pre-fix convention (archive labels
    taken as UTC), for comparison with the preliminary numbers only.
    """
    out = {}
    for name, df in load_asos_precip().items():
        s = df["precip"]
        if not shift_archive:
            arch = s.index.minute == 0
            s = pd.concat([s[arch].set_axis(s.index[arch] - EST), s[~arch]])
            s = s[~s.index.duplicated()].sort_index()
        s = s[(s.index.minute == 0) | (s.index.minute == 51)]
        s = s[(s.index >= "1996-01-01 01:00") & (s.index < "2026-01-01 01:00")]
        out[name] = s.fillna(0.0) * MM_PER_IN
    return out


def asos_tables(hourly, local):
    """Daily totals (days x stations) and hourly values (hours x stations)."""
    daily = {}
    for name, s in hourly.items():
        daily[name] = s.groupby(day_of(s.index, local)).sum()
    daily = pd.DataFrame(daily)
    hours = pd.DataFrame(hourly)
    return daily, hours


# ── Stage IV ──────────────────────────────────────────────────────────────────


def stageiv_nyc():
    """Hourly Stage IV (mm) for the NYC box cells: (times, values[t, cell])."""
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=False)
        return pd.DatetimeIndex(z["times"]), z["values"], z["lat"], z["lon"]
    ds = xr.open_dataset(STAGEIV_NC)
    lat2d, lon2d = ds["latitude"].values, ds["longitude"].values
    b = NYC_BOX
    m = (
        (lat2d >= b["lat_min"])
        & (lat2d <= b["lat_max"])
        & (lon2d >= b["lon_min"])
        & (lon2d <= b["lon_max"])
    )
    r, c = np.where(m)
    ys, xs = slice(r.min(), r.max() + 1), slice(c.min(), c.max() + 1)
    sub_m = m[ys, xs]
    times = pd.DatetimeIndex(ds["time"].values)
    blocks = []
    step = 168 * 26
    for i0 in range(0, len(times), step):
        a = ds["precipitation"].isel(time=slice(i0, i0 + step), y=ys, x=xs).values
        blocks.append(a[:, sub_m].astype(np.float32))
        print(f"  Stage IV {times[min(i0 + step, len(times)) - 1]:%Y-%m}", flush=True)
    vals = np.concatenate(blocks)
    vals[vals < 0] = np.nan
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(
        CACHE,
        times=times.values,
        values=vals,
        lat=lat2d[m],
        lon=lon2d[m],
    )
    return times, vals, lat2d[m], lon2d[m]


def stageiv_tables(times, vals, local):
    """Daily totals (days x cells; NaN if < MIN_VALID_HOURS) and hourly frame."""
    hours = pd.DataFrame(vals, index=times)
    # Full hour-ending grid within each season so missing hours count as missing
    full = []
    for y in sorted(set(times.year)):
        full.append(pd.date_range(f"{y}-05-01 01:00", f"{y}-11-01 00:00", freq="h"))
    hours = hours.reindex(pd.DatetimeIndex(np.concatenate(full)))
    days = day_of(hours.index, local)
    g = hours.groupby(days)
    daily = g.sum(min_count=1)
    nvalid = g.count()
    daily = daily.where(nvalid >= MIN_VALID_HOURS)
    return daily, hours


# ── Core metrics ──────────────────────────────────────────────────────────────


def thresholds(frame, months):
    """Per-column 99th percentile of wet (>= 0.254 mm) values."""
    f = frame[frame.index.month.isin(months)]
    return f.where(f >= WET_MM).quantile(PCTL / 100.0)


def evaluate(events, daily, hours, local, months, warm_days):
    """Metrics for one dataset / day definition / threshold season."""
    thr_d = thresholds(daily, months)
    thr_h = thresholds(hours, months)
    is_ext_day = (daily > thr_d).any(axis=1)
    # Days with any hourly extreme (for the hourly hit rate)
    hour_ext = (hours > thr_h).any(axis=1)
    is_ext_hday = hour_ext.groupby(day_of(hours.index, local)).any()

    rows = []
    for t0 in events:
        d = onset_day(t0, local)
        if d not in daily.index or daily.loc[d].isna().all():
            continue
        lo, hi = t0 - pd.Timedelta(hours=WINDOW_H), t0 + pd.Timedelta(hours=WINDOW_H)
        w = hours[(hours.index > lo) & (hours.index <= hi)]
        if w.empty or w.isna().all().all():
            continue
        rows.append(
            dict(
                t0=t0,
                day=d,
                daily_ext=bool((daily.loc[d] > thr_d).any()),
                hourly_ext=bool((w > thr_h).any().any()),
                max_rate=float(np.nanmax(w.values)),
                max_daily=float(np.nanmax(daily.loc[d].values)),
            )
        )
    ev = pd.DataFrame(rows)
    n = len(ev)
    ff_days = set(ev["day"])

    def rate(mask):
        return wilson(int(mask.sum()), n)

    # Climatological frequency over the warm-season days of the base period
    wd = [d for d in warm_days if d in is_ext_day.index and daily.loc[d].notna().any()]
    ext_days = [d for d in wd if is_ext_day.loc[d]]
    ext_hdays = [d for d in wd if d in is_ext_hday.index and is_ext_hday.loc[d]]
    out = dict(
        n_events=n,
        pct_daily_extreme=rate(ev["daily_ext"]),
        pct_hourly_extreme=rate(ev["hourly_ext"]),
        pct_hourly_not_daily=rate(ev["hourly_ext"] & ~ev["daily_ext"]),
        pct_neither=rate(~ev["hourly_ext"] & ~ev["daily_ext"]),
        n_warm_days=len(wd),
        pct_days_daily_extreme=100 * len(ext_days) / max(len(wd), 1),
        pct_days_hourly_extreme=100 * len(ext_hdays) / max(len(wd), 1),
        hit_rate_daily=wilson(len(ff_days & set(ext_days)), len(ext_days)),
        hit_rate_hourly=wilson(len(ff_days & set(ext_hdays)), len(ext_hdays)),
        base_rate_ff_days=100 * len(ff_days & set(wd)) / max(len(wd), 1),
        threshold_daily_mm=dict(
            median=float(thr_d.median()), min=float(thr_d.min()), max=float(thr_d.max())
        ),
        threshold_hourly_mm=dict(
            median=float(thr_h.median()), min=float(thr_h.min()), max=float(thr_h.max())
        ),
    )
    return out, ev


def rate_shares(ev):
    r = ev["max_rate"].dropna()
    return dict(
        pct_below_25p4=wilson(int((r < 25.4).sum()), len(r)),
        pct_at_or_above_44p45=wilson(int((r >= 44.45).sum()), len(r)),
        q_max_rate_mm_h=quantiles(r),
        q_daily_total_mm=quantiles(ev["max_daily"]),
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def warm_days_between(y0, y1, local):
    days = pd.date_range(f"{y0}-05-01", f"{y1}-10-31", freq="D")
    return [d for d in days if d.month in range(5, 11)]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    events, n_sed_ok = load_events()
    print(f"{len(events)} FF events; {n_sed_ok} matched to an SED date (+/-1 day)")
    res = dict(
        n_events=len(events),
        n_events_matched_sed=n_sed_ok,
        definition=dict(
            wet_mm=WET_MM,
            percentile=PCTL,
            window_h=WINDOW_H,
            nyc_box_stageiv=NYC_BOX,
            asos_stations="JFK, LGA, Central Park, EWR",
            stageiv_min_valid_hours_per_day=MIN_VALID_HOURS,
        ),
    )
    seasons = {"year_round": list(range(1, 13)), "may_oct": list(range(5, 11))}
    keep = {}

    # ASOS (1996-2025)
    hourly = asos_hourly_mm()
    hourly_old = asos_hourly_mm(shift_archive=False)
    for local in (False, True):
        dlab = "est_day" if local else "utc_day"
        daily, hours = asos_tables(hourly, local)
        warm = warm_days_between(1996, 2025, local)
        for slab, months in seasons.items():
            m, ev = evaluate(events, daily, hours, local, months, warm)
            m.update(rate_shares(ev))
            res[f"asos_{dlab}_{slab}"] = m
            keep[("asos", dlab, slab)] = ev
            print(
                f"ASOS {dlab} {slab}: daily {m['pct_daily_extreme'][0]:.0f}% "
                f"hourly {m['pct_hourly_extreme'][0]:.0f}% (n={m['n_events']})"
            )
        # Pre-time-zone-fix comparison (archive labels taken as UTC)
        daily_o, hours_o = asos_tables(hourly_old, local)
        m, ev = evaluate(events, daily_o, hours_o, local, seasons["year_round"], warm)
        m.update(rate_shares(ev))
        res[f"asos_{dlab}_year_round_PRE_TZ_FIX"] = m

    # Stage IV (2002-2025, May-Oct only)
    times, vals, lat, lon = stageiv_nyc()
    res["stageiv_n_cells"] = int(vals.shape[1])
    ev02 = events[events >= "2002-05-01"]
    for local in (False, True):
        dlab = "est_day" if local else "utc_day"
        daily, hours = stageiv_tables(times, vals, local)
        warm = warm_days_between(2002, 2025, local)
        m, ev = evaluate(ev02, daily, hours, local, seasons["may_oct"], warm)
        m.update(rate_shares(ev))
        res[f"stageiv_{dlab}_may_oct"] = m
        keep[("stageiv", dlab, "may_oct")] = ev
        print(
            f"Stage IV {dlab} may_oct: daily {m['pct_daily_extreme'][0]:.0f}% "
            f"hourly {m['pct_hourly_extreme'][0]:.0f}% (n={m['n_events']})"
        )
        # Area-mean variant: NYC-box mean daily total / hourly rate vs own p99
        dm = daily.mean(axis=1, skipna=True).where(daily.notna().sum(axis=1) > 0)
        hm = hours.mean(axis=1, skipna=True)
        m2, _ = evaluate(
            ev02,
            dm.to_frame("box"),
            hm.to_frame("box"),
            local,
            seasons["may_oct"],
            warm,
        )
        res[f"stageiv_{dlab}_may_oct_BOXMEAN"] = m2

    with open(os.path.join(OUT_DIR, "d4_metrics.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    write_summary(res)
    plot(res, keep)
    print(f"Wrote {OUT_DIR}")


# ── Output ────────────────────────────────────────────────────────────────────


def _f(x):
    return f"{x[0]:.0f}% ({x[1]:.0f}–{x[2]:.0f})"


def write_summary(res):
    keys = [
        ("ASOS, UTC day, year-round thr.", "asos_utc_day_year_round"),
        ("ASOS, UTC day, May–Oct thr.", "asos_utc_day_may_oct"),
        ("ASOS, EST day, year-round thr.", "asos_est_day_year_round"),
        ("ASOS, EST day, May–Oct thr.", "asos_est_day_may_oct"),
        ("Stage IV, UTC day, May–Oct thr.", "stageiv_utc_day_may_oct"),
        ("Stage IV, EST day, May–Oct thr.", "stageiv_est_day_may_oct"),
        ("Stage IV box-mean, UTC day", "stageiv_utc_day_may_oct_BOXMEAN"),
        ("Stage IV box-mean, EST day", "stageiv_est_day_may_oct_BOXMEAN"),
        ("ASOS, UTC day, year-round, PRE TZ FIX", "asos_utc_day_year_round_PRE_TZ_FIX"),
    ]
    L = [
        "# D4: flash-flood days vs daily and hourly precipitation extremes",
        "",
        "Generated by `som_analysis/extreme_day_comparison.py`. Agel-style definition:",
        "wet ≥ 0.254 mm; per-station/cell 99th percentile of wet days (wet hours);",
        "an event is extreme if any station/cell exceeds its own threshold on the",
        "onset day (daily) or within (t0−6 h, t0+6 h] (hourly). Wilson 95% CIs.",
        f"Events: {res['n_events']} ({res['n_events_matched_sed']} matched to an",
        f"SED date ±1 day). Stage IV: {res['stageiv_n_cells']} cells in the NYC box",
        "(40.5–40.9°N, 74.3–73.7°W, as Fig. S8), 2002–2025, May–Oct only.",
        "",
        "| Variant | n | FF = daily extreme | FF = hourly extreme | hourly but not daily"
        " | % of warm days daily-ext. | % hourly-ext. | hit rate daily | hit rate hourly"
        " | p99 daily (mm, median) | p99 hourly (mm/h, median) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for lab, k in keys:
        m = res[k]
        L.append(
            f"| {lab} | {m['n_events']} | {_f(m['pct_daily_extreme'])} | "
            f"{_f(m['pct_hourly_extreme'])} | {_f(m['pct_hourly_not_daily'])} | "
            f"{m['pct_days_daily_extreme']:.1f}% | {m['pct_days_hourly_extreme']:.1f}% | "
            f"{_f(m['hit_rate_daily'])} | {_f(m['hit_rate_hourly'])} | "
            f"{m['threshold_daily_mm']['median']:.1f} | "
            f"{m['threshold_hourly_mm']['median']:.1f} |"
        )
    L += ["", "## Empirical quantiles of FF rates and totals (R2 #4)", ""]
    L.append(
        "| Dataset | n | max hourly rate Q10/Q50/Q75/Q95/Q99 (mm/h) | "
        "onset-day total Q10/Q50/Q75/Q95/Q99 (mm) | < 25.4 mm/h | ≥ 44.45 mm/h |"
    )
    L.append("|---|---|---|---|---|---|")
    for lab, k in [
        ("ASOS (UTC day)", "asos_utc_day_year_round"),
        ("Stage IV NYC box (UTC day)", "stageiv_utc_day_may_oct"),
        ("ASOS, PRE TZ FIX", "asos_utc_day_year_round_PRE_TZ_FIX"),
    ]:
        m = res[k]
        qr, qd = m["q_max_rate_mm_h"], m["q_daily_total_mm"]
        qs = ("Q10", "Q50", "Q75", "Q95", "Q99")
        L.append(
            f"| {lab} | {qr['n']} | {' / '.join(f'{qr[q]:.1f}' for q in qs)} | "
            f"{' / '.join(f'{qd[q]:.1f}' for q in qs)} | "
            f"{_f(m['pct_below_25p4'])} | {_f(m['pct_at_or_above_44p45'])} |"
        )
    with open(os.path.join(OUT_DIR, "d4_summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def plot(res, keep):
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    variants = [
        ("ASOS\nyear-round", "asos_utc_day_year_round"),
        ("ASOS\nMay–Oct", "asos_utc_day_may_oct"),
        ("Stage IV\nMay–Oct", "stageiv_utc_day_may_oct"),
        ("Stage IV\nbox-mean", "stageiv_utc_day_may_oct_BOXMEAN"),
    ]
    x = np.arange(len(variants))
    for off, key, col, lab in [
        (-0.18, "pct_daily_extreme", "#9aa5b1", "daily p99 exceeded"),
        (0.18, "pct_hourly_extreme", "#2b6cb0", "hourly p99 exceeded"),
    ]:
        v = np.array([res[k][key] for _, k in variants])
        ax.bar(x + off, v[:, 0], 0.34, color=col, label=lab)
        ax.errorbar(
            x + off,
            v[:, 0],
            yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]],
            fmt="none",
            ecolor="k",
            lw=0.8,
            capsize=2,
        )
    for i, (_, k) in enumerate(variants):
        ax.plot(
            [i - 0.36, i],
            [res[k]["pct_days_daily_extreme"]] * 2,
            color="k",
            ls=":",
            lw=1,
        )
        ax.plot(
            [i, i + 0.36],
            [res[k]["pct_days_hourly_extreme"]] * 2,
            color="k",
            ls=":",
            lw=1,
        )
    ax.set_xticks(x, [v[0] for v in variants])
    ax.set_ylabel("% of flash-flood events (UTC day)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title(
        "FF events reaching the daily vs hourly 99th percentile "
        "(dotted: % of all warm-season days)",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "d4_daily_vs_hourly.png"))
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(8, 3.2), dpi=150)
    for (src, key), col in [
        (("asos", "asos_utc_day_year_round"), "#2b6cb0"),
        (("stageiv", "stageiv_utc_day_may_oct"), "#c05621"),
    ]:
        ev = keep[(src, "utc_day", "year_round" if src == "asos" else "may_oct")]
        for ax, var, thr in [
            (axs[0], "max_rate", res[key]["threshold_hourly_mm"]["median"]),
            (axs[1], "max_daily", res[key]["threshold_daily_mm"]["median"]),
        ]:
            v = np.sort(ev[var].dropna().values)
            ax.step(v, np.arange(1, len(v) + 1) / len(v), color=col, label=src)
            ax.axvline(thr, color=col, ls="--", lw=0.8)
    axs[0].set_xlabel("FF max hourly rate within ±6 h (mm/h)")
    axs[1].set_xlabel("FF onset-day total, wettest station/cell (mm)")
    axs[0].set_ylabel("ECDF")
    axs[0].legend(frameon=False, fontsize=8)
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.suptitle("Dashed: median per-station/cell 99th percentile", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "d4_ecdf.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
