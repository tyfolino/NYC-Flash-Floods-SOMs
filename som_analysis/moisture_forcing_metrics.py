"""
SOM-independent moisture and forcing metrics for NYC flash floods (revision item D3).

For the warm-season flash-flood (FF) events vs. all May-Oct days (or hours) of
1996-2025, this script computes box-mean standardized anomalies of

  * theta_e at 850 hPa, TCWV (precipitable water), and q at 850 hPa over NYC, and
  * Z500 over several upstream boxes (trough location from the evSOM A1/B1 nodes),

and reports exceedance fractions, risk ratios relative to the base rate (with
Wilson 95% intervals), medians, and AUC (probability that an FF case has a
larger moisture anomaly / more negative Z500 anomaly than a random day).

Standardization follows preprocessing/som_preprocessing.py (hourly data, a
climatological mean per calendar day, sigma per calendar day smoothed with a
14-day centered running mean), except that calendar days are keyed by
month-day rather than day-of-year so that leap years align (audit A20).
Anomalies are computed per grid point and then averaged over each box.

The FF day is defined two ways (audit A7): the UTC day of onset with UTC-day
means (as in the manuscript), and the local (EST) day of onset with EST-day
means. The preliminary numbers are also reproduced from the existing
intermediate files.

Usage (from the project root):
    OMP_NUM_THREADS=1 timeout 3600 python -m som_analysis.moisture_forcing_metrics
Outputs: results/d3/{d3_metrics.json, d3_summary.md, d3_distributions.png,
d3_risk_ratios.png}; extracted box series are cached in /mnt/drive2/SOM_d3_cache.
"""

import os

for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_k] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "d3")
CACHE_DIR = "/mnt/drive2/SOM_d3_cache"
INTERMEDIATE = "/mnt/drive2/SOM_intermediate_files"
SLICED = "/mnt/drive2/ERA5/NC_files/hourly_sliced"
YEARS = range(1996, 2026)
NYC_COUNTY_FIPS = {5, 47, 61, 81, 85}

# Boxes: (lat_north, lat_south, lon_west, lon_east)
NYC_BOX = (41.0, 40.5, -74.25, -73.75)
UPSTREAM_BOXES = {
    # Centred on the evSOM A1/B1 Z500 minimum at T0 (~39.5N, 82.75W)
    "trough_T0": (43.0, 36.0, -86.5, -79.0),
    # Centred on the A1/B1 minimum at T-12 (~41N, 87W)
    "trough_Tm12": (44.5, 37.5, -91.0, -83.5),
    "trough_T0_west5": (43.0, 36.0, -91.5, -84.0),
    "trough_T0_north4": (47.0, 40.0, -86.5, -79.0),
    "trough_T0_small": (41.0, 38.0, -84.5, -81.0),
    "trough_T0_large": (47.0, 33.0, -92.0, -76.0),
    # Arbitrary box used for the preliminary numbers
    "prelim": (46.0, 38.0, -85.0, -77.0),
}
Z500_UNION = (47.0, 33.0, -92.0, -76.0)

VARIABLES = {
    "thetae850": dict(pattern=f"{SLICED}/era5_thetae_NWHem_{{y}}.nc", var="theta_e"),
    "tcwv": dict(pattern=f"{SLICED}/era5_tcwv_NWHem_{{y}}.nc", var="tcwv"),
    "q850": dict(
        pattern="/mnt/drive2/ERA5/NC_files/thetae/era5_thetae_NWHem_{y}.nc", var="q"
    ),
    "z500": dict(pattern=f"{SLICED}/era5_z500_NWHem_{{y}}.nc", var="z"),
}
THRESHOLDS = (0.0, 0.5, 1.0)
BINS = (-np.inf, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, np.inf)


# ── Events ────────────────────────────────────────────────────────────────────


def load_event_hours():
    """FF onset hours (UTC), replicating preprocessing/som_preprocessing.py plus
    the agreed county filter (D7). Returns a DatetimeIndex (naive UTC)."""
    df = pd.read_csv(os.path.join(ROOT, "data", "storm_data_search_results.csv"))
    df = df[df["EVENT_ID"].astype(str).str.isdigit()]
    df = df[(df["CZ_TYPE"] == "C") & (df["CZ_FIPS"].isin(NYC_COUNTY_FIPS))]
    df = df[df["EVENT_TYPE"] == "Flash Flood"]
    t = df["BEGIN_TIME"].fillna(0).astype(int).astype(str).str.zfill(4)
    begin = pd.to_datetime(
        df["BEGIN_DATE"] + " " + t, format="%m/%d/%Y %H%M", errors="coerce"
    )
    df = df.assign(BEGIN_DATETIME=begin)
    hours = (
        df.drop_duplicates(subset=["EPISODE_ID"], keep="first")["BEGIN_DATETIME"]
        .dt.floor("h")
        .tolist()
    )
    utc = pd.to_datetime(hours).tz_localize("EST").tz_convert("UTC").tz_convert(None)
    utc = utc[~utc.normalize().duplicated(keep="first")]
    warm = utc[(utc.month >= 5) & (utc.month <= 10) & (utc.year >= 1996)]
    return pd.DatetimeIndex(warm[warm.year <= 2025]).sort_values()


# ── Extraction and standardization ───────────────────────────────────────────


def _extract_year(args):
    name, year, box = args
    spec = VARIABLES[name]
    ds = xr.open_dataset(spec["pattern"].format(y=year))
    da = ds[spec["var"]]
    if "pressure_level" in da.dims:
        da = da.isel(pressure_level=0)
    da = da.sel(latitude=slice(box[0], box[1]), longitude=slice(box[2], box[3]))
    da = da.load()
    ds.close()
    return (
        year,
        da["valid_time"].values,
        da["latitude"].values,
        da["longitude"].values,
        da.values.astype(np.float32),
    )


def extract(name, box, workers):
    """Hourly (time, lat, lon) array over a box for 1996-2025 (cached)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}_{'_'.join(map(str, box))}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["time"], z["lat"], z["lon"], z["data"]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = sorted(ex.map(_extract_year, [(name, y, box) for y in YEARS]))
    time = np.concatenate([p[1] for p in parts])
    data = np.concatenate([p[4] for p in parts], axis=0)
    lat, lon = parts[0][2], parts[0][3]
    np.savez(path, time=time, lat=lat, lon=lon, data=data)
    return time, lat, lon, data


def standardize(time, data):
    """Per-grid-point standardized anomalies keyed by month-day.

    Mean per month-day; sigma per month-day smoothed with a 14-day centred
    running mean (min 7), as in som_preprocessing.py but leap-year safe.
    """
    t = pd.DatetimeIndex(time)
    key = t.month * 100 + t.day
    keys, inv = np.unique(key, return_inverse=True)
    flat = data.reshape(len(t), -1).astype(np.float64)
    n = np.bincount(inv).astype(np.float64)
    order = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(inv))[:-1]]
    s1 = np.add.reduceat(flat[order], starts, axis=0)
    s2 = np.add.reduceat(flat[order] ** 2, starts, axis=0)
    mean = s1 / n[:, None]
    std = np.sqrt(np.maximum(s2 / n[:, None] - mean**2, 0.0))
    std_s = pd.DataFrame(std).rolling(14, center=True, min_periods=7).mean().to_numpy()
    anom = (flat - mean[inv]) / std_s[inv]
    return anom.reshape(data.shape).astype(np.float32)


def box_mean(lat, lon, anom, box):
    la = (lat <= box[0]) & (lat >= box[1])
    lo = (lon >= box[2]) & (lon <= box[3])
    return anom[:, la][:, :, lo].mean(axis=(1, 2))


# ── Statistics ────────────────────────────────────────────────────────────────


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (c - h, c + h)


def auc(ff, allv):
    """P(random FF value > random all-days value) (ties count 1/2)."""
    allv = np.sort(allv)
    lo = np.searchsorted(allv, ff, side="left")
    hi = np.searchsorted(allv, ff, side="right")
    return float(np.mean((lo + 0.5 * (hi - lo)) / len(allv)))


def metrics(series, is_ff, sign=+1):
    """Exceedance fractions, risk ratios, medians and AUC.

    sign=+1: exceedance is anom > t (moisture); sign=-1: anom < -t (troughing).
    """
    x = sign * series
    n_all, n_ff = len(x), int(is_ff.sum())
    base = n_ff / n_all
    out = dict(
        n_ff=n_ff,
        n_all=n_all,
        base_rate=base,
        median_ff=float(np.median(series[is_ff])),
        median_all=float(np.median(series)),
        auc=auc(x[is_ff], x),
        thresholds={},
        bins=[],
    )
    for t in THRESHOLDS:
        above = x > t
        k, n = int((above & is_ff).sum()), int(above.sum())
        lo, hi = wilson(k, n)
        out["thresholds"][f"{t:g}"] = dict(
            frac_ff=k / n_ff,
            frac_all=n / n_all,
            risk_ratio=(k / n) / base if n else np.nan,
            rr_ci95=(lo / base, hi / base),
            n_ff=k,
            n=n,
        )
    for a, b in zip(BINS[:-1], BINS[1:], strict=True):
        inb = (series >= a) & (series < b)
        k, n = int((inb & is_ff).sum()), int(inb.sum())
        lo, hi = wilson(k, n)
        out["bins"].append(
            dict(
                lo=float(a),
                hi=float(b),
                n=n,
                n_ff=k,
                risk_ratio=(k / n) / base if n else np.nan,
                rr_ci95=(lo / base, hi / base),
            )
        )
    return out


# ── Reproduction of the preliminary numbers ──────────────────────────────────


def reproduce_preliminary():
    def box(f, b):
        ds = xr.open_dataset(os.path.join(INTERMEDIATE, f))
        v = list(ds.data_vars)[0]
        return (
            ds[v]
            .sel(latitude=slice(b[0], b[1]), longitude=slice(b[2], b[3]))
            .mean(["latitude", "longitude"])
            .load()
            .values
        )

    th_ff = box("era5_thetae_norm_ffe_daily.nc", NYC_BOX)
    th_all = box("era5_thetae_norm_daily.nc", NYC_BOX)
    th_h = box("era5_thetae_norm_ffe.nc", NYC_BOX)
    up = UPSTREAM_BOXES["prelim"]
    z_ff = box("era5_Z500_norm_ffe_daily.nc", up)
    z_all = box("era5_Z500_norm_daily.nc", up)
    base = len(th_ff) / len(th_all)
    return dict(
        n_ff=len(th_ff),
        n_all=len(th_all),
        thetae_daily_gt0=(float((th_ff > 0).mean()), float((th_all > 0).mean())),
        thetae_daily_gt1=(float((th_ff > 1).mean()), float((th_all > 1).mean())),
        thetae_hourlyT0_gt0=float((th_h > 0).mean()),
        thetae_hourlyT0_gt1=float((th_h > 1).mean()),
        thetae_daily_rr_gt1=float(((th_ff > 1).sum() / (th_all > 1).sum()) / base),
        z500_prelim_lt0=(float((z_ff < 0).mean()), float((z_all < 0).mean())),
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def daily_means(time, series, local=False):
    s = pd.Series(series, index=pd.DatetimeIndex(time))
    if local:  # EST day = 05 UTC to 05 UTC
        s.index = s.index - pd.Timedelta(hours=5)
    g = s.groupby(s.index.normalize())
    d = g.mean()
    return d[g.count() == 24]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.workers > 4:
        raise SystemExit("Use at most 4 workers (shared server).")
    os.makedirs(OUT_DIR, exist_ok=True)

    events = load_event_hours()
    print(f"FF events (warm season, county-filtered): {len(events)}", flush=True)

    results = dict(events=dict(n=len(events)), preliminary=reproduce_preliminary())
    print("Preliminary reproduction:", results["preliminary"], flush=True)

    # Box-mean standardized anomaly series (hourly)
    series = {}
    for name in ("thetae850", "tcwv", "q850"):
        time, lat, lon, data = extract(name, NYC_BOX, args.workers)
        series[name] = box_mean(lat, lon, standardize(time, data), NYC_BOX)
        print(f"{name}: {data.shape}", flush=True)
    time_z, lat, lon, data = extract("z500", Z500_UNION, args.workers)
    assert np.array_equal(time_z, time), "time axes differ between variables"
    anom = standardize(time_z, data)
    del data
    for b, box in UPSTREAM_BOXES.items():
        series[f"z500_{b}"] = box_mean(lat, lon, anom, box)
    del anom

    t_idx = pd.DatetimeIndex(time)
    missing = events.difference(t_idx)
    assert len(missing) == 0, f"events missing from ERA5 hours: {missing}"
    is_ff_hour = t_idx.isin(events)

    ff_day_utc = events.normalize().unique()
    ff_day_est = (events - pd.Timedelta(hours=5)).normalize().unique()

    out = {}
    for name, s in series.items():
        sign = -1 if name.startswith("z500") else +1
        res = dict(hourly_T0=metrics(s, is_ff_hour, sign))
        for lab, local, ffd in (
            ("daily_utc", False, ff_day_utc),
            ("daily_est", True, ff_day_est),
        ):
            d = daily_means(time, s, local=local)
            is_ff = d.index.isin(ffd)
            res[lab] = metrics(d.to_numpy(), is_ff, sign)
            res[lab]["n_ff_days_lost_to_incomplete_days"] = int(len(ffd) - is_ff.sum())
        out[name] = res
    results["metrics"] = out

    # Joint moisture/forcing: theta_e > 0 and Z500(trough_T0) < 0 at the FF day
    d_th = daily_means(time, series["thetae850"])
    d_z = daily_means(time, series["z500_trough_T0"])
    ff = d_th.index.isin(ff_day_utc)
    joint = {}
    for lab, m in (
        ("moist_and_trough", (d_th > 0) & (d_z < 0)),
        ("moist_no_trough", (d_th > 0) & (d_z >= 0)),
        ("dry_and_trough", (d_th <= 0) & (d_z < 0)),
        ("dry_no_trough", (d_th <= 0) & (d_z >= 0)),
    ):
        m = m.to_numpy()
        joint[lab] = dict(
            frac_ff=float(m[ff].mean()),
            frac_all=float(m.mean()),
            risk_ratio=float((m & ff).sum() / m.sum() / ff.mean()),
        )
    results["joint_daily_utc_thetae_vs_z500_trough_T0"] = joint

    with open(os.path.join(OUT_DIR, "d3_metrics.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)
    write_summary(results)
    plot(time, series, is_ff_hour, ff_day_utc)
    print(f"Wrote {OUT_DIR}", flush=True)


def write_summary(r):
    m = r["metrics"]
    lines = [
        "# D3 moisture and forcing metrics",
        "",
        f"FF events: {r['events']['n']} (warm season 1996-2025, county-filtered).",
        "Standardized anomalies by month-day climatology; box means. Exceedance:",
        "moisture anom > t; Z500 anom < -t. RR = risk ratio vs base rate (Wilson 95%).",
        "",
        "## Preliminary numbers reproduced from intermediate files",
        "",
        "```",
        json.dumps(r["preliminary"], indent=1),
        "```",
        "",
    ]
    for scale in ("daily_utc", "daily_est", "hourly_T0"):
        lines += [
            f"## {scale}",
            "",
            "| variable | median FF / all | AUC | >0: FF / all | >0.5: FF / all "
            "| >1: FF / all | RR >1 (95% CI) |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, res in m.items():
            x = res[scale]
            th = x["thresholds"]
            cells = [
                f"{100 * th[t]['frac_ff']:.0f}% / {100 * th[t]['frac_all']:.0f}%"
                for t in ("0", "0.5", "1")
            ]
            rr = th["1"]
            lines.append(
                f"| {name} | {x['median_ff']:+.2f} / {x['median_all']:+.2f} "
                f"| {x['auc']:.2f} | "
                + " | ".join(cells)
                + f" | {rr['risk_ratio']:.2f} "
                f"({rr['rr_ci95'][0]:.2f}-{rr['rr_ci95'][1]:.2f}) |"
            )
        lines.append("")
    lines += ["## Joint (daily UTC): theta_e NYC vs Z500 trough_T0", ""]
    lines += ["| class | FF | all | RR |", "|---|---|---|---|"]
    for k, v in r["joint_daily_utc_thetae_vs_z500_trough_T0"].items():
        lines.append(
            f"| {k} | {100 * v['frac_ff']:.0f}% | {100 * v['frac_all']:.0f}% "
            f"| {v['risk_ratio']:.2f} |"
        )
    with open(os.path.join(OUT_DIR, "d3_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def plot(time, series, is_ff_hour, ff_day_utc):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["thetae850", "tcwv", "q850", "z500_trough_T0", "z500_prelim"]
    fig, axs = plt.subplots(1, len(names), figsize=(15, 3), sharey=True)
    for ax, name in zip(axs, names, strict=True):
        d = daily_means(time, series[name])
        ff = d.index.isin(ff_day_utc)
        bins = np.arange(-4, 4.01, 0.25)
        ax.hist(d[~ff], bins=bins, density=True, alpha=0.5, label="all days")
        ax.hist(d[ff], bins=bins, density=True, alpha=0.6, label="FF days")
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("standardized anomaly (daily UTC)")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "d3_distributions.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(OUT_DIR, "d3_metrics.json")) as f:
        m = json.load(f)["metrics"]
    fig, ax = plt.subplots(figsize=(7, 4))
    centers = np.array([-1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75])
    for k, name in enumerate(names):
        b = m[name]["daily_utc"]["bins"]
        rr = np.array([x["risk_ratio"] for x in b])
        lo = np.array([x["rr_ci95"][0] for x in b])
        hi = np.array([x["rr_ci95"][1] for x in b])
        xs = centers + (k - 2) * 0.05
        ax.errorbar(
            xs, rr, yerr=[rr - lo, hi - rr], marker="o", ms=3, capsize=2, label=name
        )
    ax.axhline(1, color="k", lw=0.5)
    ax.set_xticks(
        centers, ["<-1", "-1..-.5", "-.5..0", "0..0.5", ".5..1", "1..1.5", ">1.5"]
    )
    ax.set_xlabel("standardized anomaly bin (daily UTC)")
    ax.set_ylabel("FF risk ratio vs base rate")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "d3_risk_ratios.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
