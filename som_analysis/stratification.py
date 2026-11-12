"""
Stratification of NYC flash-flood (FF) events by regional rain area and by peak
rain rate (revision item D2; Reviewer 1, M3).

For each warm-season FF event (evSOM BMU list, 118 events; Stage IV subset
2002-2025, n ~ 100), this script computes

  * the Stage IV regional rain area: cells (and km^2) of the Fig. 4 regional
    domain (38.5-44N, 78-70W; open ocean masked with
    node_statistics.coastal_land_mask) whose maximum hourly rate in the
    hour-ending window (t0 - 6 h, t0 + 6 h] reaches 10, 25 and 50 mm/h;
  * the peak hourly rate over NYC from Stage IV (NYC box of Fig. S8,
    40.5-40.9N, 74.3-73.7W, unmasked as in Fig. S8) and from ASOS (UTC loader).

Events are split into widespread vs localized (median split of the area at each
threshold; top vs bottom tercile as a sensitivity test) and heavy vs modest
(median split and 25.4 mm/h, for Stage IV and ASOS). For each split it compares

  (a) the evSOM, static-SOM and all-days-SOM node mix (chi-square statistic
      with a Monte Carlo permutation p-value);
  (b) the D3 metrics: NYC theta_e,850 and TCWV anomalies and the upstream Z500
      anomaly in the evSOM A1/B1 trough box, at onset (T0) and as UTC-day means
      (Mann-Whitney U, rank-biserial effect size);
  (c) whether "moisture nearly always, troughing sometimes" holds in each group.

The BMU CSVs and outputs follow som_analysis.config (NYCFF_OUTPUT_ROOT), so the
script reruns unchanged on the final SOMs. The D3 box series are read from the
D3 cache (/mnt/drive2/SOM_d3_cache) and standardized exactly as in D3
(month-day climatology). The Stage IV NYC peak is cross-checked against the D4
NYC extract (/mnt/drive2/SOM_d4_cache) when it exists.

Usage (from the project root):
    OMP_NUM_THREADS=1 timeout 3600 python -m som_analysis.stratification
Outputs: <RESULTS_DIR>/d2/{d2_metrics.json, d2_events.csv, d2_summary.md,
d2_stratification.png}
"""

import os

for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_k] = "1"

import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from scipy.stats import mannwhitneyu, spearmanr  # noqa: E402

from som_analysis.config import (  # noqa: E402
    RESULTS_DIR,
    STAGEIV_NC,
    get_alldays_paths,
    get_evsom_paths,
    get_paths,
)
from som_analysis.moisture_forcing_metrics import (  # noqa: E402
    CACHE_DIR as D3_CACHE_DIR,
)
from som_analysis.moisture_forcing_metrics import (  # noqa: E402
    NYC_BOX as D3_NYC_BOX,
)
from som_analysis.moisture_forcing_metrics import (  # noqa: E402
    UPSTREAM_BOXES,
    Z500_UNION,
    box_mean,
    daily_means,
    standardize,
    wilson,
)
from som_analysis.node_statistics import (  # noqa: E402
    coastal_land_mask,
    compute_asos_max_precip,
)
from som_analysis.timeutils import hour_ending_window, load_asos_precip  # noqa: E402

OUT_DIR = os.path.join(RESULTS_DIR, "d2")
D4_CACHE = "/mnt/drive2/SOM_d4_cache/stageiv_nyc_box.npz"
WINDOW_H = 6
AREA_THRESHOLDS = (10, 25, 50)  # mm/h
PRIMARY_AREA = 25
REG = dict(lat_min=38.5, lat_max=44.0, lon_min=-78.0, lon_max=-70.0)
NYC = dict(lat_min=40.5, lat_max=40.9, lon_min=-74.3, lon_max=-73.7)
TROUGH_BOX = "trough_T0"  # evSOM A1/B1 Z500 minimum at T0 (D3)
IN_TO_MM = 25.4
N_PERM = 20000
RNG = np.random.default_rng(0)


# ── Events and SOM nodes ─────────────────────────────────────────────────────


def load_events():
    """Events from the evSOM BMU CSV, joined to the static and all-days nodes."""
    ev = pd.read_csv(get_evsom_paths("thetae")["bmu_csv_path"])
    ev["timestamp"] = pd.to_datetime(ev["timestamp"])
    ev = ev.rename(columns={"node": "evsom"})[["timestamp", "evsom"]]
    st = pd.read_csv(get_paths("thetae")["bmu_csv_path"])
    st["timestamp"] = pd.to_datetime(st["timestamp"])
    ev = ev.merge(
        st.rename(columns={"node": "static"})[["timestamp", "static"]],
        on="timestamp",
        how="left",
        validate="one_to_one",
    )
    ad = pd.read_csv(get_alldays_paths("thetae")["bmu_csv_path"])
    ad["day"] = pd.to_datetime(ad["timestamp"]).dt.normalize()
    ev["day"] = ev["timestamp"].dt.normalize()
    ev = ev.merge(
        ad.rename(columns={"node": "alldays"})[["day", "alldays"]],
        on="day",
        how="left",
        validate="one_to_one",
    )
    for c in ("static", "alldays"):
        if ev[c].isna().any():
            raise ValueError(f"{c} BMUs missing for some evSOM events")
    return ev.sort_values("timestamp").reset_index(drop=True)


# ── Stage IV ─────────────────────────────────────────────────────────────────


def cell_area_km2(lat2d, lon2d):
    """Approximate cell areas (km^2) from the 2-D coordinates (HRAP ~4 km)."""
    dlat_y, dlat_x = np.gradient(lat2d)
    dlon_y, dlon_x = np.gradient(lon2d)
    kx = 111.32 * np.cos(np.deg2rad(lat2d))
    ky = 110.57
    ax, ay = dlon_x * kx, dlat_x * ky
    bx, by = dlon_y * kx, dlat_y * ky
    return np.abs(ax * by - ay * bx)


def stageiv_event_metrics(events):
    """Regional rain area per threshold and the NYC-box peak rate (mm/h)."""
    ds = xr.open_dataset(STAGEIV_NC)
    times = pd.DatetimeIndex(ds["time"].values)
    lat2d = ds["latitude"].values
    lon2d = ds["longitude"].values
    reg = (
        (lat2d >= REG["lat_min"])
        & (lat2d <= REG["lat_max"])
        & (lon2d >= REG["lon_min"])
        & (lon2d <= REG["lon_max"])
    )
    r, c = np.where(reg)
    ys, xs = slice(r.min(), r.max() + 1), slice(c.min(), c.max() + 1)
    lat_r, lon_r = lat2d[ys, xs], lon2d[ys, xs]
    valid = reg[ys, xs] & coastal_land_mask(lat_r, lon_r)
    area = cell_area_km2(lat_r, lon_r)
    nyc = (
        (lat_r >= NYC["lat_min"])
        & (lat_r <= NYC["lat_max"])
        & (lon_r >= NYC["lon_min"])
        & (lon_r <= NYC["lon_max"])
    )
    print(
        f"Regional cells: {int(valid.sum())} valid (land/coastal); "
        f"median cell area {np.median(area[valid]):.1f} km^2; NYC cells {int(nyc.sum())}",
        flush=True,
    )

    rows = []
    for t0 in events["timestamp"]:
        idx = np.where(hour_ending_window(times, t0, WINDOW_H))[0]
        rec = {"timestamp": t0, "s4_n_hours": len(idx)}
        if len(idx) < 2 * WINDOW_H:  # outside Stage IV coverage
            rows.append(rec)
            continue
        p = (
            ds["precipitation"]
            .isel(time=slice(idx[0], idx[-1] + 1), y=ys, x=xs)
            .values.astype(float)
        )
        p[p < 0] = np.nan
        with np.errstate(all="ignore"):
            cmax = np.nanmax(p, axis=0)
        rec["s4_nyc_peak_mm"] = float(np.nanmax(cmax[nyc]))
        rec["s4_region_peak_mm"] = float(np.nanmax(np.where(valid, cmax, np.nan)))
        for thr in AREA_THRESHOLDS:
            hit = valid & (cmax >= thr)
            rec[f"cells_ge{thr}"] = int(hit.sum())
            rec[f"area_ge{thr}_km2"] = float(area[hit].sum())
        rows.append(rec)
    ds.close()
    return pd.DataFrame(rows)


def check_against_d4(df):
    """Cross-check the NYC peak against the D4 extract (same file and box)."""
    if not os.path.exists(D4_CACHE):
        return None
    z = np.load(D4_CACHE)
    t = pd.DatetimeIndex(z["times"])
    v = z["values"]
    diffs = []
    for _, row in df.dropna(subset=["s4_nyc_peak_mm"]).iterrows():
        m = hour_ending_window(t, row["timestamp"], WINDOW_H)
        if m.sum() == 0:
            continue
        with np.errstate(all="ignore"):
            d4 = float(np.nanmax(v[m]))
        diffs.append(abs(d4 - row["s4_nyc_peak_mm"]))
    return float(np.max(diffs)) if diffs else None


# ── D3 anomalies ─────────────────────────────────────────────────────────────


def d3_series(events):
    """NYC theta_e / TCWV and trough-box Z500 anomalies at T0 and UTC-day means."""
    out = pd.DataFrame({"timestamp": events["timestamp"]})
    specs = {
        "thetae": ("thetae850", D3_NYC_BOX, D3_NYC_BOX),
        "tcwv": ("tcwv", D3_NYC_BOX, D3_NYC_BOX),
        "z500": ("z500", Z500_UNION, UPSTREAM_BOXES[TROUGH_BOX]),
    }
    for name, (cache, cache_box, box) in specs.items():
        z = np.load(
            os.path.join(D3_CACHE_DIR, f"{cache}_{'_'.join(map(str, cache_box))}.npz")
        )
        time, lat, lon, data = z["time"], z["lat"], z["lon"], z["data"]
        # Subset to the target box before standardizing (anomalies are per grid
        # point, so this is identical to D3 and much cheaper for Z500).
        la = (lat <= box[0]) & (lat >= box[1])
        lo = (lon >= box[2]) & (lon <= box[3])
        data = data[:, la][:, :, lo]
        series = box_mean(lat[la], lon[lo], standardize(time, data), box)
        s = pd.Series(series, index=pd.DatetimeIndex(time))
        out[f"{name}_T0"] = s.reindex(events["timestamp"]).to_numpy()
        d = daily_means(time, series)
        out[f"{name}_day"] = d.reindex(events["timestamp"].dt.normalize()).to_numpy()
    return out


# ── Statistics ───────────────────────────────────────────────────────────────


def _chi2(gc, nc, ng, nn):
    tab = np.bincount(gc * nn + nc, minlength=ng * nn).reshape(ng, nn).astype(float)
    exp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nansum((tab - exp) ** 2 / exp))


def node_mix_test(groups, nodes):
    """Chi-square statistic and Monte Carlo permutation p-value (group labels
    permuted; appropriate for the small, sparse node counts)."""
    g = np.asarray(groups)
    gc, gu = pd.factorize(g)
    nc, nu = pd.factorize(np.asarray(nodes))
    obs = _chi2(gc, nc, len(gu), len(nu))
    perm = np.array(
        [_chi2(RNG.permutation(gc), nc, len(gu), len(nu)) for _ in range(N_PERM)]
    )
    p = (1 + np.sum(perm >= obs - 1e-12)) / (N_PERM + 1)
    tab = pd.crosstab(pd.Series(g, name="group"), pd.Series(nodes, name="node"))
    return dict(chi2=obs, p_mc=float(p), table=tab.to_dict(orient="index"))


def mwu(a, b):
    """Mann-Whitney U (two-sided) and rank-biserial r (positive: a > b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return dict(p=np.nan, r_rb=np.nan, med_a=np.nan, med_b=np.nan)
    u = mannwhitneyu(a, b, alternative="two-sided")
    return dict(
        p=float(u.pvalue),
        r_rb=float(2 * u.statistic / (len(a) * len(b)) - 1),
        med_a=float(np.median(a)),
        med_b=float(np.median(b)),
    )


def frac(mask):
    mask = np.asarray(mask, bool)
    k, n = int(mask.sum()), len(mask)
    lo, hi = wilson(k, n)
    return dict(k=k, n=n, pct=100 * k / n if n else np.nan, ci=[100 * lo, 100 * hi])


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


# ── Stratifications ──────────────────────────────────────────────────────────


def build_splits(df):
    """Boolean 'high' membership (widespread / heavy) per split; NaN = excluded."""
    splits = {}
    for thr in AREA_THRESHOLDS:
        a = df[f"area_ge{thr}_km2"]
        med = a.median()
        splits[f"area{thr}_median"] = dict(
            high=a > med,
            valid=a.notna(),
            labels=("localized", "widespread"),
            rule=f"area(>= {thr} mm/h) > median ({med:.0f} km^2)",
        )
        q1, q2 = a.quantile([1 / 3, 2 / 3])
        splits[f"area{thr}_tercile"] = dict(
            high=a > q2,
            valid=a.notna() & ((a <= q1) | (a > q2)),
            labels=("localized", "widespread"),
            rule=f"area(>= {thr} mm/h): bottom tercile (<= {q1:.0f}) vs top (> {q2:.0f} km^2)",
        )
    for src, col in (("s4", "s4_nyc_peak_mm"), ("asos", "asos_peak_mm")):
        v = df[col]
        med = v.median()
        splits[f"{src}_rate_median"] = dict(
            high=v > med,
            valid=v.notna(),
            labels=("modest", "heavy"),
            rule=f"{src.upper()} NYC peak > median ({med:.1f} mm/h)",
        )
        splits[f"{src}_rate_1in"] = dict(
            high=v >= IN_TO_MM,
            valid=v.notna(),
            labels=("modest", "heavy"),
            rule=f"{src.upper()} NYC peak >= 25.4 mm/h",
        )
    return splits


def evaluate(df, splits):
    res = {}
    for name, s in splits.items():
        sub = df[s["valid"]]
        hi = s["high"][s["valid"]].to_numpy()
        lab = np.where(hi, s["labels"][1], s["labels"][0])
        r = dict(
            rule=s["rule"],
            n=dict(zip(s["labels"], [int((~hi).sum()), int(hi.sum())], strict=True)),
        )
        r["node_mix"] = {
            som: node_mix_test(lab, sub[som].to_numpy())
            for som in ("evsom", "static", "alldays")
        }
        # Robust all-days result: the trough + moisture corner (A4/A5)
        corner = sub["alldays"].isin(["A4", "A5"]).to_numpy()
        r["alldays_A4A5_share"] = {
            s["labels"][1]: frac(corner[hi]),
            s["labels"][0]: frac(corner[~hi]),
        }
        r["d3"] = {}
        for var in (
            "thetae_T0",
            "tcwv_T0",
            "z500_T0",
            "thetae_day",
            "tcwv_day",
            "z500_day",
        ):
            v = sub[var].to_numpy()
            r["d3"][var] = mwu(v[hi], v[~hi])
        r["moisture_vs_trough"] = {}
        for g, m in ((s["labels"][1], hi), (s["labels"][0], ~hi)):
            th, tc, z = (
                sub[c].to_numpy()[m] for c in ("thetae_T0", "tcwv_T0", "z500_T0")
            )
            r["moisture_vs_trough"][g] = dict(
                thetae_pos=frac(th > 0),
                tcwv_pos=frac(tc > 0),
                z500_neg=frac(z < 0),
                moist_and_trough=frac((th > 0) & (z < 0)),
                moist_no_trough=frac((th > 0) & (z >= 0)),
                dry_and_trough=frac((th <= 0) & (z < 0)),
                dry_no_trough=frac((th <= 0) & (z >= 0)),
            )
        res[name] = r
    return res


def threshold_dependence(df):
    s = df.dropna(subset=[f"area_ge{t}_km2" for t in AREA_THRESHOLDS])
    out = {
        "n": len(s),
        "spearman": {},
        "median_split_kappa": {},
        "median_split_agree_pct": {},
    }
    meds = {
        t: s[f"area_ge{t}_km2"] > s[f"area_ge{t}_km2"].median() for t in AREA_THRESHOLDS
    }
    for i, a in enumerate(AREA_THRESHOLDS):
        for b in AREA_THRESHOLDS[i + 1 :]:
            k = f"{a}_vs_{b}"
            out["spearman"][k] = float(
                spearmanr(s[f"area_ge{a}_km2"], s[f"area_ge{b}_km2"])[0]
            )
            out["median_split_kappa"][k] = kappa(meds[a], meds[b])
            out["median_split_agree_pct"][k] = float(100 * np.mean(meds[a] == meds[b]))
    out["area_vs_s4_peak_spearman"] = {
        t: float(
            spearmanr(s[f"area_ge{t}_km2"], s["s4_nyc_peak_mm"], nan_policy="omit")[0]
        )
        for t in AREA_THRESHOLDS
    }
    out["median_km2"] = {
        t: float(s[f"area_ge{t}_km2"].median()) for t in AREA_THRESHOLDS
    }
    out["zero_area_events"] = {
        t: int((s[f"area_ge{t}_km2"] == 0).sum()) for t in AREA_THRESHOLDS
    }
    return out


# ── Output ───────────────────────────────────────────────────────────────────


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def write_summary(df, res, dep, d4_check):
    L = []
    L.append("# D2 stratification (preview on the published SOMs)\n")
    L.append(
        f"Events: {len(df)} warm-season FF events; Stage IV coverage n = "
        f"{int(df['area_ge25_km2'].notna().sum())}; ASOS n = {int(df['asos_peak_mm'].notna().sum())}. "
        "Window (t0-6h, t0+6h], hour-ending. Area = land/coastal cells of 38.5-44N, 78-70W "
        "(open ocean masked). D3 anomalies: month-day climatology (as D3); Z500 in the "
        f"{TROUGH_BOX} box {UPSTREAM_BOXES[TROUGH_BOX]}.\n"
    )
    if d4_check is not None:
        L.append(
            f"Stage IV NYC peak vs D4 extract: max |diff| = {d4_check:.3g} mm/h.\n"
        )
    L.append("## Threshold dependence of the rain-area classification\n")
    L.append(
        f"- Median area (km^2): {dep['median_km2']}; events with zero area: {dep['zero_area_events']}"
    )
    L.append(f"- Spearman between thresholds: {dep['spearman']}")
    L.append(
        f"- Median-split agreement (%): {dep['median_split_agree_pct']}; Cohen's kappa: "
        f"{ {k: round(v, 2) for k, v in dep['median_split_kappa'].items()} }"
    )
    L.append(
        f"- Spearman(area, Stage IV NYC peak): {dep['area_vs_s4_peak_spearman']}\n"
    )
    L.append("## Node mix and D3 metrics by group\n")
    L.append(
        "| Split | n (low/high) | evSOM p | static p | all-days p | A4+A5 % (high vs low) | "
        "θe T0 med (high/low), p | TCWV T0 p | Z500 T0 med (high/low), p |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|")
    for name, r in res.items():
        lo_l, hi_l = list(r["n"].keys())
        a = r["alldays_A4A5_share"]
        th, tc, z = r["d3"]["thetae_T0"], r["d3"]["tcwv_T0"], r["d3"]["z500_T0"]
        L.append(
            f"| {name} | {r['n'][lo_l]}/{r['n'][hi_l]} | {r['node_mix']['evsom']['p_mc']:.2f} | "
            f"{r['node_mix']['static']['p_mc']:.2f} | {r['node_mix']['alldays']['p_mc']:.2f} | "
            f"{a[hi_l]['pct']:.0f} vs {a[lo_l]['pct']:.0f} | "
            f"{th['med_a']:.2f}/{th['med_b']:.2f}, {th['p']:.2f} | {tc['p']:.2f} | "
            f"{z['med_a']:.2f}/{z['med_b']:.2f}, {z['p']:.2f} |"
        )
    L.append(
        "\nRules: " + "; ".join(f"{k}: {r['rule']}" for k, r in res.items()) + "\n"
    )
    L.append("## Moisture nearly always, troughing sometimes? (T0, % of group)\n")
    L.append(
        "| Split | group | θe>0 | TCWV>0 | Z500<0 | moist+trough | moist, no trough | dry+trough | dry, no trough |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|")
    for name, r in res.items():
        for g, m in r["moisture_vs_trough"].items():
            L.append(
                f"| {name} | {g} | "
                + " | ".join(
                    f"{m[k]['pct']:.0f}"
                    for k in (
                        "thetae_pos",
                        "tcwv_pos",
                        "z500_neg",
                        "moist_and_trough",
                        "moist_no_trough",
                        "dry_and_trough",
                        "dry_no_trough",
                    )
                )
                + " |"
            )
    L.append("")
    L.append("## evSOM node mix (counts)\n")
    for name in ("area25_median", "s4_rate_median", "asos_rate_median"):
        L.append(f"- {name}: {res[name]['node_mix']['evsom']['table']}")
    with open(os.path.join(OUT_DIR, "d2_summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")


NODE_COLORS = {"A1": "#1b9e77", "A2": "#d95f02", "B1": "#7570b3", "B2": "#e7298a"}


def plot(df, splits):
    fig = plt.figure(figsize=(11, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, :2])
    s = df.dropna(subset=[f"area_ge{PRIMARY_AREA}_km2", "s4_nyc_peak_mm"])
    for node, col in NODE_COLORS.items():
        m = s["evsom"] == node
        ax.scatter(
            s.loc[m, f"area_ge{PRIMARY_AREA}_km2"] + 1,
            s.loc[m, "s4_nyc_peak_mm"],
            c=col,
            s=28,
            label=f"{node} (n={int(m.sum())})",
            edgecolor="k",
            linewidth=0.3,
        )
    ax.set_xscale("log")
    ax.axvline(
        s[f"area_ge{PRIMARY_AREA}_km2"].median() + 1, color="0.4", ls="--", lw=0.8
    )
    ax.axhline(s["s4_nyc_peak_mm"].median(), color="0.4", ls="--", lw=0.8)
    ax.axhline(IN_TO_MM, color="0.4", ls=":", lw=0.8)
    ax.set_xlabel(f"Regional area with Stage IV rate ≥ {PRIMARY_AREA} mm/h (km², +1)")
    ax.set_ylabel("Stage IV NYC peak rate (mm/h)")
    ax.legend(title="evSOM node", fontsize=8)
    ax.set_title("(a) Rain area vs NYC peak rate (±6 h)", loc="left", fontsize=10)

    ax2 = fig.add_subplot(gs[0, 2])
    sh = df.dropna(subset=["s4_nyc_peak_mm", "asos_peak_mm"])
    ax2.scatter(sh["s4_nyc_peak_mm"], sh["asos_peak_mm"], s=14, c="0.3")
    lim = [0, max(sh["s4_nyc_peak_mm"].max(), sh["asos_peak_mm"].max()) * 1.05]
    ax2.plot(lim, lim, "k:", lw=0.8)
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.set_xlabel("Stage IV NYC peak (mm/h)")
    ax2.set_ylabel("ASOS peak (mm/h)")
    ax2.set_title("(b) Peak rate: Stage IV vs ASOS", loc="left", fontsize=10)

    groups = [
        (f"area{PRIMARY_AREA}_median", "Area"),
        ("s4_rate_median", "Stage IV rate"),
    ]
    for k, (var, lab) in enumerate(
        (
            ("thetae_T0", "NYC θe,850 anomaly at T0 (σ)"),
            ("tcwv_T0", "NYC TCWV anomaly at T0 (σ)"),
            ("z500_T0", "Upstream Z500 anomaly at T0 (σ)"),
        )
    ):
        a = fig.add_subplot(gs[1, k])
        data, ticks = [], []
        for split, short in groups:
            sp = splits[split]
            v = df[var][sp["valid"]]
            hi = sp["high"][sp["valid"]]
            data += [v[~hi].dropna(), v[hi].dropna()]
            ticks += [f"{short}\n{sp['labels'][0]}", f"{short}\n{sp['labels'][1]}"]
        a.boxplot(data, widths=0.6, showfliers=True, flierprops=dict(markersize=3))
        a.set_xticks(range(1, len(ticks) + 1), ticks, fontsize=7)
        a.axhline(0, color="k", lw=0.6)
        a.set_ylabel(lab, fontsize=8)
        a.set_title(f"({'cde'[k]})", loc="left", fontsize=10)
    fig.savefig(os.path.join(OUT_DIR, "d2_stratification.png"), dpi=200)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    events = load_events()
    print(f"Events: {len(events)}", flush=True)
    s4 = stageiv_event_metrics(events)
    df = events.merge(s4, on="timestamp", how="left")
    d4_check = check_against_d4(df)
    print(f"Stage IV NYC peak vs D4 cache: max |diff| = {d4_check}", flush=True)
    asos = load_asos_precip()
    df["asos_peak_mm"] = (
        np.asarray(compute_asos_max_precip(df, asos, WINDOW_H), float) * IN_TO_MM
    )
    df = df.merge(d3_series(events), on="timestamp", how="left")

    splits = build_splits(df)
    res = evaluate(df, splits)
    dep = threshold_dependence(df)
    out = dict(
        n_events=len(df),
        n_stageiv=int(df["area_ge25_km2"].notna().sum()),
        n_asos=int(df["asos_peak_mm"].notna().sum()),
        window_hours=WINDOW_H,
        regional_domain=REG,
        nyc_box=NYC,
        trough_box={TROUGH_BOX: UPSTREAM_BOXES[TROUGH_BOX]},
        d4_cross_check_max_abs_diff_mm=d4_check,
        threshold_dependence=dep,
        splits=res,
        n_permutations=N_PERM,
    )
    with open(os.path.join(OUT_DIR, "d2_metrics.json"), "w") as f:
        json.dump(_jsonable(out), f, indent=1)
    df.to_csv(os.path.join(OUT_DIR, "d2_events.csv"), index=False)
    write_summary(df, res, dep, d4_check)
    plot(df, splits)
    print(f"Wrote {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
