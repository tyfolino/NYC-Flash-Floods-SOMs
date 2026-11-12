"""
Every manuscript number the code can produce, in one JSON file.

Reads the SOM outputs under som_analysis.config.OUTPUT_ROOT (caches in
figs/, BMU and TC CSVs in data/, D3/D4 results in results/) and writes
results/manuscript_numbers.json. Sections that cannot be computed (e.g. no
results/d4 yet) are skipped with a note rather than failing.

Usage (from the project root):
    python -m som_analysis.manuscript_numbers [--skip-precip] [--skip-fig4]
"""

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from scipy.stats import (  # noqa: E402  # noqa: E402
    chi2_contingency,
    fisher_exact,
    kruskal,
    mannwhitneyu,
)
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from .config import (  # noqa: E402
    DATA_DIR,
    FIGS_DIR,
    OUTPUT_ROOT,
    RESULTS_DIR,
    SOM_INTERMEDIATE_PATH,
)
from .helpers import node_label  # noqa: E402
from .timeutils import ff_event_hours_utc, load_sed  # noqa: E402

WARM_MONTHS = (5, 6, 7, 8, 9, 10)
YEARS = (1996, 2025)
NYC_BOX_ERA5 = dict(lat=(40.5, 41.0), lon=(-74.25, -73.75))  # Fig. S7 / D3 box
NYC_BOX_FIG4 = dict(lat=(40.3, 41.2), lon=(-74.8, -73.4))  # Fig. 4 NYC panel
NODES_2X2 = [(0, 0), (1, 0), (0, 1), (1, 1)]  # A1, A2, B1, B2

EVSOM_BMU = "som_2x2_evsom_24h_bmus_thetae.csv"
STATIC_BMU = "som_2x2_bmus_thetae.csv"
EVSOM_TC = "som_2x2_evsom_24h_tc_associations.csv"
ALLDAYS_CACHE = os.path.join("Z500-and-thetae-alldays-SOM", ".cache", "som_results.npz")


def _r(x, nd=4):
    """Round floats (and numpy scalars) for the JSON file."""
    if x is None:
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    x = float(x)
    return None if not np.isfinite(x) else round(x, nd)


def _labels(df):
    return [
        node_label(int(i), int(j)) for i, j in zip(df.node_i, df.node_j, strict=True)
    ]


def _read_bmu(name):
    return pd.read_csv(os.path.join(DATA_DIR, name), parse_dates=["timestamp"])


# ── Sections ──────────────────────────────────────────────────────────────────


def events_section():
    sed = load_sed()
    ep = sed.drop_duplicates(subset=["EPISODE_ID"], keep="first")
    days = ff_event_hours_utc()
    yr = (days.year >= YEARS[0]) & (days.year <= YEARS[1])
    warm = days[yr & days.month.isin(WARM_MONTHS)]
    ep_t = pd.DatetimeIndex(ep["BEGIN_UTC"])
    cold_ep = ep[~ep_t.month.isin(WARM_MONTHS)]
    cold_rows = sed[sed["EPISODE_ID"].isin(cold_ep["EPISODE_ID"])]
    return dict(
        sed_rows=len(sed),
        sed_episodes=int(sed["EPISODE_ID"].nunique()),
        ff_days_all_seasons=len(days),
        ff_days_warm_season=len(warm),
        warm_season_share=_r(len(warm) / len(days)),
        cold_season_episodes=len(cold_ep),
        cold_season_deaths_direct=int(cold_rows["DEATHS_DIRECT"].sum()),
        cold_season_deaths_indirect=int(cold_rows["DEATHS_INDIRECT"].sum()),
        cold_season_injuries_direct=int(cold_rows["INJURIES_DIRECT"].sum()),
        cold_season_episodes_with_damage=int(
            (cold_rows.groupby("EPISODE_ID")["DAMAGE_PROPERTY_NUM"].sum() > 0).sum()
        ),
    )


def alldays_section():
    path = os.path.join(FIGS_DIR, ALLDAYS_CACHE)
    z = np.load(path, allow_pickle=True)
    counts, totals = z["counts"], z["totals"]
    xdim, ydim = counts.shape
    is_ff = z["is_ff"].astype(bool)
    base = is_ff.mean()
    nodes = {}
    for i in range(xdim):
        for j in range(ydim):
            n, c = int(totals[i, j]), int(counts[i, j])
            risk = c / n if n else float("nan")
            nodes[node_label(i, j)] = dict(
                days=n, ff_days=c, risk=_r(risk), risk_ratio=_r(risk / base, 3)
            )
    ranked = sorted(nodes, key=lambda k: -(nodes[k]["risk_ratio"] or 0))
    out = dict(
        grid=f"{xdim}x{ydim}",
        n_days=int(is_ff.size),
        n_ff_days=int(is_ff.sum()),
        base_rate=_r(base),
        nodes=nodes,
        top_nodes=ranked[:3],
        max_risk_ratio=nodes[ranked[0]]["risk_ratio"],
        n_nodes_without_ff=int((counts == 0).sum()),
        every_node_has_ff=bool((counts > 0).all()),
    )
    if "train_config" in z:
        out["train_config"] = json.loads(str(z["train_config"]))
    return out


def two_by_two_section():
    ev = _read_bmu(EVSOM_BMU)
    st = _read_bmu(STATIC_BMU)
    out = {}
    for name, df in (("evsom", ev), ("static", st)):
        lab = pd.Series(_labels(df))
        cnt = lab.value_counts().reindex(["A1", "A2", "B1", "B2"], fill_value=0)
        top2 = cnt.sort_values(ascending=False).iloc[:2]
        out[name] = dict(
            n=len(df),
            node_counts={k: int(v) for k, v in cnt.items()},
            node_shares={k: _r(v / len(df), 3) for k, v in cnt.items()},
            a1_plus_b2_share=_r((cnt["A1"] + cnt["B2"]) / len(df), 3),
            most_populated=list(top2.index),
        )
    m = ev.merge(st, on="timestamp", suffixes=("_ev", "_st"))
    lab_ev = [node_label(i, j) for i, j in zip(m.node_i_ev, m.node_j_ev, strict=True)]
    lab_st = [node_label(i, j) for i, j in zip(m.node_i_st, m.node_j_st, strict=True)]
    out["static_vs_evsom"] = dict(
        n_matched=len(m),
        agreement=_r(np.mean(np.array(lab_ev) == np.array(lab_st)), 3),
        ari=_r(adjusted_rand_score(lab_st, lab_ev), 3),
        contingency=pd.crosstab(
            pd.Series(lab_st, name="static"), pd.Series(lab_ev, name="evsom")
        ).to_dict(),
    )
    return out


def tc_section():
    tc = pd.read_csv(os.path.join(DATA_DIR, EVSOM_TC), parse_dates=["timestamp"])
    tc["node"] = _labels(tc)
    n_tc = int(tc["tc_present"].sum())
    per = {}
    table = []
    for nd in ["A1", "A2", "B1", "B2"]:
        sub = tc[tc["node"] == nd]
        k = int(sub["tc_present"].sum())
        per[nd] = dict(tc=k, n=len(sub), frac=_r(k / len(sub), 3) if len(sub) else None)
        table.append([k, len(sub) - k])
    chi2_p = chi2_contingency(np.array(table))[1]
    top = max(per, key=lambda k: per[k]["frac"] or 0)
    rest = [
        r for nd, r in zip(["A1", "A2", "B1", "B2"], table, strict=True) if nd != top
    ]
    fisher_p = fisher_exact(
        [table[["A1", "A2", "B1", "B2"].index(top)], np.sum(rest, axis=0)]
    )[1]
    statuses = (
        tc.loc[tc["tc_present"], "max_status"].fillna("").str.split(", ").explode()
    )
    return dict(
        n=len(tc),
        n_tc=n_tc,
        frac=_r(n_tc / len(tc), 3),
        per_node=per,
        highest_node=top,
        chi2_p=_r(chi2_p, 3),
        fisher_top_vs_rest_p=_r(fisher_p, 3),
        n_tc_subtropical_only=int(
            tc.loc[tc["tc_present"], "max_status"]
            .fillna("")
            .apply(lambda s: all(x in ("SD", "SS") for x in s.split(", ") if x))
            .sum()
        ),
        max_status_counts=statuses.value_counts().to_dict(),
    )


def precip_section():
    """Fig. S8 numbers (Stage IV and ASOS max hourly rate within +/-6 h)."""
    from .node_statistics import (
        compute_asos_max_precip,
        compute_stageiv_max_precip,
        load_asos_precip,
    )

    b = _read_bmu(EVSOM_BMU)
    b["node"] = _labels(b)
    s4, *_ = compute_stageiv_max_precip(b, window_hours=6)
    b["stageiv"] = s4
    b["asos"] = compute_asos_max_precip(b, load_asos_precip(), window_hours=6)
    out = {}
    for col in ("stageiv", "asos"):
        groups = {
            nd: b.loc[b.node == nd, col].dropna() for nd in ["A1", "A2", "B1", "B2"]
        }
        allv = b[col].dropna()
        wet = pd.concat([groups["B1"], groups["B2"]])
        dry = pd.concat([groups["A1"], groups["A2"]])
        out[col] = dict(
            n=len(allv),
            median_in=_r(allv.median(), 3),
            node_n={k: len(v) for k, v in groups.items()},
            node_median_in={k: _r(v.median(), 3) for k, v in groups.items()},
            frac_below_1in=_r((allv < 1).mean(), 3),
            frac_at_or_above_1p75in=_r((allv >= 1.75).mean(), 3),
            kruskal_p=_r(kruskal(*groups.values()).pvalue, 3),
            mw_b1b2_vs_a1a2_p=_r(mannwhitneyu(wet, dry).pvalue, 4),
        )
    return out


def fig4_section():
    """Fig. 4 numbers. Regional panel: PMM pooled over the region. NYC panel:
    PMM pooled over NYC-panel cells only (as plotted); the regional-pool NYC
    values (as submitted) are kept for traceability."""
    from .node_statistics import (
        compute_stageiv_node_composites,
        probability_matched_mean,
        subdomain_fields,
    )

    b = _read_bmu(EVSOM_BMU)
    nf, lat, lon = compute_stageiv_node_composites(b, 2, 2)
    ext = [*NYC_BOX_FIG4["lon"], *NYC_BOX_FIG4["lat"]]
    nf_nyc = subdomain_fields(nf, lat, lon, ext)
    nyc = (
        (lat >= NYC_BOX_FIG4["lat"][0])
        & (lat <= NYC_BOX_FIG4["lat"][1])
        & (lon >= NYC_BOX_FIG4["lon"][0])
        & (lon <= NYC_BOX_FIG4["lon"][1])
    )
    out = {}
    for key in NODES_2X2:
        pm = probability_matched_mean(nf[key])
        pm_nyc = probability_matched_mean(nf_nyc[key])
        out[node_label(*key)] = dict(
            n_with_data=int((~np.all(np.isnan(nf[key]), axis=(1, 2))).sum()),
            n_with_nyc_data=int((~np.all(np.isnan(nf_nyc[key]), axis=(1, 2))).sum()),
            nyc_mean_in=_r(np.nanmean(pm_nyc[nyc]), 3),
            nyc_max_in=_r(np.nanmax(pm_nyc[nyc]), 2),
            nyc_cells_ge_1in=int((pm_nyc[nyc] >= 1).sum()),
            nyc_mean_in_regional_pool=_r(np.nanmean(pm[nyc]), 3),
            nyc_max_in_regional_pool=_r(np.nanmax(pm[nyc]), 2),
            regional_max_in=_r(np.nanmax(pm), 2),
            regional_cells_ge_0p5in=int((pm >= 0.5).sum()),
            regional_cells_ge_1in=int((pm >= 1).sum()),
        )
    return out


def s7_section():
    """NYC-box mean IVT / TCWV at onset by evSOM node (Fig. S7)."""
    b = _read_bmu(EVSOM_BMU)
    b["node"] = _labels(b)
    out = {}
    for name, fname in (("ivt", "era5_ivt_ffe.nc"), ("tcwv", "era5_tcwv_ffe.nc")):
        ds = xr.load_dataset(os.path.join(SOM_INTERMEDIATE_PATH, fname))
        da = ds[name] if name in ds else ds[list(ds.data_vars)[0]]
        da = da.sel(valid_time=pd.DatetimeIndex(b["timestamp"]))
        box = da.where(
            (da.latitude >= NYC_BOX_ERA5["lat"][0])
            & (da.latitude <= NYC_BOX_ERA5["lat"][1])
            & (da.longitude >= NYC_BOX_ERA5["lon"][0])
            & (da.longitude <= NYC_BOX_ERA5["lon"][1])
        ).mean(["latitude", "longitude"])
        vals = pd.Series(box.values, index=b.index)
        out[name] = {
            nd: _r(vals[b.node == nd].mean(), 1) for nd in ["A1", "A2", "B1", "B2"]
        }
    return out


def d3_section():
    path = os.path.join(RESULTS_DIR, "d3", "d3_metrics.json")
    if not os.path.exists(path):
        return dict(skipped=f"{path} not found (run moisture_forcing_metrics)")
    d = json.load(open(path))
    out = {}
    for var, m in d["metrics"].items():
        row = {}
        for tag, v in m.items():
            th = v.get("thresholds", {})
            row[tag] = dict(
                auc=_r(v.get("auc"), 3),
                frac_ff_gt0=_r(th.get("0", {}).get("frac_ff"), 3),
                frac_all_gt0=_r(th.get("0", {}).get("frac_all"), 3),
                frac_ff_gt1=_r(th.get("1", {}).get("frac_ff"), 3),
                frac_all_gt1=_r(th.get("1", {}).get("frac_all"), 3),
                risk_ratio_gt1=_r(th.get("1", {}).get("risk_ratio"), 2),
                risk_ratio_gt1_ci95=[
                    _r(x, 2) for x in th.get("1", {}).get("rr_ci95", [])
                ],
            )
        out[var] = row
    out["joint_thetae_vs_trough"] = {
        k: {kk: _r(vv, 3) for kk, vv in v.items()}
        for k, v in d.get("joint_daily_utc_thetae_vs_z500_trough_T0", {}).items()
    }
    return out


def d4_section():
    path = os.path.join(RESULTS_DIR, "d4", "d4_metrics.json")
    if not os.path.exists(path):
        return dict(skipped=f"{path} not found (run extreme_day_comparison)")
    d = json.load(open(path))
    keep = (
        "pct_daily_extreme",
        "pct_hourly_extreme",
        "pct_hourly_not_daily",
        "pct_neither",
        "hit_rate_daily",
        "hit_rate_hourly",
        "pct_below_25p4",
        "pct_at_or_above_44p45",
        "q_max_rate_mm_h",
        "q_daily_total_mm",
        "threshold_daily_mm",
        "threshold_hourly_mm",
    )
    out = {"definition": d.get("definition")}
    for k, v in d.items():
        if isinstance(v, dict) and any(x in v for x in keep):
            out[k] = {x: v[x] for x in keep if x in v} | {"n_events": v.get("n_events")}
    return out


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-precip", action="store_true", help="Skip Fig. S8 numbers.")
    ap.add_argument("--skip-fig4", action="store_true", help="Skip Fig. 4 numbers.")
    ap.add_argument(
        "--out", default=os.path.join(RESULTS_DIR, "manuscript_numbers.json")
    )
    args = ap.parse_args()

    sections = [
        ("events", events_section),
        ("alldays_som", alldays_section),
        ("ff_soms_2x2", two_by_two_section),
        ("tc_association_evsom", tc_section),
        ("figS7_nyc_box", s7_section),
        ("d3_moisture_forcing", d3_section),
        ("d4_extreme_days", d4_section),
    ]
    if not args.skip_precip:
        sections.insert(4, ("figS8_precip", precip_section))
    if not args.skip_fig4:
        sections.insert(5, ("fig4_stageiv_pmm", fig4_section))

    out = dict(
        output_root=OUTPUT_ROOT,
        intermediate_path=SOM_INTERMEDIATE_PATH,
        generated=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
    )
    for name, fn in sections:
        print(f"[manuscript_numbers] {name} ...", flush=True)
        try:
            out[name] = fn()
        except FileNotFoundError as err:
            out[name] = dict(skipped=f"missing input: {err}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
