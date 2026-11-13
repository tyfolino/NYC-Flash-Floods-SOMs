"""
Phase 2 (D1): evolution-SOM analyses on the stage-3b/5 seed ensembles.

For one evSOM window (--hours 24|48|72), using the v2 seed ensembles written by
``robustness_experiment.py`` (stages 4-5):

1. Seed baseline: within-method ARI/agreement (static T=0 vs static, evSOM vs
   evSOM) against cross-method (static vs evSOM). Cross-method values well below
   the within-method spread mean the reclassification is signal, not seed noise.
2. Representative members: the seed with the highest mean ARI to the others.
   2x2 nodes are named by matching to the published SOM (static -> published
   static; evSOM of any window -> published 24-h evSOM), so A1/A2/B1/B2 keep
   their meaning across windows. The all-days 5x3 member is flipped so that its
   highest-risk node is A5 (published convention: row A moist, column 5 trough).
3. Reclassified events: static -> evSOM contingency; for each evSOM node and each
   reclassification, mean anomalies at T-24 / T-12 / T0 in a few boxes
   (Midwest and Plains Z500 for trough position, Canada Z500 for a northern
   trough, NYC theta-e for moisture).
4. All-days pathways: the 5x3 all-days node on UTC days d-3 .. d0 for each event
   (d0 = UTC day of onset), by evSOM node, with node risk (metrics). The figure
   shows only the days the evSOM window spans (24 h: d-1..d0, 48 h: d-2..d0,
   72 h: d-3..d0): day-to-day moves shared by >= 2 events as arrows, one-off
   moves as faint lines, d0 counts as rings. Inside the window the paths differ
   partly by construction (both SOMs see Z500 and theta-e), so the figure shows
   how each archetype evolves, not independent evidence.
5. Lead-time test: assign each event to the evSOM using only hours <= T-k
   (masked distance, same weighted features as training); share matching the
   full-window node vs chance (sum of squared node frequencies). In-sample: the
   map was trained on the full windows, so this is an upper bound.

Usage (from the project root):
    OMP_NUM_THREADS=1 python -m som_analysis.evsom_phase2 --hours 24

Outputs: results/phase2/h{H}/ (phase2_metrics.json, phase2_summary.md,
pathways PNG, reclassified-events CSV).
"""

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import glob  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
from collections import Counter  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from scipy.stats import chi2_contingency, kruskal  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402

from som_analysis.helpers import load_moist_var  # noqa: E402
from som_analysis.robustness_experiment import (  # noqa: E402
    CFG,
    INTER,
    OUT_DIR,
    PUBLISHED_CACHE,
    ROOT,
    load_X,
)

RUNS = os.path.join(OUT_DIR, "runs")
ENSEMBLE = {
    "static": "static_len5_init-random_mode-random_s2-1.0_lr1-0.5_lr2-0.02_inter-v2",
    "evsom": "evsom_len5_init-random_mode-random_s2-1.0_lr1-0.5_lr2-0.02_inter-v2",
    "alldays": (
        "alldays_len5_init-random_mode-random_s2-1.0_lr1-0.3_lr2-0.02_inter-v2"
        "_xdim-5_ydim-3"
    ),
}
ALLDAYS_GRID = (5, 3)
BOXES = {  # (lat0, lat1, lon0, lon1), degrees E negative west
    "Z500_Midwest": (38, 46, -92, -84),
    "Z500_Plains": (36, 46, -100, -94),
    "Z500_Canada": (48, 54, -92, -78),
    "thetae_NYC": (40.0, 41.5, -74.5, -73.0),
}


def name(k, ydim):
    """Flat index i*ydim + j -> label (letter = row j, number = column i + 1)."""
    return f"{chr(65 + k % ydim)}{k // ydim + 1}"


def ensemble(kind, hours=24):
    tag = ENSEMBLE[kind] + (
        f"_hours-{hours}" if kind == "evsom" and hours != 24 else ""
    )
    files = sorted(glob.glob(os.path.join(RUNS, f"{tag}_seed*.npz")))
    if not files:
        raise SystemExit(f"No runs for {tag}")
    runs = []
    for f in files:
        z = np.load(f)
        seed = int(f.rsplit("_seed", 1)[1].split(".")[0])
        runs.append(
            dict(seed=seed, labels=z["labels"].astype(int), weights=z["weights"])
        )
    return runs


def representative(runs):
    L = [r["labels"] for r in runs]
    m = [
        np.mean([ari(a, b) for j, b in enumerate(L) if j != i]) for i, a in enumerate(L)
    ]
    i = int(np.argmax(m))
    return runs[i], float(m[i])


def published_labels(kind):
    cz = np.load(os.path.join(ROOT, PUBLISHED_CACHE[kind]), allow_pickle=True)
    b = cz["bmus"]
    return b[:, 0] * 2 + b[:, 1]


def match_to(labels, ref, n=4):
    """Permutation of node ids maximizing agreement with ref; returns the
    relabelled labels, the old->new map, and the agreement."""
    C = np.zeros((n, n))
    np.add.at(C, (labels, ref), 1)
    r, c = linear_sum_assignment(-C)
    perm = dict(zip(r.tolist(), c.tolist(), strict=True))
    return np.array([perm[x] for x in labels]), perm, float(C[r, c].sum() / len(labels))


def agreement(a, b, n=4):
    return match_to(a, b, n)[2]


def orient_alldays(labels, is_ff, xdim, ydim):
    """Flip the grid so the highest-risk node sits at (i=xdim-1, j=0) = 'A5'."""
    nn = xdim * ydim
    risk = np.bincount(labels[is_ff], minlength=nn) / np.maximum(
        np.bincount(labels, minlength=nn), 1
    )
    k = int(risk.argmax())
    i, j = divmod(k, ydim)
    fi, fj = i != xdim - 1, j != 0
    ii, jj = np.divmod(labels, ydim)
    ii = xdim - 1 - ii if fi else ii
    jj = ydim - 1 - jj if fj else jj
    return ii * ydim + jj, dict(flip_columns=bool(fi), flip_rows=bool(fj))


def q(x):
    x = np.asarray(x, float)
    return dict(
        median=float(np.median(x)),
        p25=float(np.percentile(x, 25)),
        p75=float(np.percentile(x, 75)),
        min=float(x.min()),
        max=float(x.max()),
    )


def box_means(da):
    lon = da.longitude.values
    lon = np.where(lon > 180, lon - 360, lon)
    lat = da.latitude.values
    out = {}
    for key, (la0, la1, lo0, lo1) in BOXES.items():
        sel_lat = (lat >= la0) & (lat <= la1)
        sel_lon = (lon >= lo0) & (lon <= lo1)
        out[key] = da.isel(latitude=sel_lat, longitude=sel_lon).mean(
            ["latitude", "longitude"]
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--hours", type=int, default=24, choices=(24, 48, 72))
    args = ap.parse_args()
    H = args.hours
    out_dir = os.path.join(ROOT, "results", "phase2", f"h{H}")
    os.makedirs(out_dir, exist_ok=True)
    P = INTER["v2"]
    suf = "" if H == 24 else f"{H}h"
    metrics = dict(hours=H)

    # ── Event times (same order in the static and evSOM inputs) ─────────────
    ts = xr.open_dataarray(f"{P}era5_Z500_norm_weighted_ffe.nc")
    te = xr.open_dataarray(f"{P}era5_Z500_norm_weighted_ffe_evsom{suf}.nc")
    t_static = pd.DatetimeIndex(ts[ts.dims[0]].values)
    t_ev = pd.DatetimeIndex(te["event_time"].values)
    assert t_static.equals(t_ev), "event order differs between static and evSOM"
    n_ev = len(t_ev)

    # ── 1. Seed baseline ─────────────────────────────────────────────────────
    S, E = ensemble("static"), ensemble("evsom", H)
    SL, EL = [r["labels"] for r in S], [r["labels"] for r in E]
    pairs = {
        "static_static": list(itertools.combinations(SL, 2)),
        "evsom_evsom": list(itertools.combinations(EL, 2)),
        "static_evsom": [(a, b) for a in SL for b in EL],
    }
    base = {}
    for k, ps in pairs.items():
        base[k] = dict(
            ari=q([ari(a, b) for a, b in ps]),
            agreement=q([agreement(a, b) for a, b in ps]),
        )
    cross = np.array([ari(a, b) for a, b in pairs["static_evsom"]])
    base["cross_above_within_evsom_p5"] = float(
        np.mean(cross > np.percentile([ari(a, b) for a, b in pairs["evsom_evsom"]], 5))
    )
    base["n_seeds"] = dict(static=len(S), evsom=len(E))
    metrics["seed_baseline"] = base

    # ── 2. Representative members, named like the published SOMs ────────────
    rs, rs_ari = representative(S)
    re_, re_ari = representative(E)
    s_lab, s_perm, s_match = match_to(rs["labels"], published_labels("static"))
    e_lab, e_perm, e_match = match_to(re_["labels"], published_labels("evsom"))
    metrics["representative"] = dict(
        static=dict(seed=rs["seed"], mean_ari=rs_ari, agreement_with_published=s_match),
        evsom=dict(
            seed=re_["seed"],
            mean_ari=re_ari,
            agreement_with_published_24h=e_match,
        ),
    )
    ev_counts = np.bincount(e_lab, minlength=4)
    metrics["evsom_node_counts"] = {name(k, 2): int(ev_counts[k]) for k in range(4)}

    # ── 3. Reclassified events ───────────────────────────────────────────────
    C = np.zeros((4, 4), int)
    np.add.at(C, (s_lab, e_lab), 1)
    metrics["static_to_evsom"] = {
        name(a, 2): {name(b, 2): int(C[a, b]) for b in range(4)} for a in range(4)
    }
    metrics["n_reclassified"] = int((s_lab != e_lab).sum())

    z = xr.open_dataarray(f"{P}era5_Z500_norm_ffe_evsom{suf}.nc")
    m = load_moist_var(f"{P}era5_thetae_norm_ffe_evsom{suf}.nc", CFG["var_name"])
    bz, bm = box_means(z), box_means(m)
    series = {k: v.values for k, v in bz.items() if k.startswith("Z500")}
    series["thetae_NYC"] = bm["thetae_NYC"].values  # (event, hour)
    offs = z.hour_offset.values
    lead_hours = [h for h in (-(H - 1), -48, -24, -12, 0) if h >= -(H - 1)]
    lead_hours = sorted(set(lead_hours))

    def profile(sel):
        return {
            k: {f"T{h:+d}": float(np.nanmean(v[sel][:, offs == h])) for h in lead_hours}
            for k, v in series.items()
        }

    metrics["evsom_node_profiles"] = {
        name(k, 2): profile(e_lab == k) for k in range(4) if (e_lab == k).any()
    }
    recl = {}
    for a, b in zip(*np.nonzero(C), strict=True):
        if a != b:
            recl[f"{name(a, 2)}->{name(b, 2)}"] = dict(
                n=int(C[a, b]), **profile((s_lab == a) & (e_lab == b))
            )
    metrics["reclassified_profiles"] = recl
    pd.DataFrame(
        dict(
            onset_utc=t_ev.strftime("%Y-%m-%d %H:%M"),
            static_node=[name(k, 2) for k in s_lab],
            evsom_node=[name(k, 2) for k in e_lab],
            reclassified=s_lab != e_lab,
        )
    ).to_csv(os.path.join(out_dir, "events_static_vs_evsom.csv"), index=False)

    # ── 4. All-days pathways (5x3, representative member) ──────────────────
    xd, yd = ALLDAYS_GRID
    A = ensemble("alldays")
    ra, ra_ari = representative(A)
    daily = xr.open_dataarray(f"{P}era5_Z500_norm_weighted_daily.nc")
    days = pd.DatetimeIndex(daily[daily.dims[0]].values).normalize()
    is_ff = np.isin(days, t_ev.normalize())
    a_lab, flips = orient_alldays(ra["labels"], is_ff, xd, yd)
    nn = xd * yd
    tot = np.bincount(a_lab, minlength=nn)
    ffc = np.bincount(a_lab[is_ff], minlength=nn)
    p0 = is_ff.mean()
    risk = ffc / np.maximum(tot, 1) / p0
    metrics["alldays"] = dict(
        seed=ra["seed"],
        mean_ari=ra_ari,
        orientation=flips,
        n_ff_days_matched=int(is_ff.sum()),
        risk_ratio={name(k, yd): float(risk[k]) for k in range(nn)},
        n_days={name(k, yd): int(tot[k]) for k in range(nn)},
        n_ff={name(k, yd): int(ffc[k]) for k in range(nn)},
    )
    day_idx = {d: i for i, d in enumerate(days)}
    lags = [-3, -2, -1, 0]
    path = np.full((n_ev, len(lags)), -1)
    for e, t in enumerate(t_ev):
        for c, lag in enumerate(lags):
            i = day_idx.get(t.normalize() + pd.Timedelta(days=lag))
            if i is not None:
                path[e, c] = a_lab[i]
    hi = risk >= 2.0
    pw = {}
    for k in range(4):
        sel = e_lab == k
        if not sel.any():
            continue
        d = {}
        for c, lag in enumerate(lags):
            v = path[sel, c]
            v = v[v >= 0]
            top = np.bincount(v, minlength=nn)
            d[f"d{lag:+d}"] = dict(
                n=int(len(v)),
                share_in_high_risk_nodes=float(hi[v].mean()) if len(v) else None,
                mean_risk_ratio=float(risk[v].mean()) if len(v) else None,
                top_nodes={
                    name(j, yd): int(top[j]) for j in np.argsort(-top)[:3] if top[j]
                },
            )
        pw[name(k, 2)] = d
    metrics["pathways"] = dict(
        high_risk_nodes=[name(k, yd) for k in np.nonzero(hi)[0]], **pw
    )
    plot_pathways(
        path, e_lab, risk, xd, yd, lags, os.path.join(out_dir, f"pathways_h{H}"), H
    )

    # ── 4b. Independent variables: which grouping separates them better? ────
    metrics["independent_test"] = independent_test(t_ev, s_lab, e_lab)

    # ── 5. Lead-time test (in-sample) ───────────────────────────────────────
    X = load_X("evsom", "v2", H)
    W = re_["weights"].astype(np.float64)
    nh = len(offs)
    feat_hour = np.tile(np.repeat(offs, X.shape[1] // (2 * nh)), 2)
    chance = float(((ev_counts / n_ev) ** 2).sum())
    lead = {}
    for k in range(0, H, 6):
        mask = feat_hour <= -k
        Xm = X[:, mask]
        d = np.stack([((Xm - W[j, mask]) ** 2).sum(1) for j in range(len(W))], 1)
        raw = d.argmin(1)
        named = np.array([e_perm[x] for x in raw])
        lead[f"T-{k}"] = dict(
            n_hours_used=int(nh - k),
            match_full=float((named == e_lab).mean()),
            by_node={
                name(j, 2): float((named[e_lab == j] == j).mean())
                for j in range(4)
                if (e_lab == j).any()
            },
        )
    metrics["lead_time"] = dict(chance=chance, in_sample=True, cutoffs=lead)

    with open(os.path.join(out_dir, "phase2_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    write_summary(metrics, os.path.join(out_dir, "phase2_summary.md"))
    print(f"Wrote {out_dir}")


# Variables neither SOM was trained on (Z500 and theta-e are the inputs). TCWV
# is not an input but is correlated with theta-e, so it is reported separately.
INDEPENDENT = {
    "area_ge10_km2": "Stage IV area >= 10 mm/h",
    "area_ge25_km2": "Stage IV area >= 25 mm/h",
    "s4_region_peak_mm": "Stage IV regional peak rate",
    "s4_nyc_peak_mm": "Stage IV NYC peak rate",
    "asos_peak_mm": "ASOS peak hourly rate",
    "tc": "TC within 48 h (D7 rule)",
}
RELATED = {"tcwv_T0": "TCWV at T0", "tcwv_day": "TCWV daily mean"}


def _effect(g, y, binary):
    """Rank epsilon^2 (Kruskal-Wallis H / (n - 1)) or Cramer's V (binary y)."""
    ok = ~np.isnan(y)
    g, y = g[ok], y[ok]
    if binary:
        C = pd.crosstab(g, y).values
        if C.shape[1] < 2 or C.shape[0] < 2:
            return np.nan
        chi2 = chi2_contingency(C, correction=False)[0]
        return float(np.sqrt(chi2 / (C.sum() * (min(C.shape) - 1))))
    groups = [y[g == k] for k in np.unique(g)]
    groups = [x for x in groups if len(x)]
    if len(groups) < 2:
        return np.nan
    return float(kruskal(*groups).statistic / (len(y) - 1))


def independent_test(t_ev, s_lab, e_lab, n_boot=2000, n_perm=2000):
    d2 = pd.read_csv(os.path.join(ROOT, "results", "d2", "d2_events.csv"))
    tc = pd.read_csv(
        os.path.join(ROOT, "data", "som_2x2_evsom_24h_tc_associations.csv")
    )
    ts = pd.DatetimeIndex(t_ev).strftime("%Y-%m-%d %H:%M:%S")
    assert (d2["timestamp"].values == ts).all(), "d2_events.csv order differs"
    assert (tc["timestamp"].values == ts).all(), "TC associations order differs"
    d2["tc"] = tc["tc_present"].astype(float).values
    rng = np.random.default_rng(0)
    n = len(ts)
    out = {}
    for col, label in {**INDEPENDENT, **RELATED}.items():
        y = d2[col].to_numpy(float)
        binary = col == "tc"
        es, ee = _effect(s_lab, y, binary), _effect(e_lab, y, binary)
        diff = []
        for _ in range(n_boot):
            i = rng.integers(0, n, n)
            diff.append(
                _effect(e_lab[i], y[i], binary) - _effect(s_lab[i], y[i], binary)
            )
        diff = np.array(diff)
        diff = diff[~np.isnan(diff)]
        null = np.array(
            [_effect(rng.permutation(e_lab), y, binary) for _ in range(n_perm)]
        )
        out[col] = dict(
            label=label,
            independent=col in INDEPENDENT,
            n=int((~np.isnan(y)).sum()),
            effect="cramers_v" if binary else "epsilon2",
            static=es,
            evsom=ee,
            p_static=float((null >= es).mean()),
            p_evsom=float((null >= ee).mean()),
            diff_ci95=[
                float(np.percentile(diff, 2.5)),
                float(np.percentile(diff, 97.5)),
            ],
            share_evsom_higher=float((diff > 0).mean()),
        )
    return out


PATH_MIN_N = 2  # moves shared by fewer events are drawn as faint lines
PATH_END_C = "#b2182b"
PATH_BG = mcolors.LinearSegmentedColormap.from_list(
    "risk_bg", plt.cm.Greys(np.linspace(0.0, 0.45, 256))
)  # light, so arrows read on every node


def plot_pathways(path, e_lab, risk, xd, yd, lags, out, H):
    """Day-to-day moves through the all-days SOM over the evSOM window.

    path: (event, len(lags)) all-days node per event and day (-1 = missing).
    Only the days the H-hour window spans are drawn. Writes out.png / out.pdf.
    """
    win = [c for c, lag in enumerate(lags) if lag >= -(H // 24)]
    pw, wl = path[:, win], [lags[c] for c in win]
    n_step = len(wl) - 1
    step_c = [plt.cm.Blues(v) for v in np.linspace(0.95, 0.55, n_step)[::-1]]

    def day(lag):
        return f"d{lag}".replace("-", "−")

    step_lbl = [f"{day(a)} → {day(b)}" for a, b in zip(wl[:-1], wl[1:], strict=True)]

    def xy(k):
        i, j = divmod(int(k), yd)
        return float(i), float(j)

    with plt.rc_context({"font.size": 7}):
        _draw_pathways(pw, wl, e_lab, risk, xd, yd, out, H, step_c, step_lbl, xy, day)


def _draw_pathways(pw, wl, e_lab, risk, xd, yd, out, H, step_c, step_lbl, xy, day):
    n_step = len(wl) - 1
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.9), layout="constrained")
    g = np.zeros((yd, xd))
    for k in range(xd * yd):
        i, j = divmod(k, yd)
        g[j, i] = risk[k]
    for k, ax in enumerate(axes.T.ravel()):  # A1, B1, A2, B2
        im = ax.imshow(g, cmap=PATH_BG, vmin=0, vmax=3, origin="upper", zorder=0)
        for kk in range(xd * yd):
            i, j = divmod(kk, yd)
            ax.text(
                i - 0.45,
                j - 0.40,
                name(kk, yd),
                ha="left",
                va="top",
                fontsize=5.5,
                color="0.3",
                zorder=1,
            )
            if risk[kk] >= 2:
                ax.add_patch(
                    Rectangle(
                        (i - 0.5, j - 0.5), 1, 1, fill=False, ec="k", lw=1.1, zorder=1
                    )
                )
        sel = np.nonzero(e_lab == k)[0]
        for s in range(n_step):
            mv = Counter(
                (int(pw[e, s]), int(pw[e, s + 1]))
                for e in sel
                if pw[e, s] >= 0 and pw[e, s + 1] >= 0
            )
            for (a, b), n in sorted(mv.items(), key=lambda t: t[1]):
                if a == b:
                    continue
                if n < PATH_MIN_N:
                    ax.add_patch(
                        FancyArrowPatch(
                            xy(a),
                            xy(b),
                            arrowstyle="-",
                            connectionstyle="arc3,rad=0.15",
                            lw=0.5,
                            color="0.55",
                            alpha=0.35,
                            shrinkA=6,
                            shrinkB=6,
                            zorder=2,
                        )
                    )
                    continue
                ax.add_patch(
                    FancyArrowPatch(
                        xy(a),
                        xy(b),
                        arrowstyle="-|>",
                        connectionstyle="arc3,rad=0.15",
                        mutation_scale=6 + 1.2 * n,
                        lw=0.8 + 0.7 * n,
                        color=step_c[s],
                        shrinkA=6,
                        shrinkB=8,
                        zorder=3 + s,
                    )
                )
        end = Counter(int(v) for v in pw[sel, -1] if v >= 0)
        for nd, n in end.items():
            x, y = xy(nd)
            ax.scatter(x, y, s=40 + 7 * n, fc="w", ec=PATH_END_C, lw=1.0, zorder=7)
            ax.text(
                x,
                y,
                str(n),
                ha="center",
                va="center",
                fontsize=5.5,
                color=PATH_END_C,
                weight="bold",
                zorder=8,
            )
        ax.set_xticks(range(xd), [str(i + 1) for i in range(xd)])
        ax.set_yticks(range(yd), [chr(65 + j) for j in range(yd)])
        ax.tick_params(length=0)
        ax.set_title(f"evSOM {name(k, 2)} (n = {len(sel)})", fontsize=8)
    handles = [
        Line2D([], [], color=c, lw=2, label=f"{lbl}, ≥{PATH_MIN_N} events")
        for c, lbl in zip(step_c, step_lbl, strict=True)
    ]
    handles += [
        Line2D(
            [], [], color="0.55", alpha=0.5, lw=0.8, label="move by 1 event (no arrow)"
        ),
        Patch(fc="none", ec="k", lw=1.1, label="node risk ≥ 2×"),
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            mfc="w",
            mec=PATH_END_C,
            ms=7,
            label="events in node on d0",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=(len(handles) + 1) // 2,
        fontsize=6.5,
        frameon=False,
    )
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.01)
    cb.set_label("all-days node FF risk / climatology")
    fig.suptitle(
        f"All-days 5×3 SOM node on UTC days {day(wl[0])}…d0 of the {H}-h evSOM window",
        fontsize=8,
    )
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=300)
    plt.close(fig)


def write_summary(mt, out):
    b = mt["seed_baseline"]
    L = [f"# Phase 2 (D1): {mt['hours']}-h evSOM\n"]
    L.append("## Seed baseline (ARI median [min–max])\n")
    for k in ("static_static", "evsom_evsom", "static_evsom"):
        a, g = b[k]["ari"], b[k]["agreement"]
        L.append(
            f"- {k}: ARI {a['median']:.2f} [{a['min']:.2f}–{a['max']:.2f}], "
            f"agreement {g['median']:.2f} [{g['min']:.2f}–{g['max']:.2f}]"
        )
    L.append(
        f"- share of cross-method pairs above the within-evSOM 5th percentile: "
        f"{b['cross_above_within_evsom_p5']:.2f}\n"
    )
    r = mt["representative"]
    L.append(
        f"Representative members: static seed {r['static']['seed']} "
        f"(agreement with published {r['static']['agreement_with_published']:.2f}); "
        f"evSOM seed {r['evsom']['seed']} (agreement with published 24 h "
        f"{r['evsom']['agreement_with_published_24h']:.2f}). evSOM node counts: "
        f"{mt['evsom_node_counts']}.\n"
    )
    L.append(f"## Reclassified events: {mt['n_reclassified']}\n")
    L.append("| static → evSOM | " + " | ".join(mt["static_to_evsom"]) + " |")
    L.append("|---" * 5 + "|")
    for a, row in mt["static_to_evsom"].items():
        L.append(f"| {a} | " + " | ".join(str(v) for v in row.values()) + " |")
    L.append("\nMean standardized anomalies (box means) by evSOM node:\n")
    for nd, prof in mt["evsom_node_profiles"].items():
        L.append(
            f"- **{nd}**: "
            + "; ".join(
                f"{k} " + ", ".join(f"{h} {v:+.2f}" for h, v in d.items())
                for k, d in prof.items()
            )
        )
    L.append("\nReclassifications:\n")
    for t, d in sorted(mt["reclassified_profiles"].items(), key=lambda kv: -kv[1]["n"]):
        L.append(
            f"- **{t}** (n={d['n']}): "
            + "; ".join(
                f"{k} " + ", ".join(f"{h} {v:+.2f}" for h, v in dd.items())
                for k, dd in d.items()
                if k != "n"
            )
        )
    a = mt["alldays"]
    L.append(
        f"\n## All-days pathways (5×3 seed {a['seed']}, {a['n_ff_days_matched']} FF days)\n"
    )
    L.append(f"High-risk nodes (≥2×): {mt['pathways']['high_risk_nodes']}\n")
    for nd, d in mt["pathways"].items():
        if nd == "high_risk_nodes":
            continue
        L.append(
            f"- **{nd}**: "
            + "; ".join(
                f"{lag} {v['share_in_high_risk_nodes']:.0%} high-risk (top {v['top_nodes']})"
                for lag, v in d.items()
                if v["n"]
            )
        )
    L.append("\n## Independent variables: static vs evSOM grouping\n")
    L.append(
        "Effect size = rank ε² (Kruskal–Wallis) or Cramér's V (TC). p = permutation "
        "p of each grouping; Δ CI = paired bootstrap 95% CI of evSOM − static.\n"
    )
    L.append("| variable | n | static | evSOM | Δ 95% CI | P(evSOM > static) |")
    L.append("|---|---|---|---|---|---|")
    for v in mt["independent_test"].values():
        tag = "" if v["independent"] else " (related to θe)"
        L.append(
            f"| {v['label']}{tag} | {v['n']} | {v['static']:.3f} (p={v['p_static']:.3f}) "
            f"| {v['evsom']:.3f} (p={v['p_evsom']:.3f}) | "
            f"[{v['diff_ci95'][0]:+.3f}, {v['diff_ci95'][1]:+.3f}] | {v['share_evsom_higher']:.2f} |"
        )
    lt = mt["lead_time"]
    L.append(f"\n## Lead-time test (in-sample; chance = {lt['chance']:.2f})\n")
    for k, v in lt["cutoffs"].items():
        L.append(
            f"- {k} ({v['n_hours_used']} h used): {v['match_full']:.2f} match; by node "
            + ", ".join(f"{n} {x:.2f}" for n, x in v["by_node"].items())
        )
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
