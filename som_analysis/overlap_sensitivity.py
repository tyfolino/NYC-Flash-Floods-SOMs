"""
Overlapping evSOM windows: does dropping them change the evSOM?

Events are de-duplicated per UTC calendar day, which does not stop events on
consecutive days from sharing hours of their evSOM windows. With 48-h windows,
5 pairs of the 118 events are < 48 h apart (1 pair with 24-h windows). This
script retrains the evSOM without the later event of each overlapping pair,
with the settings of the final ensemble (v2 intermediates, 5x length,
lr 0.5 -> 0.02, Gaussian sigma2 = 1, random init) for several seeds, and
compares the node assignment of the remaining events with the representative
full-data evSOM (published node names). The yardstick is the seed-to-seed
spread of the full-data ensemble on the same events: if dropping the
overlaps moves events no more than changing the seed does, the overlaps do
not matter.

Usage (BLAS threads capped; one process per seed):
    OMP_NUM_THREADS=1 nice -n 10 python -m som_analysis.overlap_sensitivity \
        --hours 48 --seeds 10 --workers 10

Outputs: /mnt/drive2/SOM_robustness/overlap/h{H}_seed{S}.npz (labels,
weights) and results/phase2/overlap_h{H}/{summary.md, metrics.json}.
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
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402

from som_analysis.evsom_phase2 import (  # noqa: E402
    ensemble,
    match_to,
    name,
    published_labels,
    representative,
)
from som_analysis.robustness_experiment import (  # noqa: E402
    INTER,
    OUT_DIR,
    ROOT,
    load_X,
)
from som_analysis.training import train_two_phase  # noqa: E402

RUN_DIR = os.path.join(OUT_DIR, "overlap")
_X = None  # training matrix, set in the parent before forking


def event_times(hours):
    suf = "" if hours == 24 else f"{hours}h"
    f = f"{INTER['v2']}era5_Z500_norm_weighted_ffe_evsom{suf}.nc"
    with xr.open_dataarray(f) as da:
        return pd.DatetimeIndex(da["event_time"].values)


def overlapping(t, hours):
    """Indices (in file order) of the later event of each pair of time-
    consecutive events less than `hours` apart, and the pairs."""
    order = np.argsort(t.values)
    ts = t[order]
    gap_h = np.diff(ts.values).astype("timedelta64[m]").astype(float) / 60
    pairs = [
        (int(order[k]), int(order[k + 1]), float(gap_h[k]))
        for k in np.nonzero(gap_h < hours)[0]
    ]
    return sorted({b for _, b, _ in pairs}), pairs


def train(seed, keep):
    X = _X[keep]
    args = SimpleNamespace(
        seed=seed,
        init="random",
        mode="random",
        n1=1000,
        n2=1000,
        length=5,
        lr1=0.5,
        lr2=0.02,
        sigma1=None,
        sigma2=1.0,
        neighborhood="gaussian",
        topology="rectangular",
    )
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        som, _ = train_two_phase(X, 2, 2, args)
    w = som.get_weights().reshape(4, -1)
    # BMUs of all 118 events (kept ones = training data; dropped ones projected)
    d = ((_X[:, None, :] - w[None, :, :]) ** 2).sum(-1)
    return seed, d.argmin(1).astype(np.int16), w.astype(np.float32), time.time() - t0


def q(x):
    x = np.asarray(x, float)
    return dict(median=float(np.median(x)), min=float(x.min()), max=float(x.max()))


def main():
    global _X
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=int, default=48)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--workers", type=int, default=10)
    a = p.parse_args()
    H = a.hours

    t = event_times(H)
    drop, pairs = overlapping(t, H)
    keep = np.setdiff1d(np.arange(len(t)), drop)
    print(
        f"{len(t)} events; {len(pairs)} overlapping pairs; dropping {drop}", flush=True
    )

    # Reference: representative full-data member, published node names.
    runs = ensemble("evsom", H)
    rep, _ = representative(runs)
    pub = published_labels("evsom")
    ref = match_to(rep["labels"], pub)[0]

    os.makedirs(RUN_DIR, exist_ok=True)
    todo = [
        s
        for s in range(a.seeds)
        if not os.path.exists(os.path.join(RUN_DIR, f"h{H}_seed{s}.npz"))
    ]
    if todo:
        _X = load_X("evsom", "v2", H)
        ctx = mp.get_context("fork")  # share _X copy-on-write
        with ProcessPoolExecutor(min(a.workers, len(todo)), mp_context=ctx) as ex:
            for s, lab, w, sec in ex.map(train, todo, [keep] * len(todo)):
                np.savez_compressed(
                    os.path.join(RUN_DIR, f"h{H}_seed{s}.npz"),
                    labels=lab,
                    weights=w,
                    keep=keep,
                    seconds=sec,
                )
                print(f"seed {s} done in {sec:.0f} s", flush=True)

    # Drop-overlap runs vs the reference, on the kept events
    sub_agree, sub_ari, dropped_same = [], [], []
    for s in range(a.seeds):
        z = np.load(os.path.join(RUN_DIR, f"h{H}_seed{s}.npz"))
        lab = z["labels"].astype(int)
        m, perm, agr = match_to(lab[keep], ref[keep])
        sub_agree.append(agr)
        sub_ari.append(ari(lab[keep], ref[keep]))
        lab_named = np.array([perm[x] for x in lab])
        dropped_same.append(float((lab_named[drop] == ref[drop]).mean()))
    # Yardstick: full-data members vs the reference, on the same kept events
    base_agree, base_ari = [], []
    for r in runs:
        if r["seed"] == rep["seed"]:
            continue
        base_agree.append(match_to(r["labels"][keep], ref[keep])[2])
        base_ari.append(ari(r["labels"][keep], ref[keep]))

    pair_rows = [
        dict(
            first=str(t[i]),
            second=str(t[j]),
            gap_h=g,
            node_first=name(int(ref[i]), 2),
            node_second=name(int(ref[j]), 2),
        )
        for i, j, g in pairs
    ]
    out = dict(
        hours=H,
        n_events=len(t),
        n_pairs=len(pairs),
        dropped=[str(t[k]) for k in drop],
        reference_seed=rep["seed"],
        seeds=a.seeds,
        pairs=pair_rows,
        drop_overlap_vs_reference=dict(agreement=q(sub_agree), ari=q(sub_ari)),
        full_ensemble_vs_reference=dict(agreement=q(base_agree), ari=q(base_ari)),
        dropped_events_same_node_when_projected=q(dropped_same),
    )
    od = os.path.join(ROOT, "results", "phase2", f"overlap_h{H}")
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "metrics.json"), "w") as f:
        json.dump(out, f, indent=1)
    S, B = out["drop_overlap_vs_reference"], out["full_ensemble_vs_reference"]
    lines = [
        f"# Overlapping {H}-h evSOM windows: sensitivity",
        "",
        f"{len(pairs)} of {len(t)} events are < {H} h after the previous event, so their "
        "windows share hours with it:",
        "",
        "| first event | second event | gap (h) | evSOM nodes |",
        "|---|---|---|---|",
        *[
            f"| {r['first'][:13]}Z | {r['second'][:13]}Z | {r['gap_h']:.0f} | "
            f"{r['node_first']}, {r['node_second']} |"
            for r in pair_rows
        ],
        "",
        f"Retrained without the second event of each pair ({len(keep)} events, "
        f"{a.seeds} seeds, final-ensemble settings) and compared with the "
        f"representative full-data evSOM (seed {rep['seed']}) on the {len(keep)} events:",
        "",
        "| comparison | matched agreement, median [min–max] | ARI, median [min–max] |",
        "|---|---|---|",
        f"| without overlaps vs reference | {S['agreement']['median']:.3f} "
        f"[{S['agreement']['min']:.3f}–{S['agreement']['max']:.3f}] | "
        f"{S['ari']['median']:.2f} [{S['ari']['min']:.2f}–{S['ari']['max']:.2f}] |",
        f"| full-data ensemble ({len(base_agree)} other seeds) vs reference | "
        f"{B['agreement']['median']:.3f} [{B['agreement']['min']:.3f}–"
        f"{B['agreement']['max']:.3f}] | {B['ari']['median']:.2f} "
        f"[{B['ari']['min']:.2f}–{B['ari']['max']:.2f}] |",
        "",
        "Dropped events projected onto the retrained SOMs land in their reference node "
        f"in a median {out['dropped_events_same_node_when_projected']['median']:.0%} "
        "of seeds' assignments.",
        "",
    ]
    with open(os.path.join(od, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
