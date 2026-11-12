"""
SOM training-method robustness experiment (GRL revision, plan section 0.5).

Retrains the three published SOMs (all-days 5x4, static FF 2x2, evSOM 2x2) under
varied training settings and many random seeds, saving per-run node assignments
(and, for the all-days SOM, weights) plus summary metrics. It does not touch the
published caches or CSVs.

Training replicates the published code path exactly (verified bit-identical for
seed 42 against the cached weights in the audit), with these factors added:
  - length   : phase-1/phase-2 iterations as a multiple of the published counts
  - init     : 'random' (published) or 'pca' (2 leading PCs, fixed)
  - mode     : 'random' (MiniSom train_random, published) or 'batch' (train_batch)
  - sigma2   : phase-2 Gaussian neighbourhood radius (published 1.0)

Compute limits (shared 64-core server): every worker is single-threaded
(BLAS/OpenMP threads = 1) and the pool is capped at --workers (default 16).

Usage (from the project root):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 7200 nice -n 10 \
        python -m som_analysis.robustness_experiment --stage 1 --workers 16
    python -m som_analysis.robustness_experiment --summarize
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
import itertools  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from minisom import MiniSom  # noqa: E402

from som_analysis.config import (  # noqa: E402
    MOISTURE_CONFIGS,
    SOM_INTERMEDIATE_PATH_V1,
    SOM_INTERMEDIATE_PATH_V2,
)
from som_analysis.helpers import load_moist_var  # noqa: E402

OUT_DIR = "/mnt/drive2/SOM_robustness"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTER = {"v1": SOM_INTERMEDIATE_PATH_V1, "v2": SOM_INTERMEDIATE_PATH_V2}
CFG = MOISTURE_CONFIGS["thetae"]

# Published training parameters (Table S1 / training scripts)
PUBLISHED = {
    "alldays": dict(xdim=5, ydim=4, lr1=0.3, lr2=0.1, n1=5000, n2=5000),
    "static": dict(xdim=2, ydim=2, lr1=0.5, lr2=0.1, n1=1000, n2=1000),
    "evsom": dict(xdim=2, ydim=2, lr1=0.5, lr2=0.1, n1=1000, n2=1000),
}
PUBLISHED_CACHE = {
    "alldays": "figs/Z500-and-thetae-alldays-SOM/.cache/som_results.npz",
    "static": "figs/Z500-and-thetae-SOM/.cache/som_results.npz",
    "evsom": "figs/Z500-and-thetae-evSOM-24h/.cache/som_results.npz",
}

_X_CACHE: dict = {}


# ── Data ──────────────────────────────────────────────────────────────────────


def load_X(kind, inter="v1"):
    """Training matrix for a SOM kind, exactly as the training scripts build it.

    inter: "v1" (published intermediates) or "v2" (month-day, smoothed-mean
    climatology).
    """
    if (kind, inter) in _X_CACHE:
        return _X_CACHE[(kind, inter)]
    P = INTER[inter]
    if kind == "static":
        m = load_moist_var(f"{P}era5_thetae_norm_weighted_ffe.nc", CFG["var_name"])
        z = xr.load_dataarray(f"{P}era5_Z500_norm_weighted_ffe.nc")
        X = np.concatenate(
            (
                z.stack(features=["latitude", "longitude"]).values,
                m.stack(features=["latitude", "longitude"]).values,
            ),
            axis=1,
        )
    elif kind == "evsom":
        m = load_moist_var(
            f"{P}era5_thetae_norm_weighted_ffe_evsom.nc", CFG["var_name"]
        )
        z = xr.load_dataarray(f"{P}era5_Z500_norm_weighted_ffe_evsom.nc")
        n = z.sizes["event_time"]
        X = np.concatenate((z.values.reshape(n, -1), m.values.reshape(n, -1)), axis=1)
    elif kind == "alldays":
        m = load_moist_var(f"{P}era5_thetae_norm_weighted_daily.nc", CFG["var_name"])
        z = xr.load_dataarray(f"{P}era5_Z500_norm_weighted_daily.nc")
        X = np.concatenate(
            (
                z.stack(features=["latitude", "longitude"]).values,
                m.stack(features=["latitude", "longitude"]).values,
            ),
            axis=1,
        )
    else:
        raise ValueError(kind)
    _X_CACHE[(kind, inter)] = X
    return X


def pca_init_path(kind):
    return os.path.join(OUT_DIR, f"pca_init_{kind}.npy")


def build_pca_inits(kinds):
    """Fixed PCA initial weights per kind (deterministic; computed once)."""
    from sklearn.decomposition import PCA

    for kind in kinds:
        path = pca_init_path(kind)
        if os.path.exists(path):
            continue
        X = load_X(kind)
        p = PUBLISHED[kind]
        pca = PCA(n_components=2, svd_solver="randomized", random_state=0).fit(X)
        w = np.zeros((p["xdim"], p["ydim"], X.shape[1]), dtype=np.float64)
        for i, c1 in enumerate(np.linspace(-1, 1, p["xdim"])):
            for j, c2 in enumerate(np.linspace(-1, 1, p["ydim"])):
                w[i, j] = pca.mean_ + c1 * pca.components_[0] + c2 * pca.components_[1]
        np.save(path, w)
        print(f"Saved PCA init for {kind}: {path}", flush=True)


# ── One run ───────────────────────────────────────────────────────────────────


# Stage-2 settings; defaults reproduce the published/stage-1 setup, so stage-1
# run IDs are unchanged and stage-1 runs serve as the stage-2 baselines.
DEFAULTS = dict(
    lr1=None,
    lr2=None,
    nbh="gaussian",
    topo="rectangular",
    sigma1=None,
    n1=None,
    n2=None,
    inter="v1",
    xdim=None,
    ydim=None,
)


def full(c):
    return {**DEFAULTS, **c}


def run_id(c):
    rid = (
        f"{c['kind']}_len{c['length']}_init-{c['init']}_mode-{c['mode']}"
        f"_s2-{c['sigma2']}"
    )
    for k, v in DEFAULTS.items():
        if c.get(k, v) != v:
            rid += f"_{k}-{c[k]}"
    return rid + f"_seed{c['seed']}"


def train_one(c):
    """Train one SOM configuration; save labels (+weights) and metrics."""
    rid = run_id(c)
    out = os.path.join(OUT_DIR, "runs", f"{rid}.npz")
    if os.path.exists(out):
        return rid, "exists"
    t0 = time.time()
    c = full(c)
    kind = c["kind"]
    X = load_X(kind, c["inter"])
    p = PUBLISHED[kind]
    xdim = p["xdim"] if c["xdim"] is None else c["xdim"]
    ydim = p["ydim"] if c["ydim"] is None else c["ydim"]
    lr1 = p["lr1"] if c["lr1"] is None else c["lr1"]
    lr2 = p["lr2"] if c["lr2"] is None else c["lr2"]
    sigma1 = np.sqrt(xdim**2 + ydim**2) if c["sigma1"] is None else c["sigma1"]
    if c["mode"] == "offline":
        # True (Kohonen 2013) batch SOM: n1/n2 are full passes over the data,
        # learning rate held constant (1.0 = pure batch averaging).
        n1, n2 = int(c["n1"]), int(c["n2"])
    else:
        n1 = int((c["n1"] or p["n1"]) * c["length"])
        n2 = int((c["n2"] or p["n2"]) * c["length"])

    som = MiniSom(
        xdim,
        ydim,
        input_len=X.shape[1],
        sigma=sigma1,
        learning_rate=lr1,
        decay_function="linear_decay_to_zero",
        sigma_decay_function="linear_decay_to_one",
        neighborhood_function=c["nbh"],
        topology=c["topo"],
        random_seed=c["seed"],
    )
    if c["init"] == "pca":
        som._weights = np.load(pca_init_path(kind)).copy()
    else:
        som.random_weights_init(X)

    if c["mode"] == "offline":
        som._learning_rate_decay_function = lambda lr, t, T: lr
        train = som.train_batch_offline
    else:
        # NB: MiniSom's train_batch is sequential training in fixed
        # (chronological) order, not batch SOM.
        train = som.train_random if c["mode"] == "random" else som.train_batch
    train(X, n1)
    # Phase 2: constant radius sigma2 (linear_decay_to_one keeps sigma=1 constant;
    # for sigma2 != 1 we switch to a constant-sigma decay), smaller learning rate.
    som._sigma = float(c["sigma2"])
    if c["sigma2"] != 1.0:
        som._sigma_decay_function = lambda s, t, T: s
    som._learning_rate = lr2
    train(X, n2)

    W = som.get_weights()
    d = som._distance_from_weights(X)
    lab = d.argmin(1).astype(np.int16)  # flat index i*ydim + j
    qe = float(som.quantization_error(X))
    te = float(som.topographic_error(X)) if xdim * ydim > 4 else float("nan")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save = dict(
        labels=lab, qe=qe, te=te, config=json.dumps(c), seconds=time.time() - t0
    )
    save["weights"] = W.reshape(xdim * ydim, -1).astype(np.float32)
    np.savez_compressed(out, **save)
    return rid, f"done in {time.time() - t0:.0f}s (QE={qe:.2f})"


# ── Experiment design ─────────────────────────────────────────────────────────


def stage1_configs():
    """Stage 1: training length x initialization x mode (all-days); radius,
    length, init (2x2 SOMs, which are already ~8.5 epochs per phase)."""
    cfgs = []
    # All-days 5x4: published length = 0.9 epoch per phase -> test 1x, 5x, 10x.
    for length, init, seed in itertools.product(
        [1, 5, 10], ["random", "pca"], range(8)
    ):
        if init == "pca" and seed > 3:
            continue  # PCA init: only the sample order varies; 4 seeds suffice
        cfgs.append(
            dict(
                kind="alldays",
                length=length,
                init=init,
                mode="random",
                sigma2=1.0,
                seed=seed,
            )
        )
    # Batch training at 10x: seeds only change the random init.
    for seed in range(4):
        cfgs.append(
            dict(
                kind="alldays",
                length=10,
                init="random",
                mode="batch",
                sigma2=1.0,
                seed=seed,
            )
        )
    cfgs.append(
        dict(kind="alldays", length=10, init="pca", mode="batch", sigma2=1.0, seed=0)
    )
    # 2x2 SOMs: phase-2 radius x length x init, 10 seeds each.
    for kind, s2, length, init, seed in itertools.product(
        ["static", "evsom"], [1.0, 0.5], [1, 5], ["random", "pca"], range(10)
    ):
        cfgs.append(
            dict(
                kind=kind, length=length, init=init, mode="random", sigma2=s2, seed=seed
            )
        )
    # Order: long all-days runs first so the pool stays busy to the end.
    cfgs.sort(key=lambda c: -(c["length"] * (20 if c["kind"] == "alldays" else 1)))
    return cfgs


def stage2_configs():
    """Stage 2: learning rate, neighbourhood function/radius, topology, initial
    radius and true batch training. Random init. Baselines come from stage 1
    (all-days length 5 seeds 0-7; 2x2 length 1 seeds 0-9)."""
    cfgs = []
    base = dict(init="random", mode="random", sigma2=1.0)
    alldays = [
        dict(lr1=0.5, lr2=0.1),
        dict(lr1=0.1, lr2=0.05),
        dict(lr1=0.3, lr2=0.02),
        dict(sigma2=0.5),
        dict(sigma2=1.5),
        dict(nbh="bubble", sigma2=1.0),  # phase 2 = winner only (k-means-like)
        dict(nbh="bubble", sigma2=2.0),  # phase 2 = winner + first ring
        dict(topo="hexagonal"),
        dict(topo="hexagonal", lr2=0.02),
        dict(sigma1=3.0),
    ]
    for v, seed in itertools.product(alldays, range(6)):
        cfgs.append({**base, "kind": "alldays", "length": 5, **v, "seed": seed})
    for init, seed in itertools.product(["random", "pca"], range(6)):
        if init == "pca" and seed > 0:
            continue  # offline batch + PCA init is fully deterministic
        cfgs.append(
            dict(
                kind="alldays",
                length=1,
                init=init,
                mode="offline",
                sigma2=1.0,
                lr1=1.0,
                lr2=1.0,
                n1=30,
                n2=20,
                seed=seed,
            )
        )
    small = [
        dict(lr1=0.3, lr2=0.1),
        dict(lr1=0.5, lr2=0.02),
        dict(nbh="bubble", sigma2=1.0),
        dict(topo="hexagonal"),
        dict(mode="offline", lr1=1.0, lr2=1.0, n1=30, n2=20),
    ]
    for kind, v, seed in itertools.product(["static", "evsom"], small, range(8)):
        cfgs.append({**base, "kind": kind, "length": 1, **v, "seed": seed})
    cfgs.sort(
        key=lambda c: -(
            c["length"] * (20 if c["kind"] == "alldays" else 1)
            + (5 if c["kind"] == "evsom" else 0)
        )
    )
    return cfgs


FINAL = dict(init="random", mode="random", sigma2=1.0, inter="v2")
GRIDS = [(3, 3), (4, 3), (4, 4), (5, 3), (5, 4), (6, 4), (5, 5), (6, 5)]


def stage3_configs():
    """Stage 3a: all-days grid sweep with the stage-2 winning settings
    (lr 0.3 -> 0.02, 5x length, Gaussian sigma2=1, rectangular, random init)
    on the v2 intermediates; 10 seeds per grid."""
    cfgs = []
    for (xd, yd), seed in itertools.product(GRIDS, range(10)):
        cfgs.append(
            {
                **FINAL,
                "kind": "alldays",
                "length": 5,
                "lr1": 0.3,
                "lr2": 0.02,
                "xdim": xd,
                "ydim": yd,
                "seed": seed,
            }
        )
    cfgs.sort(key=lambda c: -c["xdim"] * c["ydim"])
    return cfgs


def stage4_configs(grid=(5, 4), n_seeds=30, length_alldays=5):
    """Stage 3b: final ensembles on v2. All-days on the chosen grid; static
    and evSOM 2x2 with lr 0.5 -> 0.02 at 1x and 5x length."""
    xd, yd = grid
    cfgs = []
    for seed in range(n_seeds):
        cfgs.append(
            {
                **FINAL,
                "kind": "alldays",
                "length": length_alldays,
                "lr1": 0.3,
                "lr2": 0.02,
                "xdim": xd,
                "ydim": yd,
                "seed": seed,
            }
        )
        for kind, length in itertools.product(["static", "evsom"], [1, 5]):
            cfgs.append(
                {
                    **FINAL,
                    "kind": kind,
                    "length": length,
                    "lr1": 0.5,
                    "lr2": 0.02,
                    "seed": seed,
                }
            )
    cfgs.sort(key=lambda c: -(c["length"] * (20 if c["kind"] == "alldays" else 1)))
    return cfgs


def grid_edges(xdim, ydim):
    """Rook-neighbour edges of a rectangular grid (flat index i * ydim + j)."""
    e = []
    for i in range(xdim):
        for j in range(ydim):
            if i + 1 < xdim:
                e.append((i * ydim + j, (i + 1) * ydim + j))
            if j + 1 < ydim:
                e.append((i * ydim + j, i * ydim + j + 1))
    return e


def edge_crossings(Y, edges):
    """Number of pairs of grid edges that cross in a 2-D projection (twisting)."""

    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

    n = 0
    for (a, b), (c, d) in itertools.combinations(edges, 2):
        if len({a, b, c, d}) < 4:
            continue
        p1, p2, p3, p4 = Y[a], Y[b], Y[c], Y[d]
        if (
            ccw(p1, p2, p3) * ccw(p1, p2, p4) < 0
            and ccw(p3, p4, p1) * ccw(p3, p4, p2) < 0
        ):
            n += 1
    return n


# ── Summary ───────────────────────────────────────────────────────────────────


def summarize():
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score as ari

    runs = []
    for f in sorted(os.listdir(os.path.join(OUT_DIR, "runs"))):
        z = np.load(os.path.join(OUT_DIR, "runs", f))
        c = json.loads(str(z["config"]))
        runs.append((c, z["labels"], z["weights"], float(z["qe"]), float(z["te"])))

    pub = {}
    for kind, rel in PUBLISHED_CACHE.items():
        cz = np.load(os.path.join(ROOT, rel), allow_pickle=True)
        ydim = PUBLISHED[kind]["ydim"]
        b = cz["bmus"]
        n_nodes = PUBLISHED[kind]["xdim"] * ydim
        pub[kind] = dict(
            labels=b[:, 0] * ydim + b[:, 1],
            weights=cz["weights"].reshape(n_nodes, -1) if "weights" in cz else None,
            is_ff=cz["is_ff"] if "is_ff" in cz else None,
        )

    groups: dict = {}
    for r in runs:
        c = full(r[0])
        key = tuple(sorted((k, v) for k, v in c.items() if k != "seed"))
        groups.setdefault(key, []).append(r)

    rows = []
    for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        cfg = dict(key)
        kind = cfg["kind"]
        labs = [r[1] for r in rs]
        pair = [ari(a, b) for a, b in itertools.combinations(labs, 2)] or [np.nan]
        vs_pub = [ari(pub[kind]["labels"], lab) for lab in labs]
        row = dict(
            {k: v for k, v in cfg.items() if v != DEFAULTS.get(k, object())},
            n=len(rs),
            ari_pair_median=float(np.median(pair)),
            ari_pair_min=float(np.min(pair)),
            ari_vs_pub_median=float(np.median(vs_pub)),
            qe_median=float(np.median([r[3] for r in rs])),
            te_median=float(np.median([r[4] for r in rs])),
        )
        if kind == "alldays":
            from som_analysis.helpers import sammon_map

            is_ff = pub["alldays"]["is_ff"].astype(bool)
            p0 = is_ff.mean()
            xd = cfg["xdim"] or PUBLISHED["alldays"]["xdim"]
            yd = cfg["ydim"] or PUBLISHED["alldays"]["ydim"]
            nn = xd * yd
            edges = grid_edges(xd, yd)
            empty, maxr, corner, cross = [], [], [], []
            pw = pub["alldays"]["weights"]
            for _c, lab, W, *_ in rs:
                assert len(lab) == len(is_ff), "day order differs from published"
                tot = np.bincount(lab, minlength=nn)
                ff = np.bincount(lab[is_ff], minlength=nn)
                risk = np.where(tot > 0, ff / np.maximum(tot, 1), 0)
                empty.append(int((ff == 0).sum()))
                maxr.append(float(risk.max() / p0))
                if pw is not None:
                    C = ((W[:, None, :].astype(np.float64) - pw[None]) ** 2).sum(-1)
                    if nn == len(pw):  # same grid: one-to-one matching
                        ri, ci = linear_sum_assignment(C)
                        m = dict(zip(ri, ci, strict=True))
                        top = m[int(risk.argmax())]
                    else:  # other grid: nearest published node
                        top = int(C[int(risk.argmax())].argmin())
                    corner.append(top)  # published flat index of top node
                Y, _ = sammon_map(W.astype(np.float64), n_restarts=5, seed=0)
                cross.append(edge_crossings(Y, edges))
            row.update(
                empty_nodes_median=float(np.median(empty)),
                empty_nodes_max=int(np.max(empty)),
                max_risk_ratio_min=float(np.min(maxr)),
                max_risk_ratio_median=float(np.median(maxr)),
                max_risk_ratio_max=float(np.max(maxr)),
                top_node_published_labels=sorted(
                    {f"{chr(65 + k % 4)}{k // 4 + 1}" for k in corner}
                ),
                # published flat indices: A4 = 12, A5 = 16
                top_node_in_A4_A5=(
                    float(np.mean([k in (12, 16) for k in corner]))
                    if corner
                    else float("nan")
                ),
                sammon_crossings_median=float(np.median(cross)),
                sammon_crossings_max=int(np.max(cross)),
            )
        else:
            # Share of the two most-populated nodes (published: A1+B2 = 69%)
            shares = [
                np.sort(np.bincount(lab, minlength=4))[-2:].sum() / len(lab)
                for lab in labs
            ]
            row.update(
                top2_share_min=float(np.min(shares)),
                top2_share_max=float(np.max(shares)),
            )
        rows.append(row)

    out = os.path.join(OUT_DIR, "summary.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    for row in rows:
        print(json.dumps(row))
    print(f"\nWrote {out}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--grid", default="5x4", help="stage 4: all-days grid, e.g. 6x4")
    ap.add_argument("--seeds", type=int, default=30, help="stage 4: ensemble size")
    args = ap.parse_args()

    if args.summarize:
        summarize()
        return
    if args.workers > 32:
        raise SystemExit("Refusing more than 32 workers on the shared server.")

    os.makedirs(os.path.join(OUT_DIR, "runs"), exist_ok=True)
    if args.stage == 1:
        cfgs = stage1_configs()
    elif args.stage == 2:
        cfgs = stage2_configs()
    elif args.stage == 3:
        cfgs = stage3_configs()
    else:
        grid = tuple(int(v) for v in args.grid.lower().split("x"))
        cfgs = stage4_configs(grid=grid, n_seeds=args.seeds)
    build_pca_inits(sorted({c["kind"] for c in cfgs if c["init"] == "pca"}))
    todo = [
        c
        for c in cfgs
        if not os.path.exists(os.path.join(OUT_DIR, "runs", f"{run_id(c)}.npz"))
    ]
    print(
        f"Stage {args.stage}: {len(cfgs)} runs, {len(todo)} to do, "
        f"{args.workers} single-threaded workers",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(train_one, c): run_id(c) for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rid = futs[fut]
            try:
                _, msg = fut.result()
            except Exception as err:  # keep going; report at the end
                msg = f"FAILED: {err!r}"
            print(f"[{i}/{len(todo)}] {rid}: {msg}", flush=True)


if __name__ == "__main__":
    main()
