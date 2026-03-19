"""
All-Days SOM Training for NYC Flash Flood Analysis
By: Ty Janoski

Trains a Self-Organizing Map on Z500 + moisture variable fields using ALL
available warm-season ERA5 days (1996-2024), then computes per-node flash
flood risk based on NOAA Storm Data events.

Usage:
    python -m som_analysis.train_alldays_som --moisture-var IVT
    python -m som_analysis.train_alldays_som --moisture-var thetae --xdim 5 --ydim 4
    python -m som_analysis.train_alldays_som --moisture-var IVT --rotate --seed 0
"""

import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr
from minisom import MiniSom
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

from .config import (
    ALLDAYS_MOISTURE_VARS,
    MOISTURE_CONFIGS,
    SOM_INTERMEDIATE_PATH,
    STORM_DATA_CSV,
    get_alldays_paths,
)
from .helpers import load_moist_var, node_label


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a SOM on all warm-season ERA5 days for flash-flood risk."
    )
    parser.add_argument(
        "--moisture-var",
        required=True,
        choices=ALLDAYS_MOISTURE_VARS,
        help="Moisture variable to use (IVT or thetae).",
    )
    parser.add_argument(
        "--moisture-weight",
        type=float,
        default=1,
        help="Scalar multiplier for moisture features (default: 1).",
    )
    parser.add_argument(
        "--xdim",
        type=int,
        default=5,
        help="Number of SOM columns (default: 5).",
    )
    parser.add_argument(
        "--ydim",
        type=int,
        default=4,
        help="Number of SOM rows (default: 4).",
    )
    parser.add_argument(
        "--n1",
        type=int,
        default=5000,
        help="Number of iterations for coarse training phase (default: 5000).",
    )
    parser.add_argument(
        "--n2",
        type=int,
        default=5000,
        help="Number of iterations for fine training phase (default: 5000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="Apply 90-degree clockwise rotation to the trained SOM.",
    )
    parser.add_argument(
        "--init",
        choices=["random", "pca"],
        default="random",
        help="Weight initialization method: 'random' or 'pca' (default: random).",
    )
    parser.add_argument(
        "--snapshot-hour",
        type=int,
        default=None,
        help="Use a single hourly snapshot per day (e.g. 20 for 2000 UTC) "
             "instead of daily means.",
    )
    parser.add_argument(
        "--topology",
        choices=["rectangular", "hexagonal"],
        default="rectangular",
        help="SOM grid topology (default: rectangular). Hexagonal reduces corner "
             "clustering by giving each interior node 6 neighbors instead of 4.",
    )
    return parser.parse_args()


def load_flash_flood_days(storm_csv):
    """Parse NOAA Storm Data CSV and return a set of UTC flash-flood dates.

    Combines BEGIN_DATE + BEGIN_TIME, localises to EST, converts to UTC,
    floors to day, and deduplicates by EPISODE_ID.
    """
    df = pd.read_csv(storm_csv)
    df = df[df["EVENT_ID"].astype(str).str.isdigit()].drop_duplicates(
        subset=["EPISODE_ID"], keep="first"
    )
    df["BEGIN_DATETIME"] = (
        pd.to_datetime(
            df["BEGIN_DATE"]
            + " "
            + df["BEGIN_TIME"].fillna(0).astype(int).astype(str).str.zfill(4),
            format="%m/%d/%Y %H%M",
            errors="coerce",
        )
        .dt.tz_localize("EST", ambiguous="NaT", nonexistent="NaT")
        .dt.tz_convert("UTC")
    )
    event_days = sorted(
        df["BEGIN_DATETIME"].dropna().dt.floor("D").dt.tz_localize(None).unique()
    )
    return event_days


def main():
    args = parse_args()
    cfg = MOISTURE_CONFIGS[args.moisture_var]
    paths = get_alldays_paths(args.moisture_var, args.xdim, args.ydim, snapshot_hour=args.snapshot_hour)

    xdim, ydim = args.xdim, args.ydim
    pfx = cfg["file_prefix"]
    var_name = cfg["var_name"]
    moist_time_dim = cfg["time_dim"]

    # ── Load data ─────────────────────────────────────────────────────────────
    print(
        f"Loading data for {args.moisture_var} (weight={args.moisture_weight})..."
    )

    if args.snapshot_hour is not None:
        alldays_suffix = f"_snapshot_{args.snapshot_hour:02d}utc"
    else:
        alldays_suffix = "_daily"

    moist_norm_weighted_daily = load_moist_var(
        f"{SOM_INTERMEDIATE_PATH}{pfx}_norm_weighted{alldays_suffix}.nc", var_name
    )
    moist_norm_daily = load_moist_var(
        f"{SOM_INTERMEDIATE_PATH}{pfx}_norm{alldays_suffix}.nc", var_name
    )
    moist_daily = load_moist_var(
        f"{SOM_INTERMEDIATE_PATH}{pfx}{alldays_suffix}.nc", var_name
    )

    z500_norm_weighted_daily = xr.load_dataarray(
        f"{SOM_INTERMEDIATE_PATH}era5_Z500_norm_weighted{alldays_suffix}.nc"
    )
    z500_norm_daily = xr.load_dataarray(
        f"{SOM_INTERMEDIATE_PATH}era5_Z500_norm{alldays_suffix}.nc"
    )
    z500_daily = xr.load_dataarray(
        f"{SOM_INTERMEDIATE_PATH}era5_Z500{alldays_suffix}.nc"
    )

    # ── Flash-flood day mask ───────────────────────────────────────────────────
    print(f"Loading flash-flood events from {STORM_DATA_CSV} ...")
    event_days = load_flash_flood_days(STORM_DATA_CSV)
    print(f"  Found {len(event_days)} unique flash-flood days.")

    z500_time_dim = "valid_time" if "valid_time" in z500_norm_weighted_daily.dims else "time"
    som_days = pd.to_datetime(z500_norm_weighted_daily[z500_time_dim].values).tz_localize(
        None
    )
    is_ff = np.isin(som_days.normalize(), pd.to_datetime(event_days))
    event_indices = np.where(is_ff)[0]
    print(
        f"  Matched {is_ff.sum()} SOM days to flash-flood events "
        f"(out of {len(som_days)} total)."
    )

    # ── Flatten & concatenate ─────────────────────────────────────────────────
    z500_lat_dim = "latitude" if "latitude" in z500_norm_weighted_daily.dims else "lat"
    z500_lon_dim = "longitude" if "longitude" in z500_norm_weighted_daily.dims else "lon"
    z500_flat = z500_norm_weighted_daily.stack(features=[z500_lat_dim, z500_lon_dim]).values
    moist_flat = moist_norm_weighted_daily.stack(
        features=[cfg["lat_dim"], cfg["lon_dim"]]
    ).values

    X = np.concatenate(
        (z500_flat, moist_flat * args.moisture_weight), axis=1
    )
    print(f"Training matrix shape: {X.shape}")

    # ── SOM training ──────────────────────────────────────────────────────────
    sig1 = np.sqrt(xdim**2 + ydim**2)
    sig2 = 1.0
    lr1, lr2 = 0.3, 0.1

    som = MiniSom(
        xdim,
        ydim,
        input_len=X.shape[1],
        sigma=sig1,
        learning_rate=lr1,
        decay_function="linear_decay_to_zero",
        sigma_decay_function="linear_decay_to_one",
        neighborhood_function="gaussian",
        topology=args.topology,
        random_seed=args.seed,
    )

    if args.init == "pca":
        pca = PCA(n_components=2, random_state=args.seed)
        pca.fit(X)
        w = np.zeros((xdim, ydim, X.shape[1]))
        for i, c1 in enumerate(np.linspace(-1, 1, xdim)):
            for j, c2 in enumerate(np.linspace(-1, 1, ydim)):
                w[i, j] = pca.mean_ + c1 * pca.components_[0] + c2 * pca.components_[1]
        som._weights = w
        print("Initialized weights via PCA.")
    else:
        som.random_weights_init(X)

    print(f"\nPhase 1: {args.n1} iterations (sigma={sig1:.2f}, lr={lr1})")
    som.train_random(X, args.n1, verbose=True)

    som._sigma = sig2  # type: ignore
    som._learning_rate = lr2
    print(f"\nPhase 2: {args.n2} iterations (sigma={sig2:.2f}, lr={lr2})")
    som.train_random(X, args.n2, verbose=True)

    # ── Extract results ───────────────────────────────────────────────────────
    weights = som.get_weights().reshape(xdim * ydim, -1)
    bmus = np.array([som.winner(x) for x in X])

    # Spatial dimensions (use Z500 coordinates as the shared plotting grid)
    lat = z500_daily[z500_lat_dim].values
    lon = z500_daily[z500_lon_dim].values
    n_lat, n_lon = lat.size, lon.size
    n_features = n_lat * n_lon

    # Split and reshape weights
    z500_nodes = weights[:, :n_features].reshape(xdim, ydim, n_lat, n_lon)
    moist_nodes = weights[:, n_features:].reshape(xdim, ydim, n_lat, n_lon)
    moist_nodes = moist_nodes / args.moisture_weight

    # ── Optional rotation ─────────────────────────────────────────────────────
    if args.rotate:
        z500_nodes = np.rot90(z500_nodes, k=-1, axes=(0, 1))
        moist_nodes = np.rot90(moist_nodes, k=-1, axes=(0, 1))
        bmus = np.column_stack([bmus[:, 1], xdim - 1 - bmus[:, 0]])

        weights_rotated = np.zeros_like(weights)
        for i in range(xdim):
            for j in range(ydim):
                idx = i * ydim + j
                weights_rotated[idx, :n_features] = z500_nodes[i, j].flatten()
                weights_rotated[idx, n_features:] = moist_nodes[i, j].flatten()
        weights = weights_rotated

        print("\nSOM rotated 90 degrees clockwise.")

    # ── U-matrix, hit map, MDS ────────────────────────────────────────────────
    u_matrix = som.distance_map().T
    hit_map = np.zeros((xdim, ydim))
    for i, j in bmus:
        hit_map[i, j] += 1
    hit_map = hit_map.T

    D = pairwise_distances(weights)
    coords = MDS(
        n_components=2,
        metric="precomputed",
        random_state=42,
        n_init=4,
        init="random",  # pyright: ignore[reportCallIssue]
    ).fit_transform(D)

    # ── Per-node flash-flood risk ─────────────────────────────────────────────
    totals = np.zeros((xdim, ydim), dtype=int)
    counts = np.zeros((xdim, ydim), dtype=int)

    for i in range(xdim):
        for j in range(ydim):
            idx_node = np.where((bmus[:, 0] == i) & (bmus[:, 1] == j))[0]
            totals[i, j] = len(idx_node)
            counts[i, j] = len(np.intersect1d(idx_node, event_indices))

    risk = np.where(totals > 0, counts / totals, np.nan)

    # ── Save BMU CSV (FF days only) ───────────────────────────────────────────
    os.makedirs(os.path.dirname(paths["bmu_csv_path"]), exist_ok=True)
    ff_timestamps = pd.to_datetime(
        z500_norm_weighted_daily[z500_time_dim].values[event_indices]
    )
    ff_bmu_df = pd.DataFrame(
        {
            "timestamp": ff_timestamps,
            "node_i": bmus[event_indices, 0],
            "node_j": bmus[event_indices, 1],
            "node": [node_label(b[0], b[1]) for b in bmus[event_indices]],
        }
    )
    ff_bmu_df.to_csv(paths["bmu_csv_path"], index=False)
    print(f"\nSaved {len(ff_bmu_df)} FF-day BMU assignments to {paths['bmu_csv_path']}")

    # ── Save cache ────────────────────────────────────────────────────────────
    fig_dir = paths["fig_dir"]
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(f"{fig_dir}/indiv-nodes", exist_ok=True)

    cache_dir = os.path.join(fig_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(
        os.path.join(cache_dir, "som_results.npz"),
        weights=weights,
        z500_nodes=z500_nodes,
        moist_nodes=moist_nodes,
        bmus=bmus,
        u_matrix=u_matrix,
        hit_map=hit_map,
        coords=coords,
        event_indices=event_indices,
        counts=counts,
        totals=totals,
        risk=risk,
        is_ff=is_ff,
        lat=lat,
        lon=lon,
    )
    print(f"Saved SOM arrays to {cache_dir}/som_results.npz")

    # ── Summary ───────────────────────────────────────────────────────────────
    overall_risk = is_ff.sum() / len(som_days)
    print("\n" + "=" * 60)
    print("ALL-DAYS SOM TRAINING SUMMARY")
    print("=" * 60)
    print(f"Moisture variable : {args.moisture_var}")
    print(f"Moisture weight   : {args.moisture_weight}")
    print(f"Snapshot hour     : {args.snapshot_hour}")
    print(f"Grid size         : {xdim}x{ydim}")
    print(f"Training samples  : {X.shape[0]}")
    print(f"Feature dimension : {X.shape[1]}")
    print(f"Initialization    : {args.init}")
    print(f"Topology          : {args.topology}")
    print(f"Random seed       : {args.seed}")
    print(f"Rotated           : {args.rotate}")
    print(f"Flash-flood days  : {is_ff.sum()} / {len(som_days)} "
          f"({overall_risk:.1%} overall risk)")
    print("\nNode counts and flash-flood risk:")
    for i in range(xdim):
        for j in range(ydim):
            n = totals[i, j]
            c = counts[i, j]
            r = risk[i, j]
            print(f"  {node_label(i, j)}: {c}/{n}  ({r:.1%})")


if __name__ == "__main__":
    main()
