"""
SOM Training for NYC Flash Flood Analysis
By: Ty Janoski

Trains a 2x2 Self-Organizing Map on Z500 + moisture variable fields,
extracts weights and BMU assignments, and saves results.

Usage:
    python -m som_analysis.train_som --moisture-var thetae
    python -m som_analysis.train_som --moisture-var IVT --moisture-weight 2
    python -m som_analysis.train_som --moisture-var thetae --rotate --seed 42
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

from .config import MOISTURE_CONFIGS, SOM_INTERMEDIATE_PATH, get_paths
from .helpers import load_moist_var, node_label


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 2x2 SOM on Z500 + moisture variable fields."
    )
    parser.add_argument(
        "--moisture-var",
        required=True,
        choices=list(MOISTURE_CONFIGS.keys()),
        help="Moisture variable to use (IVT, tcwv, thetae).",
    )
    parser.add_argument(
        "--moisture-weight",
        type=float,
        default=1,
        help="Scalar multiplier for moisture features (default: 1).",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--n1",
        type=int,
        default=1000,
        help="Number of iterations for coarse training phase (default: 1000).",
    )
    parser.add_argument(
        "--n2",
        type=int,
        default=1000,
        help="Number of iterations for fine training phase (default: 1000).",
    )
    parser.add_argument(
        "--daily",
        action="store_true",
        help="Use daily-mean FFE fields instead of hourly snapshots.",
    )
    parser.add_argument(
        "--topology",
        choices=["rectangular", "hexagonal"],
        default="rectangular",
        help="SOM grid topology (default: rectangular).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = MOISTURE_CONFIGS[args.moisture_var]
    paths = get_paths(args.moisture_var, args.moisture_weight, daily=args.daily)

    xdim, ydim = 2, 2
    pfx = cfg["file_prefix"]
    var_name = cfg["var_name"]
    moist_time_dim = cfg["time_dim"]
    ffe_suffix = "_ffe_daily" if args.daily else "_ffe"

    # ── Load data ─────────────────────────────────────────────────────────────
    print(
        f"Loading data for {args.moisture_var} "
        f"(weight={args.moisture_weight}, daily={args.daily})..."
    )

    moist_norm_weighted_ffe = load_moist_var(
        f"{SOM_INTERMEDIATE_PATH}{pfx}_norm_weighted{ffe_suffix}.nc", var_name
    )
    moist_norm_ffe = load_moist_var(
        f"{SOM_INTERMEDIATE_PATH}{pfx}_norm{ffe_suffix}.nc", var_name
    )
    moist_ffe = load_moist_var(f"{SOM_INTERMEDIATE_PATH}{pfx}{ffe_suffix}.nc", var_name)

    z500_norm_weighted_ffe = xr.load_dataarray(
        f"{SOM_INTERMEDIATE_PATH}era5_Z500_norm_weighted{ffe_suffix}.nc"
    )

    # ── Flatten & concatenate ─────────────────────────────────────────────────
    z500_flat = z500_norm_weighted_ffe.stack(features=["latitude", "longitude"]).values
    moist_flat = moist_norm_weighted_ffe.stack(
        features=[cfg["lat_dim"], cfg["lon_dim"]]
    ).values

    X = np.concatenate((z500_flat, moist_flat * args.moisture_weight), axis=1)
    print(f"Training matrix shape: {X.shape}")

    # ── SOM training ──────────────────────────────────────────────────────────
    sig1 = np.sqrt(xdim**2 + ydim**2)
    sig2 = 1.0
    lr1, lr2 = 0.5, 0.1

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

    # Spatial dimensions
    lat = moist_norm_ffe[cfg["lat_dim"]]
    lon = moist_norm_ffe[cfg["lon_dim"]]
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

        # Rebuild weights from rotated node arrays
        weights_rotated = np.zeros_like(weights)
        for i in range(xdim):
            for j in range(ydim):
                idx = i * ydim + j
                weights_rotated[idx, :n_features] = z500_nodes[i, j].flatten()
                weights_rotated[idx, n_features:] = moist_nodes[i, j].flatten()
        weights = weights_rotated

        print("\nSOM rotated 90 degrees clockwise.")

    # ── U-matrix and hit map ──────────────────────────────────────────────────
    u_matrix = som.distance_map().T
    hit_map = np.zeros((xdim, ydim))
    for i, j in bmus:
        hit_map[i, j] += 1
    hit_map = hit_map.T

    # ── Sammon / MDS coordinates ──────────────────────────────────────────────
    D = pairwise_distances(weights)
    coords = MDS(
        n_components=2,
        metric="precomputed",
        random_state=42,
        n_init=4,
        init="random",  # pyright: ignore[reportCallIssue]
    ).fit_transform(D)

    # ── Save BMU CSV ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(paths["bmu_csv_path"]), exist_ok=True)
    bmu_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(moist_ffe[moist_time_dim].values),
            "node_i": bmus[:, 0],
            "node_j": bmus[:, 1],
            "node": [node_label(b[0], b[1]) for b in bmus],
        }
    )
    bmu_df.to_csv(paths["bmu_csv_path"], index=False)
    print(f"\nSaved {len(bmu_df)} BMU assignments to {paths['bmu_csv_path']}")

    # ── Save intermediate arrays for plot_som.py ──────────────────────────────
    fig_dir = paths["fig_dir"]
    os.makedirs(f"{fig_dir}/indiv-nodes", exist_ok=True)

    cache_dir = os.path.join(fig_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_name = "som_results_daily.npz" if args.daily else "som_results.npz"
    np.savez(
        os.path.join(cache_dir, cache_name),
        weights=weights,
        z500_nodes=z500_nodes,
        moist_nodes=moist_nodes,
        bmus=bmus,
        u_matrix=u_matrix,
        hit_map=hit_map,
        coords=coords,
    )
    print(f"Saved SOM arrays to {cache_dir}/som_results.npz")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("SOM TRAINING SUMMARY")
    print("=" * 50)
    print(f"Moisture variable : {args.moisture_var}")
    print(f"Moisture weight   : {args.moisture_weight}")
    print(f"Daily fields      : {args.daily}")
    print(f"Grid size         : {xdim}x{ydim}")
    print(f"Training samples  : {X.shape[0]}")
    print(f"Feature dimension : {X.shape[1]}")
    print(f"Initialization    : {args.init}")
    print(f"Topology          : {args.topology}")
    print(f"Random seed       : {args.seed}")
    print(f"Rotated           : {args.rotate}")
    print("\nNode counts:")
    for i in range(xdim):
        for j in range(ydim):
            n = int(hit_map.T[i, j])
            print(f"  {node_label(i, j)}: {n}")


if __name__ == "__main__":
    main()
