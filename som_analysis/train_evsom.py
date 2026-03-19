"""
Evolution SOM Training for NYC Flash Flood Analysis
By: Ty Janoski

Trains a 2x2 Self-Organizing Map on stacked N-hour windows of Z500 +
moisture variable fields.  Each event's feature vector consists of
n_hours consecutive hourly snapshots concatenated into one long vector,
so the SOM learns synoptic evolution patterns rather than single-timestep
states.

Usage:
    python -m som_analysis.train_evsom --moisture-var thetae
    python -m som_analysis.train_evsom --moisture-var thetae --n-hours 12
    python -m som_analysis.train_evsom --moisture-var thetae --moisture-weight 1 --seed 42
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

from .config import MOISTURE_CONFIGS, SOM_INTERMEDIATE_PATH, get_evsom_paths
from .helpers import load_moist_var, node_label


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 2x2 evolution SOM on stacked N-hour Z500 + moisture fields."
    )
    parser.add_argument(
        "--moisture-var",
        required=True,
        choices=list(MOISTURE_CONFIGS.keys()),
        help="Moisture variable to use (IVT, tcwv, thetae).",
    )
    parser.add_argument(
        "--n-hours",
        type=int,
        default=24,
        help="Number of consecutive hourly frames per event (default: 24).",
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
        "--init",
        choices=["random", "pca"],
        default="random",
        help="Weight initialization method: 'random' or 'pca' (default: random).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = MOISTURE_CONFIGS[args.moisture_var]
    paths = get_evsom_paths(args.moisture_var, args.moisture_weight, args.n_hours)

    xdim, ydim = 2, 2
    pfx = cfg["file_prefix"]
    var_name = cfg["var_name"]
    moist_time_dim = cfg["time_dim"]
    n_hours = args.n_hours

    # ── Load evolution data (N_events, n_hours, lat, lon) ─────────────────────
    print(f"Loading evolution data for {args.moisture_var} (n_hours={n_hours})...")

    # Moisture: use load_moist_var then grab the DataArray
    moist_nw_path = f"{SOM_INTERMEDIATE_PATH}{pfx}_norm_weighted_ffe_evsom.nc"
    moist_norm_path = f"{SOM_INTERMEDIATE_PATH}{pfx}_norm_ffe_evsom.nc"
    moist_raw_path = f"{SOM_INTERMEDIATE_PATH}{pfx}_ffe_evsom.nc"

    moist_nw = load_moist_var(moist_nw_path, var_name)
    moist_norm = load_moist_var(moist_norm_path, var_name)
    moist_raw = load_moist_var(moist_raw_path, var_name)

    z500_nw = xr.load_dataarray(
        f"{SOM_INTERMEDIATE_PATH}era5_Z500_norm_weighted_ffe_evsom.nc"
    )
    z500_norm = xr.load_dataarray(
        f"{SOM_INTERMEDIATE_PATH}era5_Z500_norm_ffe_evsom.nc"
    )

    # Verify shapes agree
    assert z500_nw.sizes["event_time"] == moist_nw.sizes["event_time"], (
        "Z500 and moisture event counts differ!"
    )
    N = z500_nw.sizes["event_time"]
    _z500_spatial = [d for d in z500_nw.dims if d not in ("event_time", "hour_offset")]
    _z500_lat_dim, _z500_lon_dim = _z500_spatial[0], _z500_spatial[1]
    n_lat_z = z500_nw.sizes[_z500_lat_dim]
    n_lon_z = z500_nw.sizes[_z500_lon_dim]
    n_lat_m = moist_nw.sizes[cfg["lat_dim"]]
    n_lon_m = moist_nw.sizes[cfg["lon_dim"]]

    # ── Flatten & concatenate ─────────────────────────────────────────────────
    # Feature layout: [Z500 h=T-(n-1), ..., Z500 h=T, moist h=T-(n-1), ..., moist h=T]
    z500_flat = z500_nw.values.reshape(N, -1)        # (N, n_hours * lat_z * lon_z)
    moist_flat = moist_nw.values.reshape(N, -1)      # (N, n_hours * lat_m * lon_m)

    X = np.concatenate([z500_flat, moist_flat * args.moisture_weight], axis=1)
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

    n_spatial_z = n_lat_z * n_lon_z
    n_spatial_m = n_lat_m * n_lon_m

    # Reshape weights back to (xdim, ydim, n_hours, lat, lon)
    z500_nodes = weights[:, : n_hours * n_spatial_z].reshape(
        xdim, ydim, n_hours, n_lat_z, n_lon_z
    )
    moist_nodes = weights[:, n_hours * n_spatial_z :].reshape(
        xdim, ydim, n_hours, n_lat_m, n_lon_m
    )
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
                z_flat = z500_nodes[i, j].reshape(n_hours * n_spatial_z)
                m_flat = (moist_nodes[i, j] * args.moisture_weight).reshape(
                    n_hours * n_spatial_m
                )
                weights_rotated[idx, : n_hours * n_spatial_z] = z_flat
                weights_rotated[idx, n_hours * n_spatial_z :] = m_flat
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
            "timestamp": pd.to_datetime(z500_nw["event_time"].values),
            "node_i": bmus[:, 0],
            "node_j": bmus[:, 1],
            "node": [node_label(b[0], b[1]) for b in bmus],
        }
    )
    bmu_df.to_csv(paths["bmu_csv_path"], index=False)
    print(f"\nSaved {len(bmu_df)} BMU assignments to {paths['bmu_csv_path']}")

    # ── Save cache for plot_evsom.py ──────────────────────────────────────────
    fig_dir = paths["fig_dir"]
    os.makedirs(f"{fig_dir}/node-animations", exist_ok=True)
    os.makedirs(f"{fig_dir}/key-hours", exist_ok=True)

    cache_dir = os.path.join(fig_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    lat_z = z500_nw[_z500_lat_dim].values
    lon_z = z500_nw[_z500_lon_dim].values
    lat_m = moist_nw[cfg["lat_dim"]].values
    lon_m = moist_nw[cfg["lon_dim"]].values

    np.savez(
        os.path.join(cache_dir, "som_results.npz"),
        weights=weights,
        z500_nodes=z500_nodes,
        moist_nodes=moist_nodes,
        bmus=bmus,
        u_matrix=u_matrix,
        hit_map=hit_map,
        coords=coords,
        n_hours=np.array(n_hours),
        lat_z=lat_z,
        lon_z=lon_z,
        lat_m=lat_m,
        lon_m=lon_m,
    )
    print(f"Saved SOM arrays to {cache_dir}/som_results.npz")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("EVOLUTION SOM TRAINING SUMMARY")
    print("=" * 50)
    print(f"Moisture variable : {args.moisture_var}")
    print(f"N hours           : {n_hours}")
    print(f"Moisture weight   : {args.moisture_weight}")
    print(f"Grid size         : {xdim}x{ydim}")
    print(f"Training samples  : {X.shape[0]}")
    print(f"Feature dimension : {X.shape[1]}")
    print(f"Initialization    : {args.init}")
    print(f"Random seed       : {args.seed}")
    print(f"Rotated           : {args.rotate}")
    print("\nNode counts:")
    for i in range(xdim):
        for j in range(ydim):
            n = int(hit_map.T[i, j])
            print(f"  {node_label(i, j)}: {n}")


if __name__ == "__main__":
    main()
