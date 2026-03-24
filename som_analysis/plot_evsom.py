"""
Evolution SOM Diagnostic Plots for NYC Flash Flood Analysis
By: Ty Janoski

Generates animated GIFs (one 4-panel figure per animation type) showing
the synoptic evolution over N hours, plus a static key-hours panel and
standard diagnostics (U-matrix, hit map, monthly histogram).

Usage:
    python -m som_analysis.plot_evsom --moisture-var thetae
    python -m som_analysis.plot_evsom --moisture-var thetae --n-hours 24 --fps 4
    python -m som_analysis.plot_evsom --moisture-var thetae --skip-anim
"""

import argparse
import os

import cartopy.crs as ccrs
import cmweather  # noqa: F401
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from .config import (
    MOISTURE_CONFIGS,
    SOM_INTERMEDIATE_PATH,
    get_evsom_paths,
    setup_plotting,
)
from .helpers import add_map_features, get_node_indices, load_moist_var, node_label


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate diagnostic plots for a trained evolution SOM."
    )
    parser.add_argument(
        "--moisture-var",
        required=True,
        choices=list(MOISTURE_CONFIGS.keys()),
        help="Moisture variable (IVT, tcwv, thetae).",
    )
    parser.add_argument(
        "--n-hours",
        type=int,
        default=24,
        help="Number of hourly frames used during training (default: 24).",
    )
    parser.add_argument(
        "--moisture-weight",
        type=float,
        default=1,
        help="Moisture weight used during training (default: 1).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="Frames per second for animated GIFs (default: 4).",
    )
    parser.add_argument(
        "--key-hours",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Hour offsets (relative to event time T=0) to show in the key-hours panel. "
            "Default: evenly spaced selection from the window."
        ),
    )
    parser.add_argument(
        "--skip-anim",
        action="store_true",
        help="Skip animated GIF generation.",
    )
    return parser.parse_args()


def _default_key_hours(n_hours, n_panels=4, step=7):
    """Return n_panels hour offsets ending at 0, spaced `step` hours apart.

    For the default step=7 and n_panels=4 this gives [-21, -14, -7, 0],
    i.e. T-21h, T-14h, T-7h, and T+0h (the event time).
    Values are clamped to the available offset range.
    """
    available = set(int(h) for h in (np.arange(n_hours) - (n_hours - 1)))
    targets = [-(n_panels - 1 - k) * step for k in range(n_panels)]
    return [int(min(available, key=lambda x, t=t: abs(x - t))) for t in targets]


def main():
    args = parse_args()
    setup_plotting()

    cfg = MOISTURE_CONFIGS[args.moisture_var]
    paths = get_evsom_paths(args.moisture_var, args.moisture_weight, args.n_hours)
    fig_dir = paths["fig_dir"]
    _lbl = paths["file_label"]
    moist_label_short = cfg["label_short"]
    pfx = cfg["file_prefix"]
    var_name = cfg["var_name"]

    xdim, ydim = 2, 2
    n_hours = args.n_hours
    hour_offsets = np.arange(n_hours) - (n_hours - 1)  # T-(n-1) ... T+0

    os.makedirs(f"{fig_dir}/node-animations", exist_ok=True)
    os.makedirs(f"{fig_dir}/key-hours", exist_ok=True)

    # ── Load cached SOM results ───────────────────────────────────────────────
    cache_path = os.path.join(fig_dir, ".cache", "som_results.npz")
    print(f"Loading SOM results from {cache_path} ...")
    cached = np.load(cache_path)
    z500_nodes = cached["z500_nodes"]  # (xdim, ydim, n_hours, lat_z, lon_z)
    moist_nodes = cached["moist_nodes"]  # (xdim, ydim, n_hours, lat_m, lon_m)
    bmus = cached["bmus"]
    u_matrix = cached["u_matrix"]
    hit_map = cached["hit_map"]
    lat_z = cached["lat_z"]
    lon_z = cached["lon_z"]

    # ── Load event data for composites ────────────────────────────────────────
    print("Loading event data for composites ...")
    z500_raw = xr.load_dataarray(f"{SOM_INTERMEDIATE_PATH}era5_Z500_ffe_evsom.nc")
    moist_raw = load_moist_var(f"{SOM_INTERMEDIATE_PATH}{pfx}_ffe_evsom.nc", var_name)

    # Composite over events for each node → (xdim, ydim, n_hours, lat, lon)
    def node_composites(da, time_dim="event_time"):
        n_h = da.sizes["hour_offset"]
        n_la = da.sizes[da.dims[2]]
        n_lo = da.sizes[da.dims[3]]
        out = np.full((xdim, ydim, n_h, n_la, n_lo), np.nan)
        for ii in range(xdim):
            for jj in range(ydim):
                idx = get_node_indices(bmus, ii, jj)
                if len(idx) > 0:
                    out[ii, jj] = da.isel({time_dim: idx}).mean(time_dim).values
        return out

    z500_raw_comp = node_composites(z500_raw)
    moist_raw_comp = node_composites(moist_raw, time_dim="event_time")

    # ── U-matrix and hit map ──────────────────────────────────────────────────
    print("Plotting U-matrix and hit map ...")
    fig, axes = plt.subplots(1, 2, layout="constrained", figsize=(6, 3), dpi=600)

    im0 = axes[0].imshow(u_matrix, cmap="viridis", origin="lower")
    axes[0].set_title("U-Matrix (Mean Inter-Node Distance)", fontsize=7)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, shrink=0.7)

    im1 = axes[1].imshow(hit_map, cmap="plasma", origin="lower")
    axes[1].set_title("Hit Map (Samples per Node)", fontsize=7)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, shrink=0.7)

    for ax in axes:
        ax.set_xticks(np.arange(xdim))
        ax.set_yticks(np.arange(ydim))
        ax.set_xlabel("X-index", fontsize=6)
        ax.set_ylabel("Y-index", fontsize=6)

    plt.savefig(f"{fig_dir}/Z500_{_lbl}_evsom_u_matrix_hit_map.png")
    plt.close()

    # ── Helper: 4-panel animated GIF (one panel per node) ────────────────────
    def _make_4panel_gif(
        z_nodes,
        m_nodes,
        levels_z,
        levels_m,
        m_cmap,
        m_cbar_label,
        gif_path,
        title_prefix="",
    ):
        """Save a single animated GIF with all nodes shown in a 2×2 grid."""
        fig, axes = plt.subplots(
            ydim,
            xdim,
            figsize=(5 * xdim, 3.2 * ydim),
            subplot_kw={"projection": ccrs.PlateCarree()},
            constrained_layout=True,
            dpi=150,
        )

        def _t_str(h_off):
            return "T+0h" if h_off == 0 else f"T{h_off:+d}h"

        def _draw(frame_idx):
            h_off = int(hour_offsets[frame_idx])
            last_cf = None
            for jj in range(ydim):
                for ii in range(xdim):
                    ax = axes[jj, ii]
                    ax.cla()
                    add_map_features(ax)
                    last_cf = ax.contourf(
                        lon_z,
                        lat_z,
                        m_nodes[ii, jj, frame_idx],
                        levels=levels_m,
                        cmap=m_cmap,
                        transform=ccrs.PlateCarree(),
                        extend="both",
                    )
                    ax.contour(
                        lon_z,
                        lat_z,
                        z_nodes[ii, jj, frame_idx],
                        levels=levels_z,
                        colors="black",
                        linewidths=0.5,
                        transform=ccrs.PlateCarree(),
                    )
                    n_ev = int(hit_map.T[ii, jj])
                    lbl = f"{node_label(ii, jj)}  N={n_ev}  —  {_t_str(h_off)}"
                    if title_prefix:
                        lbl = f"{title_prefix} {lbl}"
                    ax.set_title(lbl, fontsize=7)
            return last_cf

        # First frame + colorbar
        cf0 = _draw(0)
        cbar = fig.colorbar(
            cf0,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            pad=0.02,
            fraction=0.046,
            shrink=0.9,
        )
        cbar.set_label(m_cbar_label, fontsize=6)
        cbar.ax.tick_params(labelsize=5)

        def update(frame_idx):
            _draw(frame_idx)
            return []

        anim = animation.FuncAnimation(
            fig, update, frames=n_hours, interval=1000 // args.fps, blit=False
        )
        anim.save(gif_path, writer=animation.PillowWriter(fps=args.fps))
        plt.close(fig)
        print(f"  Saved {gif_path}")

    # ── Animated GIFs (4-panel, one per node in 2×2 grid) ────────────────────
    if not args.skip_anim:
        print("Generating animated GIF (weight space, 4-panel) ...")
        _make_4panel_gif(
            z500_nodes,
            moist_nodes,
            levels_z=np.arange(-1.4, 1.41, 0.2),
            levels_m=cfg["levels_weights"],
            m_cmap="balance",
            m_cbar_label=f"Standardized {moist_label_short} Anomaly",
            gif_path=f"{fig_dir}/node-animations/all_nodes_evsom.gif",
        )
    else:
        print("Skipping animated GIFs (--skip-anim).")

    # ── Composite animated GIF (4-panel) ─────────────────────────────────────
    if not args.skip_anim:
        levels_z_raw = range(552, 595, 3)

        print("Generating composite raw GIF (4-panel) ...")
        _make_4panel_gif(
            z500_raw_comp / 98.1,
            moist_raw_comp,
            levels_z=levels_z_raw,
            levels_m=cfg["levels_raw"],
            m_cmap=cfg["cmap_raw"],
            m_cbar_label=f"{moist_label_short} ({cfg['units_raw']})",
            gif_path=f"{fig_dir}/node-animations/all_nodes_evsom_composite_raw.gif",
            title_prefix="Composite",
        )

    # ── Static key-hours panel ────────────────────────────────────────────────
    print("Plotting key-hours panel ...")
    if args.key_hours is not None:
        key_hours = args.key_hours
    else:
        key_hours = _default_key_hours(n_hours, n_panels=4)

    # Map hour offset → index into n_hours axis
    offset_to_idx = {int(h): k for k, h in enumerate(hour_offsets)}
    key_indices = []
    for h in key_hours:
        if h not in offset_to_idx:
            # Find closest available
            closest = min(offset_to_idx.keys(), key=lambda x: abs(x - h))
            print(f"  Warning: hour_offset={h} not in data; using {closest} instead.")
            h = closest
        key_indices.append(offset_to_idx[h])
    key_hours_actual = [int(hour_offsets[k]) for k in key_indices]

    n_cols = len(key_indices)
    n_rows = xdim * ydim  # one row per node

    levels_z = np.arange(-1.4, 1.41, 0.2)
    levels_m = cfg["levels_weights"]

    # Node order: row-first through the SOM (A1, A2, B1, B2)
    node_order = [(i, j) for j in range(ydim) for i in range(xdim)]

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.2 * n_cols, 1.6 * n_rows),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
        dpi=300,
    )
    fig.set_constrained_layout_pads(h_pad=1 / 72, hspace=0.01)
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, (i, j) in enumerate(node_order):
        for col, (h_idx, h_off) in enumerate(
            zip(key_indices, key_hours_actual, strict=False)
        ):
            ax = axes[row, col]
            z_fr = z500_nodes[i, j, h_idx]
            m_fr = moist_nodes[i, j, h_idx]

            im = ax.contourf(
                lon_z,
                lat_z,
                m_fr,
                levels=levels_m,
                cmap="balance",
                transform=ccrs.PlateCarree(),
                extend="both",
            )
            ax.contour(
                lon_z,
                lat_z,
                z_fr,
                levels=levels_z,
                colors="black",
                linewidths=0.4,
                transform=ccrs.PlateCarree(),
            )
            add_map_features(ax)

            if row == 0:
                t_str = f"T{h_off:+d}h" if h_off != 0 else "T+0h"
                ax.set_title(t_str, fontsize=6)
            if col == 0:
                n_ev = int(hit_map.T[i, j])
                ax.set_ylabel(f"{node_label(i, j)} N={n_ev}", fontsize=5, labelpad=2)

    # Shared colorbar
    cbar = fig.colorbar(
        im, ax=axes.ravel().tolist(), shrink=0.4, pad=0.01, orientation="vertical"
    )
    cbar.set_label(f"Standardized {moist_label_short} Anomaly", fontsize=6)
    cbar.ax.tick_params(labelsize=5)

    plt.suptitle(
        f"Evolution SOM Key Hours: Z500 (contoured) + {moist_label_short} (shaded)",
        fontsize=8,
    )
    out_key = f"{fig_dir}/key-hours/key_hours_{_lbl}.png"
    plt.savefig(out_key, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_key}")

    # ── Composite key-hours panels ────────────────────────────────────────────
    def _save_key_hours_panel(
        z_comp, m_comp, levels_z, levels_m, m_cmap, m_cbar_label, suptitle, out_path
    ):
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(2.2 * n_cols, 1.6 * n_rows),
            subplot_kw={"projection": ccrs.PlateCarree()},
            constrained_layout=True,
            dpi=300,
        )
        fig.set_constrained_layout_pads(h_pad=1 / 72, hspace=0.01)
        _axes = axes
        if n_rows == 1:
            _axes = _axes[np.newaxis, :]
        if n_cols == 1:
            _axes = _axes[:, np.newaxis]

        for row, (i, j) in enumerate(node_order):
            for col, (h_idx, h_off) in enumerate(
                zip(key_indices, key_hours_actual, strict=False)
            ):
                ax = _axes[row, col]
                im = ax.contourf(
                    lon_z,
                    lat_z,
                    m_comp[i, j, h_idx],
                    levels=levels_m,
                    cmap=m_cmap,
                    transform=ccrs.PlateCarree(),
                    extend="both",
                )
                ax.contour(
                    lon_z,
                    lat_z,
                    z_comp[i, j, h_idx],
                    levels=levels_z,
                    colors="black",
                    linewidths=0.4,
                    transform=ccrs.PlateCarree(),
                )
                add_map_features(ax)
                if row == 0:
                    t_str = f"T{h_off:+d}h" if h_off != 0 else "T+0h"
                    ax.set_title(t_str, fontsize=6)
                if col == 0:
                    n_ev = int(hit_map.T[i, j])
                    ax.set_ylabel(
                        f"{node_label(i, j)} N={n_ev}", fontsize=5, labelpad=2
                    )

        cbar = fig.colorbar(
            im, ax=_axes.ravel().tolist(), shrink=0.4, pad=0.01, orientation="vertical"
        )
        cbar.set_label(m_cbar_label, fontsize=6)
        cbar.ax.tick_params(labelsize=5)
        plt.suptitle(suptitle, fontsize=8)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out_path}")

    print("Plotting composite key-hours panel ...")
    levels_z_raw = range(552, 595, 3)

    _save_key_hours_panel(
        z500_raw_comp / 98.1,
        moist_raw_comp,
        levels_z=levels_z_raw,
        levels_m=cfg["levels_raw"],
        m_cmap=cfg["cmap_raw"],
        m_cbar_label=f"{moist_label_short} ({cfg['units_raw']})",
        suptitle=(
            f"Evolution SOM Composite: Z500 (contoured) + {moist_label_short} (shaded)"
        ),
        out_path=f"{fig_dir}/key-hours/key_hours_{_lbl}_composite_raw.png",
    )

    # ── Monthly histograms ────────────────────────────────────────────────────
    print("Plotting monthly histograms ...")
    bmu_df = pd.read_csv(paths["bmu_csv_path"])
    months = pd.to_datetime(bmu_df["timestamp"]).dt.month.to_numpy()

    month_counts = {}
    for i in range(xdim):
        for j in range(ydim):
            idx = get_node_indices(bmus, i, j)
            node_months = months[idx]
            month_counts[(i, j)] = np.bincount(node_months, minlength=13)[1:]

    month_labels = ["May", "Jun", "Jul", "Aug", "Sep", "Oct"]
    fig, axes = plt.subplots(
        ydim, xdim, figsize=(6, 3.7), constrained_layout=True, dpi=600
    )

    for i in range(xdim):
        for j in range(ydim):
            ax = axes[j, i]
            monthly = month_counts[(i, j)][4:10]
            ax.bar(month_labels, monthly, color="teal", alpha=0.9, width=0.8)
            ax.set_title(f"{node_label(i, j)}  N={monthly.sum()}", fontsize=6)
            ax.tick_params(axis="x", bottom=False, labelsize=5)
            ax.set_ylim(0, 18)
            ax.set_yticks(np.arange(0, 17, 2))
            ax.grid(True, linewidth=0.3, alpha=0.5, axis="y")

    plt.suptitle(
        "Warm-Season (May\u2013Oct) Event Distribution per Evolution SOM Node",
        fontsize=8,
        y=1.04,
    )
    plt.savefig(f"{fig_dir}/Z500_{_lbl}_evsom_monthly_counts.png", bbox_inches="tight")
    plt.close()

    print(f"\nAll plots saved to {fig_dir}/")


if __name__ == "__main__":
    main()
