"""
Movie S1 — evSOM Combined Animation (T−23h → T=0)

Default (4-row × 2-column) layout, one frame per hour offset.
Left column  : standardized Z500 anomaly (contours) + standardized 850-hPa θe anomaly (shaded)
Right column : mean Z500 in dam (contours) + mean 850-hPa θe in K (shaded)
Row order    : A1, A2, B1, B2

Wide layout (--wide flag): 2-row × 4-column per frame.
Left 2×2  : standardized anomaly (A1/A2 top, B1/B2 bottom)
Right 2×2 : raw composite (A1/A2 top, B1/B2 bottom)

Split layout (--split flag): two separate 2×2 movies.
  movieS01_evsom_anomalies_wide  : standardized anomaly only (2×2)
  movieS01_evsom_composites_wide : raw composite only (2×2)

Output:
    movies/movieS01/movieS01_evsom_combined.gif / .mp4
    movies/movieS01/movieS01_evsom_combined_wide.gif / .mp4      (--wide)
    movies/movieS01/movieS01_evsom_anomalies_wide.gif / .mp4     (--split)
    movies/movieS01/movieS01_evsom_composites_wide.gif / .mp4    (--split)

Usage:
    python -m figure_scripts.movieS01_evsom_combined_animation
    python -m figure_scripts.movieS01_evsom_combined_animation --wide
    python -m figure_scripts.movieS01_evsom_combined_animation --split
"""

import argparse
import io
import os
import subprocess
import tempfile

import cartopy.crs as ccrs
import cmweather  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from PIL import Image

from som_analysis.config import setup_plotting
from som_analysis.helpers import add_map_features, node_label

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "movies", "movieS01")
os.makedirs(OUT_DIR, exist_ok=True)

CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "figs",
    "Z500-and-thetae-evSOM-24h",
    ".cache",
    "som_results.npz",
)
Z500_PATH = "/mnt/drive2/SOM_intermediate_files/era5_Z500_ffe_evsom.nc"
THETAE_PATH = "/mnt/drive2/SOM_intermediate_files/era5_thetae_ffe_evsom.nc"

# ── Figure / animation parameters ─────────────────────────────────────────────
XDIM, YDIM = 2, 2
FIG_WIDTH = 5.5
FIG_HEIGHT = 6.5
FIG_WIDTH_WIDE = 11.0
FIG_HEIGHT_WIDE = 3.5
FIG_WIDTH_SPLIT = 5.5  # half of wide: one 2×2 panel
FIG_HEIGHT_SPLIT = 3.5
DPI_GIF = 300
DPI_MP4 = 300
FRAME_MS = 250  # milliseconds per frame

# Left column: standardized anomaly levels
LEVELS_MOIST = np.arange(-1.2, 1.21, 0.2)
LEVELS_Z_ANOM = np.arange(-1.4, 1.41, 0.2)

# Right column: raw composite levels
LEVELS_THETAE = np.arange(295, 341, 5)  # K
LEVELS_Z_RAW = range(549, 598, 2)  # dam

# Node order: A1, A2, B1, B2
NODE_ORDER = [(i, j) for j in range(YDIM) for i in range(XDIM)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--wide",
        action="store_true",
        help="Use wide 2×4 layout for presentations",
    )
    p.add_argument(
        "--split",
        action="store_true",
        help="Produce two separate 2×2 movies: anomalies and composites",
    )
    return p.parse_args()


def _precompute_composites(z500_da, thetae_da, bmus):
    """Pre-compute per-node composite means for every hour offset."""
    composites = {}
    for i in range(XDIM):
        for j in range(YDIM):
            idx = np.where((bmus[:, 0] == i) & (bmus[:, 1] == j))[0]
            n = len(idx)
            z500_node = z500_da.isel(event_time=idx).mean("event_time").values / 98.1
            thetae_node = thetae_da.isel(event_time=idx).mean("event_time").values
            composites[(i, j)] = {"z500": z500_node, "thetae": thetae_node, "n": n}
    return composites


def _make_frame(
    z500_nodes,
    moist_nodes,
    lat_w,
    lon_w,
    composites,
    lat_c,
    lon_c,
    bmus,
    hr_idx,
    hr_offset,
    dpi,
    wide=False,
    panel="both",
):
    """Render one frame and return a PIL Image.

    panel : 'both'      — combined wide layout (2×4 or 4×2)
            'anomaly'   — anomaly-only 2×2 layout
            'composite' — composite-only 2×2 layout
    """
    proj = ccrs.PlateCarree()
    if panel in ("anomaly", "composite"):
        nrows, ncols = 2, 2
        fig_w, fig_h = FIG_WIDTH_SPLIT, FIG_HEIGHT_SPLIT
    elif wide:
        nrows, ncols = 2, 4
        fig_w, fig_h = FIG_WIDTH_WIDE, FIG_HEIGHT_WIDE
    else:
        nrows, ncols = XDIM * YDIM, 2
        fig_w, fig_h = FIG_WIDTH, FIG_HEIGHT

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        subplot_kw={"projection": proj},
        constrained_layout=True,
        dpi=dpi,
    )
    fig.get_layout_engine().set(hspace=0.0, wspace=0.04, h_pad=0.01, w_pad=0.02)

    im_left = None
    im_right = None

    for i, j in NODE_ORDER:
        lbl = node_label(i, j)
        n = int(np.sum((bmus[:, 0] == i) & (bmus[:, 1] == j)))

        if panel in ("anomaly", "composite"):
            ax = axes[j, i]
        elif wide:
            ax_l = axes[j, i]
            ax_r = axes[j, i + 2]
        else:
            row = j * XDIM + i
            ax_l = axes[row, 0]
            ax_r = axes[row, 1]

        if panel == "anomaly":
            # ── Anomaly-only panel ────────────────────────────────────────────
            im_left = ax.contourf(
                lon_w,
                lat_w,
                moist_nodes[i, j, hr_idx],
                cmap="balance",
                levels=LEVELS_MOIST,
                transform=proj,
                extend="both",
            )
            cn = ax.contour(
                lon_w,
                lat_w,
                z500_nodes[i, j, hr_idx],
                colors="black",
                linewidths=0.4,
                levels=LEVELS_Z_ANOM,
                transform=proj,
            )
            ax.clabel(cn, inline=True, fontsize=3.0, fmt="%.1f")
            add_map_features(ax)
            ax.text(
                0.0,
                1.01,
                f"{lbl}  ($n$={n})",
                transform=ax.transAxes,
                fontsize=5.5,
                fontweight="bold",
                ha="left",
                va="bottom",
            )

        elif panel == "composite":
            # ── Composite-only panel ──────────────────────────────────────────
            im_right = ax.contourf(
                lon_c,
                lat_c,
                composites[(i, j)]["thetae"][hr_idx],
                cmap="BuPu",
                levels=LEVELS_THETAE,
                transform=proj,
                extend="both",
            )
            cn = ax.contour(
                lon_c,
                lat_c,
                composites[(i, j)]["z500"][hr_idx],
                colors="black",
                linewidths=0.4,
                levels=LEVELS_Z_RAW,
                transform=proj,
            )
            ax.clabel(cn, inline=True, fontsize=3.0, fmt="%d")
            add_map_features(ax)
            ax.text(
                0.0,
                1.01,
                f"{lbl}  ($n$={n})",
                transform=ax.transAxes,
                fontsize=5.5,
                fontweight="bold",
                ha="left",
                va="bottom",
            )

        else:
            # ── Left: standardized anomaly ────────────────────────────────────
            im_left = ax_l.contourf(
                lon_w,
                lat_w,
                moist_nodes[i, j, hr_idx],
                cmap="balance",
                levels=LEVELS_MOIST,
                transform=proj,
                extend="both",
            )
            cn_l = ax_l.contour(
                lon_w,
                lat_w,
                z500_nodes[i, j, hr_idx],
                colors="black",
                linewidths=0.4,
                levels=LEVELS_Z_ANOM,
                transform=proj,
            )
            ax_l.clabel(cn_l, inline=True, fontsize=3.0, fmt="%.1f")
            add_map_features(ax_l)
            ax_l.text(
                0.0,
                1.01,
                f"{lbl}  ($n$={n})",
                transform=ax_l.transAxes,
                fontsize=5.5,
                fontweight="bold",
                ha="left",
                va="bottom",
            )

            # ── Right: raw composite ──────────────────────────────────────────
            im_right = ax_r.contourf(
                lon_c,
                lat_c,
                composites[(i, j)]["thetae"][hr_idx],
                cmap="BuPu",
                levels=LEVELS_THETAE,
                transform=proj,
                extend="both",
            )
            cn_r = ax_r.contour(
                lon_c,
                lat_c,
                composites[(i, j)]["z500"][hr_idx],
                colors="black",
                linewidths=0.4,
                levels=LEVELS_Z_RAW,
                transform=proj,
            )
            ax_r.clabel(cn_r, inline=True, fontsize=3.0, fmt="%d")
            add_map_features(ax_r)
            ax_r.text(
                0.0,
                1.01,
                f"{lbl}  ($n$={n})",
                transform=ax_r.transAxes,
                fontsize=5.5,
                fontweight="bold",
                ha="left",
                va="bottom",
            )

    # ── Colorbars ─────────────────────────────────────────────────────────────
    if panel == "anomaly":
        cbar_l = fig.colorbar(
            im_left, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02, aspect=30
        )
        cbar_l.set_label(r"Std. 850-hPa $\theta_e$ Anomaly", fontsize=5.5)
        cbar_l.ax.tick_params(labelsize=4.5)
    elif panel == "composite":
        cbar_r = fig.colorbar(
            im_right, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02, aspect=30
        )
        cbar_r.set_label(r"850-hPa $\theta_e$ (K)", fontsize=5.5)
        cbar_r.ax.tick_params(labelsize=4.5)
    else:
        ax_left_group = (
            axes[:, :2].ravel().tolist() if wide else axes[:, 0].ravel().tolist()
        )
        ax_right_group = (
            axes[:, 2:].ravel().tolist() if wide else axes[:, 1].ravel().tolist()
        )
        cbar_l = fig.colorbar(
            im_left, ax=ax_left_group, shrink=0.6, pad=0.02, aspect=30
        )
        cbar_l.set_label(r"Std. 850-hPa $\theta_e$ Anomaly", fontsize=5.5)
        cbar_l.ax.tick_params(labelsize=4.5)
        cbar_r = fig.colorbar(
            im_right, ax=ax_right_group, shrink=0.6, pad=0.02, aspect=30
        )
        cbar_r.set_label(r"850-hPa $\theta_e$ (K)", fontsize=5.5)
        cbar_r.ax.tick_params(labelsize=4.5)

    # ── Time label ────────────────────────────────────────────────────────────
    sign = "−" if hr_offset < 0 else "="
    hrs = abs(hr_offset)
    tlabel = f"T {sign} {hrs}h" if hrs > 0 else "T = 0"
    fig.suptitle(tlabel, fontsize=8)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    pil_img = Image.open(buf).copy()
    plt.close(fig)
    buf.close()
    return pil_img


def main():
    args = parse_args()
    setup_plotting()

    # ── Load node-weight data (left column) ───────────────────────────────────
    print(f"Loading cache from {CACHE_PATH} ...")
    cached = np.load(CACHE_PATH)
    z500_nodes = cached["z500_nodes"]  # (2, 2, n_hours, nlat, nlon)
    moist_nodes = cached["moist_nodes"]
    bmus = cached["bmus"]
    lat_w = cached["lat_z"]
    lon_w = cached["lon_z"]
    n_hours = z500_nodes.shape[2]
    hour_offsets = list(range(-(n_hours - 1), 1))

    # ── Load raw ERA5 data (right column) ─────────────────────────────────────
    print("Loading raw evSOM fields ...")
    z500_var = list(xr.open_dataset(Z500_PATH).data_vars)[0]
    thetae_var = list(xr.open_dataset(THETAE_PATH).data_vars)[0]
    z500_da = xr.open_dataset(Z500_PATH)[z500_var]
    thetae_da = xr.open_dataset(THETAE_PATH)[thetae_var]
    lat_c = z500_da["latitude"].values
    lon_c = z500_da["longitude"].values

    print("Pre-computing composites ...")
    composites = _precompute_composites(z500_da, thetae_da, bmus)

    # ── Determine which panels to render ──────────────────────────────────────
    if args.split:
        render_jobs = [
            ("anomaly", "movieS01_evsom_anomalies_wide"),
            ("composite", "movieS01_evsom_composites_wide"),
        ]
    else:
        suffix = "_wide" if args.wide else ""
        render_jobs = [("both", f"movieS01_evsom_combined{suffix}")]

    for panel, stem in render_jobs:
        print(f"\nRendering '{panel}' — {n_hours} frames @ {DPI_GIF} dpi ...")
        frames = []
        for hr_idx, hr_offset in enumerate(hour_offsets):
            if (hr_idx + 1) % 6 == 0 or hr_idx == 0:
                print(f"  Frame {hr_idx + 1}/{n_hours}  (T{hr_offset:+d}h)")
            frames.append(
                _make_frame(
                    z500_nodes,
                    moist_nodes,
                    lat_w,
                    lon_w,
                    composites,
                    lat_c,
                    lon_c,
                    bmus,
                    hr_idx,
                    hr_offset,
                    DPI_GIF,
                    wide=args.wide,
                    panel=panel,
                )
            )

        # ── Save GIF ──────────────────────────────────────────────────────────
        gif_path = os.path.join(OUT_DIR, f"{stem}.gif")
        print(f"Saving GIF → {gif_path}")
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=FRAME_MS,
            optimize=False,
        )

        # ── Save MP4 ──────────────────────────────────────────────────────────
        mp4_path = os.path.join(OUT_DIR, f"{stem}.mp4")
        print(f"Saving MP4 → {mp4_path}")
        with tempfile.TemporaryDirectory() as tmp:
            for k, img in enumerate(frames):
                img.save(os.path.join(tmp, f"frame_{k:03d}.png"))
            fps = 1000 / FRAME_MS
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-r",
                    str(fps),
                    "-i",
                    os.path.join(tmp, "frame_%03d.png"),
                    "-c:v",
                    "libx264",
                    "-crf",
                    "18",
                    "-vf",
                    "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-pix_fmt",
                    "yuv420p",
                    mp4_path,
                ],
                check=True,
                capture_output=True,
            )

    print("Done.")


if __name__ == "__main__":
    main()
