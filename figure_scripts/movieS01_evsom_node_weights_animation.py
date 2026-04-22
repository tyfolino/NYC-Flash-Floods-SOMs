"""
Movie S1 — evSOM Node Weights Animation (T−23h → T=0)

2×2 panel layout (A1, B1 / A2, B2), one frame per hour offset.
Standardized Z500 anomaly (contoured) + standardized 850-hPa θe anomaly (shaded).
Matches the style of Fig. 2 (key-hours snapshot).

Output: movies/movieS01/movieS01_evsom_node_weights.gif

Usage:
    python -m figure_scripts.movieS01_evsom_node_weights_animation
"""

import io
import os
import subprocess
import tempfile

import cartopy.crs as ccrs
import cmweather  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
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

# ── Figure / animation parameters ─────────────────────────────────────────────
XDIM, YDIM = 2, 2
FIG_WIDTH = 5.5
FIG_HEIGHT = 3.2
DPI_GIF = 300
DPI_MP4 = 200
FRAME_MS = 250  # milliseconds per frame

LEVELS_MOIST = np.arange(-1.2, 1.21, 0.2)
LEVELS_Z = np.arange(-1.4, 1.41, 0.2)

# Node layout: rows = j, cols = i  →  A1 B1 / A2 B2
NODE_LAYOUT = [[(i, j) for i in range(XDIM)] for j in range(YDIM)]


def _make_frame(
    z500_nodes, moist_nodes, lat, lon, bmus, hr_idx, hr_offset, im_ref, dpi
):
    """Render one frame and return a PIL Image."""
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        YDIM,
        XDIM,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        subplot_kw={"projection": proj},
        constrained_layout=True,
        dpi=dpi,
    )
    fig.get_layout_engine().set(hspace=0.0, wspace=0.02, h_pad=0.02, w_pad=0.02)

    im = None
    for row, node_row in enumerate(NODE_LAYOUT):
        for col, (i, j) in enumerate(node_row):
            ax = axes[row, col]
            lbl = node_label(i, j)
            n = int(np.sum((bmus[:, 0] == i) & (bmus[:, 1] == j)))

            im = ax.contourf(
                lon,
                lat,
                moist_nodes[i, j, hr_idx],
                cmap="balance",
                levels=LEVELS_MOIST,
                transform=proj,
                extend="both",
            )
            cn = ax.contour(
                lon,
                lat,
                z500_nodes[i, j, hr_idx],
                colors="black",
                linewidths=0.4,
                levels=LEVELS_Z,
                transform=proj,
            )
            ax.clabel(cn, inline=True, fontsize=5.0, fmt="%.1f")
            add_map_features(ax)

            ax.text(
                0.0,
                1.01,
                f"{lbl}  ($n$={n})",
                transform=ax.transAxes,
                fontsize=7.5,
                fontweight="bold",
                ha="left",
                va="bottom",
            )

    # Time label centred above figure
    sign = "−" if hr_offset < 0 else "="
    hrs = abs(hr_offset)
    tlabel = f"T {sign} {hrs}h" if hrs > 0 else "T = 0"
    fig.suptitle(tlabel, fontsize=10)

    # Shared colorbar (use im_ref levels so colorbar is identical every frame)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02, aspect=28)
    cbar.set_label(r"Standardized 850-hPa $\theta_e$ Anomaly", fontsize=8)
    cbar.ax.tick_params(labelsize=6.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    pil_img = Image.open(buf).copy()
    plt.close(fig)
    buf.close()
    return pil_img


def main():
    setup_plotting()

    print(f"Loading cache from {CACHE_PATH} ...")
    cached = np.load(CACHE_PATH)
    z500_nodes = cached["z500_nodes"]  # (2, 2, 24, nlat, nlon)
    moist_nodes = cached["moist_nodes"]
    bmus = cached["bmus"]
    lat = cached["lat_z"]
    lon = cached["lon_z"]
    n_hours = z500_nodes.shape[2]

    # hour_offsets: −23, −22, … 0
    hour_offsets = list(range(-(n_hours - 1), 1))

    print(f"Rendering {n_hours} frames (GIF @ {DPI_GIF} dpi) ...")
    gif_frames = []
    for hr_idx, hr_offset in enumerate(hour_offsets):
        if (hr_idx + 1) % 6 == 0 or hr_idx == 0:
            print(f"  Frame {hr_idx + 1}/{n_hours}  (T{hr_offset:+d}h)")
        gif_frames.append(
            _make_frame(
                z500_nodes,
                moist_nodes,
                lat,
                lon,
                bmus,
                hr_idx,
                hr_offset,
                None,
                DPI_GIF,
            )
        )

    gif_path = os.path.join(OUT_DIR, "movieS01_evsom_node_weights.gif")
    print(f"Saving GIF → {gif_path}")
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        loop=0,
        duration=FRAME_MS,
        optimize=False,
    )

    print(f"Rendering {n_hours} frames (MP4 @ {DPI_MP4} dpi) ...")
    mp4_frames = [
        _make_frame(
            z500_nodes, moist_nodes, lat, lon, bmus, hr_idx, hr_offset, None, DPI_MP4
        )
        for hr_idx, hr_offset in enumerate(hour_offsets)
    ]

    mp4_path = os.path.join(OUT_DIR, "movieS01_evsom_node_weights.mp4")
    print(f"Saving MP4 → {mp4_path}")
    with tempfile.TemporaryDirectory() as tmp:
        for k, img in enumerate(mp4_frames):
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
