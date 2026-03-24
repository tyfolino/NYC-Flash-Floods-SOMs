"""
Movie S2 — evSOM Raw Mean Composites Animation (T−23h → T=0)

2×2 panel layout (A1, B1 / A2, B2), one frame per hour offset.
Mean Z500 in dam (contoured) + mean 850-hPa θe in K (shaded),
composited over events assigned to each node.
Matches the style of Fig. 3 (key-hours snapshot).

Output: movies/movieS02/movieS02_evsom_composite.gif

Usage:
    python -m figure_scripts.movieS02_evsom_composite_animation
"""

import io
import os
import subprocess
import tempfile

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from PIL import Image

from som_analysis.config import setup_plotting
from som_analysis.helpers import add_map_features, node_label

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "movies", "movieS02")
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
FIG_HEIGHT = 3.2
DPI_GIF = 300
DPI_MP4 = 200
FRAME_MS = 250  # milliseconds per frame

LEVELS_THETAE = np.arange(295, 341, 5)  # K
LEVELS_Z = range(549, 598, 2)  # dam

# Node layout: rows = j, cols = i  →  A1 B1 / A2 B2
NODE_LAYOUT = [[(i, j) for i in range(XDIM)] for j in range(YDIM)]


def _precompute_composites(z500_da, thetae_da, bmus):
    """Pre-compute node composite means for every hour offset."""
    hour_offsets = z500_da["hour_offset"].values

    composites = {}  # (i, j) → dict with 'z500', 'thetae', 'n'
    for i in range(XDIM):
        for j in range(YDIM):
            idx = np.where((bmus[:, 0] == i) & (bmus[:, 1] == j))[0]
            n = len(idx)
            # (n_hours, nlat, nlon)
            z500_node = z500_da.isel(event_time=idx).mean("event_time").values / 98.1
            thetae_node = thetae_da.isel(event_time=idx).mean("event_time").values
            composites[(i, j)] = {"z500": z500_node, "thetae": thetae_node, "n": n}
    return composites, hour_offsets


def _make_frame(composites, hour_offsets, lat, lon, hr_idx, dpi):
    """Render one frame and return a PIL Image."""
    hr_offset = int(hour_offsets[hr_idx])
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
            n = composites[(i, j)]["n"]

            thetae_comp = composites[(i, j)]["thetae"][hr_idx]
            z500_comp = composites[(i, j)]["z500"][hr_idx]

            im = ax.contourf(
                lon,
                lat,
                thetae_comp,
                cmap="BuPu",
                levels=LEVELS_THETAE,
                transform=proj,
                extend="both",
            )
            cn = ax.contour(
                lon,
                lat,
                z500_comp,
                colors="black",
                linewidths=0.4,
                levels=LEVELS_Z,
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

    # Time label centred above figure
    sign = "−" if hr_offset < 0 else "="
    hrs = abs(hr_offset)
    tlabel = f"T {sign} {hrs}h" if hrs > 0 else "T = 0"
    fig.suptitle(tlabel, fontsize=8)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02, aspect=28)
    cbar.set_label(r"850-hPa $\theta_e$ (K)", fontsize=6)
    cbar.ax.tick_params(labelsize=4.5)

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
    bmus = cached["bmus"]

    print("Loading raw evSOM fields ...")
    z500_var = list(xr.open_dataset(Z500_PATH).data_vars)[0]
    thetae_var = list(xr.open_dataset(THETAE_PATH).data_vars)[0]
    z500_da = xr.open_dataset(Z500_PATH)[z500_var]
    thetae_da = xr.open_dataset(THETAE_PATH)[thetae_var]

    lat = z500_da["latitude"].values
    lon = z500_da["longitude"].values

    print("Pre-computing composites ...")
    composites, hour_offsets = _precompute_composites(z500_da, thetae_da, bmus)
    n_hours = len(hour_offsets)

    print(f"Rendering {n_hours} frames (GIF @ {DPI_GIF} dpi) ...")
    gif_frames = []
    for hr_idx in range(n_hours):
        hr_offset = int(hour_offsets[hr_idx])
        if (hr_idx + 1) % 6 == 0 or hr_idx == 0:
            print(f"  Frame {hr_idx + 1}/{n_hours}  (T{hr_offset:+d}h)")
        gif_frames.append(
            _make_frame(composites, hour_offsets, lat, lon, hr_idx, DPI_GIF)
        )

    gif_path = os.path.join(OUT_DIR, "movieS02_evsom_composite.gif")
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
        _make_frame(composites, hour_offsets, lat, lon, hr_idx, DPI_MP4)
        for hr_idx in range(n_hours)
    ]

    mp4_path = os.path.join(OUT_DIR, "movieS02_evsom_composite.mp4")
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
