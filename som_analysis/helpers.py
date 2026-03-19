"""
Reusable helper functions for the NYC flash flood SOM analysis.

Plotting utilities, composite computation, and data loading helpers
shared across train_som.py, plot_som.py, and node_statistics.py.
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage
import xarray as xr


def has_cutoff_low(z500_field, contour_spacing=6):
    """Return True if a cutoff low is present in a Z500 field.

    A cutoff low is identified as a local minimum whose enclosing contour
    at (local_minimum + *contour_spacing*) forms a closed loop that does
    not touch the domain boundary.

    Parameters
    ----------
    z500_field : 2-D array-like
        Z500 values in decameters (dam).
    contour_spacing : float
        Contour interval in dam; the closed-contour test level is set at
        (local_minimum + contour_spacing). Default is 6 dam.
    """
    z500 = np.asarray(z500_field)

    # Find local minima (grid points equal to the minimum in a 5×5 window)
    min_filtered = scipy.ndimage.minimum_filter(z500, size=5)
    local_minima = np.argwhere(z500 == min_filtered)

    for i, j in local_minima:
        # Skip minima too close to the domain edge
        if i < 2 or i >= z500.shape[0] - 2 or j < 2 or j >= z500.shape[1] - 2:
            continue

        contour_level = z500[i, j] + contour_spacing
        labeled, _ = scipy.ndimage.label(z500 < contour_level)
        region_label = labeled[i, j]
        if region_label == 0:
            continue

        region_mask = labeled == region_label
        touches_boundary = (
            np.any(region_mask[0, :])
            or np.any(region_mask[-1, :])
            or np.any(region_mask[:, 0])
            or np.any(region_mask[:, -1])
        )
        if not touches_boundary:
            return True

    return False


def load_moist_var(filepath, var_name):
    """Load a moisture variable, handling variable-name variations across files."""
    ds = xr.load_dataset(filepath)
    return ds[var_name] if var_name in ds else ds[list(ds.data_vars)[0]]


def node_label(i, j):
    """Convert SOM grid indices to alphanumeric label.

    Letter indexes the row top-to-bottom (A, B, C, ...) and number indexes
    the column left-to-right (1, 2, 3, ...).  So the top-left node (i=0, j=0)
    is 'A1' and the top-right node (i=xdim-1, j=0) is e.g. 'A5'.

    Parameters
    ----------
    i : int
        Column index (0 = left).
    j : int
        Row index (0 = top).
    """
    return f"{chr(ord('A') + j)}{i + 1}"


def get_node_indices(bmus, i, j):
    """Get sample indices belonging to SOM node (i, j)."""
    return np.where((bmus[:, 0] == i) & (bmus[:, 1] == j))[0]


def compute_composites(data, bmus, xdim, ydim, time_dim="valid_time"):
    """Compute composite mean for each SOM node.

    Returns (composites, counts) arrays of shape (xdim, ydim, ...) and
    (xdim, ydim) respectively.
    """
    sample_shape = data.isel({time_dim: 0}).shape
    composites = np.full((xdim, ydim) + sample_shape, np.nan)
    counts = np.zeros((xdim, ydim), dtype=int)

    for i in range(xdim):
        for j in range(ydim):
            idx = get_node_indices(bmus, i, j)
            counts[i, j] = len(idx)
            if len(idx) > 0:
                composites[i, j] = data.isel({time_dim: idx}).mean(time_dim).values
    return composites, counts


def create_som_figure(xdim, ydim, figsize=(6, 3.7), dpi=600):
    """Create a standard figure for SOM node plots."""
    fig, axes = plt.subplots(
        ydim,
        xdim,
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
        dpi=dpi,
    )
    return fig, axes


def add_map_features(ax):
    """Add standard map features to an axis."""
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.4)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_node_events(
    data,
    bmus,
    xdim,
    ydim,
    lon,
    lat,
    levels,
    cmap,
    save_pattern,
    scale=1.0,
    cbar_label=None,
    time_dim="valid_time",
    contour=False,
    z500_data=None,
    z500_levels=None,
    z500_scale=1.0,
    z500_time_dim="time",
):
    """Plot individual events for each SOM node.

    Parameters
    ----------
    data : xarray.DataArray
        Primary field to plot.
    bmus : ndarray
        Best-matching unit assignments, shape (n_events, 2).
    xdim, ydim : int
        SOM grid dimensions.
    lon, lat : array-like
        Coordinate arrays.
    levels : array-like
        Contour levels.
    cmap : str or None
        Colormap for filled contours (ignored when contour=True).
    save_pattern : str
        File path pattern with {i} and {j} placeholders.
    scale : float
        Multiplicative scale applied to field values before plotting.
    cbar_label : str, optional
        Colorbar label.
    time_dim : str
        Name of time dimension in data.
    contour : bool
        If True, plot line contours instead of filled contours.
    z500_data : xarray.DataArray, optional
        Z500 data to overlay as contours.
    z500_levels : array-like, optional
        Contour levels for Z500.
    z500_scale : float
        Scale factor for Z500 values.
    z500_time_dim : str
        Name of time dimension in z500_data.
    """
    cols = 5
    proj = ccrs.PlateCarree()

    for i in range(xdim):
        for j in range(ydim):
            idx = get_node_indices(bmus, i, j)
            n = len(idx)
            if n == 0:
                continue
            rows = int(np.ceil(n / cols))

            fig, axes = plt.subplots(
                rows,
                cols,
                figsize=(3 * cols, 2.5 * rows),
                subplot_kw={"projection": proj},
                layout="constrained",
                dpi=300,
            )

            for k, ax in enumerate(axes.flat):
                if k < n:
                    field = data.isel({time_dim: idx[k]})
                    time_val = field[time_dim].values

                    if contour:
                        im = ax.contour(
                            lon,
                            lat,
                            field.values * scale,
                            levels=levels,
                            colors="black",
                            transform=proj,
                            linewidths=0.6,
                        )
                        ax.clabel(im, im.levels, fontsize=5)
                    else:
                        im = ax.contourf(
                            lon,
                            lat,
                            field.values * scale,
                            levels=levels,
                            cmap=cmap,
                            transform=proj,
                            extend="max",
                        )

                        if z500_data is not None and z500_levels is not None:
                            z500_field = z500_data.isel({z500_time_dim: idx[k]})
                            cn = ax.contour(
                                lon,
                                lat,
                                z500_field.values * z500_scale,
                                levels=z500_levels,
                                colors="black",
                                linewidths=0.5,
                                transform=proj,
                            )
                            ax.clabel(cn, inline=True, fontsize=5, fmt="%.0f")

                    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
                    ax.add_feature(cfeature.STATES, linewidth=0.2)
                    ax.set_title(str(pd.to_datetime(time_val))[:16])
                else:
                    ax.axis("off")

            if cbar_label and not contour:
                cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
                cbar.set_label(cbar_label, fontsize=6)

            fig.suptitle(f"Node {node_label(i, j)}  N={n}", fontsize=8, y=1.02)
            plt.savefig(save_pattern.format(i=i, j=j))
            plt.close(fig)


def plot_single_events(
    data,
    bmus,
    xdim,
    ydim,
    lon,
    lat,
    levels,
    cmap,
    save_dir,
    suffix="",
    scale=1.0,
    cbar_label=None,
    time_dim="valid_time",
    contour=False,
    z500_data=None,
    z500_levels=None,
    z500_scale=1.0,
    z500_time_dim="time",
):
    """Save each event as an individual figure.

    Files are saved to ``save_dir/node_<i>_<j>/<timestamp><suffix>.png``.

    Parameters match :func:`plot_node_events` except *save_dir* replaces
    *save_pattern* and *suffix* is appended before the file extension
    (e.g. ``suffix="_precip"``).
    """
    import os

    proj = ccrs.PlateCarree()

    for i in range(xdim):
        for j in range(ydim):
            node_dir = os.path.join(save_dir, f"node_{i}_{j}")
            os.makedirs(node_dir, exist_ok=True)

            idx = get_node_indices(bmus, i, j)

            for k in idx:
                field = data.isel({time_dim: k})
                time_val = field[time_dim].values
                ts = str(pd.to_datetime(time_val).strftime("%Y-%m-%d_%HZ"))

                fig, ax = plt.subplots(
                    figsize=(4, 3),
                    subplot_kw={"projection": proj},
                    dpi=200,
                    constrained_layout=True,
                )

                if contour:
                    im = ax.contour(
                        lon, lat, field.values * scale,
                        levels=levels, colors="black",
                        transform=proj, linewidths=0.6,
                    )
                    ax.clabel(im, im.levels, fontsize=5)
                else:
                    im = ax.contourf(
                        lon, lat, field.values * scale,
                        levels=levels, cmap=cmap,
                        transform=proj, extend="max",
                    )

                    if z500_data is not None and z500_levels is not None:
                        z500_field = z500_data.isel({z500_time_dim: k})
                        cn = ax.contour(
                            lon, lat,
                            z500_field.values * z500_scale,
                            levels=z500_levels, colors="black",
                            linewidths=0.5, transform=proj,
                        )
                        ax.clabel(cn, inline=True, fontsize=5, fmt="%.0f")

                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linewidth=0.3)
                ax.add_feature(cfeature.STATES, linewidth=0.2)

                if cbar_label and not contour:
                    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
                    cbar.set_label(cbar_label, fontsize=6)

                ax.set_title(
                    f"Node {node_label(i, j)}  {ts.replace('_', ' ')}", fontsize=7
                )

                plt.savefig(
                    os.path.join(node_dir, f"{ts}{suffix}.png"),
                    bbox_inches="tight",
                )
                plt.close(fig)
