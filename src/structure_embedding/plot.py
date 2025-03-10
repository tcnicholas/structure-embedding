from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde


def map_axis(ax: plt.Axes) -> None:
    """
    Create a blank plot with no spines or ticks. 
    
    Parameters
    ----------
    ax
        The axis to modify.
    """

    for loc in ['top', 'right', 'left', 'bottom']:
        ax.spines[loc].set_visible(False)

    for axis in ['x', 'y']:
        getattr(ax, f"{axis}axis").set_ticks([])

    ax.set_aspect('equal')


def white_to_colour_cmap(base_color: str) -> LinearSegmentedColormap:
    """
    Create a custom colormap that transitions from transparent white to a
    specified base colour.

    Parameters
    ----------
    base_color
        The base colour to transition to.

    Returns
    -------
    A LinearSegmentedColormap object.
    """

    base_rgba = mcolors.to_rgba(base_color, alpha=1)
    transparent_white = (1, 1, 1, 0.)
    return mcolors.LinearSegmentedColormap.from_list(
        'white_to_colour_cmap',
        [transparent_white, base_rgba]
    )


def plot_density_2d(
    points: npt.NDArray[np.floating],
    cmap: LinearSegmentedColormap,
    ax: plt.Axes | None = None,
    levels: int = 10,
    grid_resolution: int = 200,
) -> None:
    """
    Plot a 2D density map of the given points.

    Parameters
    ----------
    points
        The points to plot.
    cmap
        The colormap to use.
    ax
        The axis to plot on. If None, the current axis is used.
    levels
        The number of levels to plot.
    grid_resolution
        The resolution of the grid to use.
    """

    if points.shape[0] < 2:
        raise ValueError("At least two points are required.")
    
    if ax is None:
        ax = plt.gca()

    x, y = points[:, 0], points[:, 1]
    kde = gaussian_kde(np.vstack([x, y]))

    xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
    zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

    xi, yi = np.mgrid[
        x.min()-0.5:x.max()+0.5:grid_resolution*1j, 
        y.min()-0.5:y.max()+0.5:grid_resolution*1j
    ]
    zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

    ax.contourf(xi, yi, zi, levels=levels, cmap=cmap)