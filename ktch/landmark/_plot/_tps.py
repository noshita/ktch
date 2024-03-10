"""Plot functions for thin-plate spline warping."""

# Copyright 2024 Koji Noshita
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from .._Procrustes_analysis import _thin_plate_spline_2d, _tps_2d


def tps_grid_2d_plot(
    x_reference, x_target, grid_size="auto", outer=0.1, n_grid_inner=10, ax=None
):
    """Plot the thin-plate spline 2D warped grid.

    Parameters
    ----------
    x_reference : array-like, shape (n_landmarks, n_dim)
        Reference configuration.
    x_target : array-like, shape (n_landmarks, n_dim)
        Target configuration.
    grid_size : str/float, optional
        Grid size, by default "auto"
    outer : float, optional
        Outer range of x_reference covered by the grid, by default 0.1
    n_grid_inner : int, optional
        Number of inner points on each grid, by default 10
    ax : :class:`matplotlib.axes.Axes`, optional
        Pre-existing matplotlib axes for the plot. Otherwise, call :func:`matplotlib.pyplot.gca` internally.

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
        Matplotlib axes.
    """

    import matplotlib.pyplot as plt

    W, c, A = _thin_plate_spline_2d(x_reference, x_target)

    if ax is None:
        ax = plt.gca()

    x_min, y_min = (1 + outer) * np.min(x_reference, axis=0)
    x_max, y_max = (1 + outer) * np.max(x_reference, axis=0)

    w = x_max - x_min
    h = y_max - y_min

    grid_size_ = grid_size
    if grid_size == "auto":
        grid_size_ = np.min([w, h]) / 10

    if w > h:
        w = w - w % grid_size_ + grid_size_
    else:
        h = h - w % grid_size + grid_size_
    n_grid_x = np.rint(w / grid_size_)
    n_grid_y = np.rint(h / grid_size_)

    n_grid_x_ = int(n_grid_x * n_grid_inner + 1)
    n_grid_y_ = int(n_grid_y * n_grid_inner + 1)

    warped = np.array(
        [
            _tps_2d(x, y, x_reference, W, c, A)
            for x in np.linspace(x_min, x_max, n_grid_x_)
            for y in np.linspace(y_min, y_max, n_grid_y_)
        ]
    )

    w_1 = warped.reshape(n_grid_x_, n_grid_y_, 2)
    w_2 = w_1.transpose(1, 0, 2)

    ax.plot(w_1[:, ::n_grid_inner, 0], w_1[:, ::n_grid_inner, 1], "gray")
    ax.plot(w_2[:, ::n_grid_inner, 0], w_2[:, ::n_grid_inner, 1], "gray")
    ax.axis("equal")

    ax.scatter(x=x_reference[:, 0], y=x_reference[:, 1], zorder=2)
    ax.scatter(x=x_target[:, 0], y=x_target[:, 1], zorder=2)
    return ax
