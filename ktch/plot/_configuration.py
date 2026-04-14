"""Configuration plot for landmark data."""

# Copyright 2026 Koji Noshita
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

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._base import require_dependencies


def _coerce_to_dataframe(
    data: pd.DataFrame | np.ndarray,
    x: str,
    y: str,
    z: str | None,
) -> pd.DataFrame:
    """Convert ndarray or DataFrame to a standardized DataFrame.

    Parameters
    ----------
    data : DataFrame or ndarray
        If ndarray, accepts 2D ``(n_landmarks, n_dim)`` for a single
        specimen or 3D ``(n_specimens, n_landmarks, n_dim)`` for multiple
        specimens.
    x, y : str
        Column names for x and y coordinates.
    z : str or None
        Column name for z coordinates, or ``None`` for 2D data.

    Returns
    -------
    DataFrame with appropriate index structure.
    """
    if isinstance(data, np.ndarray):
        dim_cols = [x, y] if z is None else [x, y, z]
        n_dim = len(dim_cols)
        if data.ndim == 2:
            if data.shape[1] < n_dim:
                raise ValueError(
                    f"Expected at least {n_dim} columns, "
                    f"got array with shape {data.shape}."
                )
            df = pd.DataFrame(data[:, :n_dim], columns=dim_cols)
            df.index.name = "coord_id"
            return df
        elif data.ndim == 3:
            if data.shape[2] < n_dim:
                raise ValueError(
                    f"Expected at least {n_dim} columns in last axis, "
                    f"got array with shape {data.shape}."
                )
            n_specimens, n_landmarks = data.shape[0], data.shape[1]
            specimen_ids = np.repeat(np.arange(n_specimens), n_landmarks)
            coord_ids = np.tile(np.arange(n_landmarks), n_specimens)
            values = data[:, :, :n_dim].reshape(-1, n_dim)
            df = pd.DataFrame(values, columns=dim_cols)
            df["specimen_id"] = specimen_ids
            df["coord_id"] = coord_ids
            return df.set_index(["specimen_id", "coord_id"])
        else:
            raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D.")
    return data


def _detect_coord_id_col(data: pd.DataFrame) -> str:
    """Detect the coordinate-id column name from index metadata.

    After ``reset_index()``, the coordinate-id column is the last level
    of the original index.
    """
    if isinstance(data.index, pd.MultiIndex):
        return data.index.names[-1]
    return data.index.name or "coord_id"


def _resolve_color_map(
    config: pd.DataFrame,
    hue: str,
    hue_order: Sequence,
    palette: str | Sequence | None,
) -> dict:
    """Build a mapping from hue values to colors."""
    import seaborn as sns

    if isinstance(palette, dict):
        missing = [k for k in hue_order if k not in palette]
        if missing:
            raise ValueError(f"palette dict is missing keys for hue values: {missing}")
        return {k: palette[k] for k in hue_order}
    elif palette is not None:
        colors = sns.color_palette(palette, n_colors=len(hue_order))
    else:
        if pd.api.types.is_numeric_dtype(config[hue]):
            cmap = sns.cubehelix_palette(as_cmap=True)
            hue_min, hue_max = min(hue_order), max(hue_order)
            colors = []
            for hue_val in hue_order:
                if hue_max > hue_min:
                    norm_val = (hue_val - hue_min) / (hue_max - hue_min)
                else:
                    norm_val = 0.5
                colors.append(cmap(norm_val))
        else:
            colors = sns.color_palette(n_colors=len(hue_order))
    return dict(zip(hue_order, colors))


def _draw_links_2d(
    config: pd.DataFrame,
    links: Sequence[Sequence[int]],
    coord_id_col: str,
    ax: Any,
    *,
    hue: str | None,
    hue_order: Sequence | None,
    color_map: dict | None,
    color_links: str,
    alpha: float,
    x: str,
    y: str,
) -> None:
    """Draw link segments on a 2D axes."""
    from matplotlib.collections import LineCollection

    if hue is None:
        segments = []
        for link in links:
            link_data = config[config[coord_id_col].isin(link)]
            if len(link_data) >= 2:
                coords = link_data[[x, y]].values[:2]
                segments.append(coords)
        if segments:
            lc = LineCollection(segments, colors=color_links, alpha=alpha)
            ax.add_collection(lc)
    else:
        for specimen in hue_order:
            specimen_data = config[config[hue] == specimen]
            segments = []
            for link in links:
                link_data = specimen_data[specimen_data[coord_id_col].isin(link)]
                if len(link_data) >= 2:
                    coords = link_data[[x, y]].values[:2]
                    segments.append(coords)
            if segments:
                lc = LineCollection(segments, colors=[color_map[specimen]], alpha=alpha)
                ax.add_collection(lc)
    ax.autoscale_view()


def _draw_links_3d(
    config: pd.DataFrame,
    links: Sequence[Sequence[int]],
    coord_id_col: str,
    ax: Any,
    *,
    hue: str | None,
    hue_order: Sequence | None,
    color_map: dict | None,
    color_links: str,
    alpha: float,
    x: str,
    y: str,
    z: str,
) -> None:
    """Draw link segments on a 3D axes."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    if hue is None:
        segments = []
        for link in links:
            link_data = config[config[coord_id_col].isin(link)]
            if len(link_data) >= 2:
                coords = link_data[[x, y, z]].values[:2]
                segments.append(coords)
        if segments:
            lc = Line3DCollection(segments, colors=color_links, alpha=alpha)
            ax.add_collection3d(lc)
    else:
        for specimen in hue_order:
            specimen_data = config[config[hue] == specimen]
            segments = []
            for link in links:
                link_data = specimen_data[specimen_data[coord_id_col].isin(link)]
                if len(link_data) >= 2:
                    coords = link_data[[x, y, z]].values[:2]
                    segments.append(coords)
            if segments:
                lc = Line3DCollection(
                    segments, colors=[color_map[specimen]], alpha=alpha
                )
                ax.add_collection3d(lc)


def _scatter_3d(
    config: pd.DataFrame,
    x: str,
    y: str,
    z: str,
    ax: Any,
    *,
    hue: str | None,
    hue_order: Sequence | None,
    color_map: dict | None,
    color: str,
    s: float,
    alpha: float,
) -> None:
    """Draw scatter points on a 3D axes."""
    if hue is None:
        ax.scatter(config[x], config[y], config[z], c=color, s=s, alpha=alpha)
    else:
        for specimen in hue_order:
            specimen_data = config[config[hue] == specimen]
            ax.scatter(
                specimen_data[x],
                specimen_data[y],
                specimen_data[z],
                c=[color_map[specimen]],
                s=s,
                alpha=alpha,
                label=specimen,
            )


def _set_box_aspect_equal(ax: Any, config: pd.DataFrame, x: str, y: str, z: str):
    """Set equal box aspect ratio for 3D axes."""
    ptp = [config[col].max() - config[col].min() for col in (x, y, z)]
    ptp = [max(p, 1e-10) for p in ptp]
    ax.set_box_aspect(ptp)


def configuration_plot(
    data: pd.DataFrame | np.ndarray,
    *,
    x: str = "x",
    y: str = "y",
    z: str | None = None,
    links: Sequence[Sequence[int]] | None = None,
    ax: object | None = None,
    hue: str | None = None,
    hue_order: Sequence | None = None,
    palette: str | Sequence | None = None,
    color: str = "gray",
    color_links: str | None = None,
    style: str | None = None,
    s: float = 10,
    alpha: float = 1.0,
) -> object:
    """Plot configurations of landmarks.

    Visualizes one or more configurations of landmarks (specimens)
    as scatter points with optional line segments connecting landmarks.
    Supports both 2D and 3D data.

    Parameters
    ----------
    data : pandas.DataFrame or numpy.ndarray
        Landmark coordinates. Accepts:

        - A DataFrame with ``MultiIndex (specimen_id, coord_id)`` for
          multiple specimens.
        - A DataFrame with a single ``coord_id`` index for one specimen
          (e.g., ``data.coords.loc[0]``).
        - A 2D array of shape ``(n_landmarks, n_dim)`` for a single
          specimen.
        - A 3D array of shape ``(n_specimens, n_landmarks, n_dim)`` for
          multiple specimens.

    x : str, default="x"
        Column name for x coordinates.
    y : str, default="y"
        Column name for y coordinates.
    z : str or None, default=None
        Column name for z coordinates. If given, a 3D plot is produced.
    links : sequence of sequence of int, optional
        Pairs of ``coord_id`` values to connect with line segments, e.g.,
        ``[[0, 1], [1, 2]]``.
    ax : matplotlib.axes.Axes or None, default=None
        Target axes. If ``None``, a new figure and axes are created.
        For 3D plots (``z`` is not ``None``), a ``projection='3d'`` axes
        is required; one is created automatically when ``ax=None``.
    hue : str or None, default=None
        Column name (after ``reset_index()``) for color grouping, e.g.,
        ``"specimen_id"``. When ``hue`` is active, ``color`` and
        ``color_links`` are ignored and colors are determined by the
        palette.
    hue_order : sequence or None, default=None
        Order of hue categories. If ``None``, uses the order of
        appearance.
    palette : str, sequence, dict, or None, default=None
        Seaborn palette name, explicit color list, or a dict mapping hue
        values to colors. A dict must contain all values in ``hue_order``.
    color : str, default="gray"
        Point color when ``hue`` is ``None``.
    color_links : str or None, default=None
        Link color when ``hue`` is ``None``. If ``None``, uses ``color``.
    style : str or None, default=None
        Column name for marker style differentiation (2D only, passed to
        ``seaborn.scatterplot``). Ignored for 3D plots.
    s : float, default=10
        Marker size.
    alpha : float, default=1.0
        Transparency for both points and links.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """
    require_dependencies("matplotlib", "seaborn")

    import matplotlib.pyplot as plt
    import seaborn as sns

    data = _coerce_to_dataframe(data, x, y, z)

    is_3d = z is not None

    # Validate axes dimensionality
    if ax is not None:
        ax_is_3d = hasattr(ax, "get_zlim")
        if is_3d and not ax_is_3d:
            raise ValueError(
                "z is specified but ax is not a 3D axes. "
                "Pass ax=None or create axes with projection='3d'."
            )
        if not is_3d and ax_is_3d:
            raise ValueError(
                "z is not specified but ax is a 3D axes. Pass ax=None or a 2D axes."
            )

    # Create axes if needed
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d" if is_3d else None)

    if color_links is None:
        color_links = color

    # Normalize input: ensure coord_id is available as a column
    coord_id_col = _detect_coord_id_col(data)
    config = data.reset_index()

    # Assign positional coord_id when the detected column does not exist
    if links and coord_id_col not in config.columns:
        config[coord_id_col] = range(len(config))

    # Resolve hue
    color_map = None
    if hue is not None:
        if hue_order is None:
            hue_order = list(config[hue].unique())
        color_map = _resolve_color_map(config, hue, hue_order, palette)

    # Draw links
    if links:
        if is_3d:
            _draw_links_3d(
                config,
                links,
                coord_id_col,
                ax,
                hue=hue,
                hue_order=hue_order,
                color_map=color_map,
                color_links=color_links,
                alpha=alpha,
                x=x,
                y=y,
                z=z,
            )
        else:
            _draw_links_2d(
                config,
                links,
                coord_id_col,
                ax,
                hue=hue,
                hue_order=hue_order,
                color_map=color_map,
                color_links=color_links,
                alpha=alpha,
                x=x,
                y=y,
            )

    # Draw points
    if is_3d:
        _scatter_3d(
            config,
            x,
            y,
            z,
            ax,
            hue=hue,
            hue_order=hue_order,
            color_map=color_map,
            color=color,
            s=s,
            alpha=alpha,
        )
        _set_box_aspect_equal(ax, config, x, y, z)
    else:
        if hue is not None:
            sns.scatterplot(
                data=config,
                x=x,
                y=y,
                ax=ax,
                hue=hue,
                hue_order=hue_order,
                palette=palette,
                style=style,
                alpha=alpha,
                s=s,
            )
        else:
            sns.scatterplot(
                data=config,
                x=x,
                y=y,
                ax=ax,
                color=color,
                style=style,
                alpha=alpha,
                s=s,
            )
        ax.set_aspect("equal")

    # Move legend outside when present
    if ax.get_legend() is not None:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    return ax
