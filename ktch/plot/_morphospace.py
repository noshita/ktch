"""Morphospace scatter plot with reconstructed shapes."""

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

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from ._base import require_dependencies
from ._params import (
    _detect_shape_type,
    _get_renderer_and_projection,
    _resolve_descriptor_params,
    _resolve_reducer_params,
    _validate_components,
)
from ._renderers import _resolve_render_kw


def morphospace_plot(
    data: Any | None = None,
    *,
    x: str | npt.ArrayLike | None = None,
    y: str | npt.ArrayLike | None = None,
    hue: str | npt.ArrayLike | None = None,
    reducer: Any | None = None,
    reducer_inverse_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    n_components: int | None = None,
    descriptor: Any | None = None,
    descriptor_inverse_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    components: tuple[int, int] = (0, 1),
    shape_type: str = "auto",
    render_fn: Callable[..., None] | None = None,
    n_dim: int | None = None,
    links: Sequence[Sequence[int]] | None = None,
    n_shapes: int = 5,
    shape_scale: float = 1.0,
    shape_color: str = "lightgray",
    shape_alpha: float = 0.7,
    palette: str | Sequence | None = None,
    hue_order: Sequence | None = None,
    scatter_kw: dict[str, Any] | None = None,
    ax: object | None = None,
    **render_kw: Any,
) -> object:
    """Scatter plot of specimens in morphospace with shape insets.

    Draws a scatter plot of scores from dimension reduction (reducer)
    and overlays reconstructed shapes at a regular grid of positions
    in the low-dimensional space.

    The function uses the same two-stage inverse transform pipeline as
    :func:`shape_variation_plot`:
    ``scores -> [reducer_inverse_transform] -> coefficients ->
    [descriptor_inverse_transform] -> shape coordinates``.

    This function calls ``fig.canvas.draw()`` internally to compute accurate
    pixel positions for inset axes.
    Inset positions are fixed at draw time and will not automatically update
    if the figure is later resized or saved at a different DPI.
    For best results, set the final figure size before calling this function.

    Parameters
    ----------
    data : DataFrame, optional
        DataFrame containing scores and metadata. If provided, ``x``, ``y``,
        ``hue`` refer to column names.
    x : str or array-like, optional
        Horizontal axis values (column name or array).
    y : str or array-like, optional
        Vertical axis values (column name or array).
    hue : str or array-like, optional
        Grouping variable for scatter coloring.
    reducer : fitted estimator, optional
        Convenience parameter. Extracts ``reducer_inverse_transform`` via
        ``.inverse_transform`` and ``n_components`` via ``.n_components_``
        (fallback to ``.n_components``).
    reducer_inverse_transform : callable, optional
        Overrides ``reducer.inverse_transform``.
    n_components : int, optional
        Overrides ``reducer.n_components_``.
    descriptor : fitted estimator, optional
        Convenience parameter. Extracts ``descriptor_inverse_transform``
        via ``.inverse_transform``.
    descriptor_inverse_transform : callable, optional
        Overrides ``descriptor.inverse_transform``.
    components : tuple of (int, int)
        0-indexed component indices for (horizontal, vertical) axes.
    shape_type : str
        Shape rendering type. One of ``"auto"``, ``"curve_2d"``,
        ``"curve_3d"``, ``"surface_3d"``, ``"landmarks_2d"``,
        ``"landmarks_3d"``.
    render_fn : callable, optional
        Custom renderer ``(coords, ax, **kw) -> None``.
    n_dim : int, optional
        Spatial dimensionality (for GPA identity case).
    links : sequence of sequence of int, optional
        Landmark link pairs.
    n_shapes : int
        Number of shapes along each axis (total: ``n_shapes * n_shapes``).
    shape_scale : float
        Scale factor for inset shape size.
    shape_color : str
        Color for reconstructed shapes.
    shape_alpha : float
        Transparency for reconstructed shapes.
    palette : str or sequence, optional
        Forwarded to ``sns.scatterplot``.
    hue_order : sequence, optional
        Forwarded to ``sns.scatterplot``.
    scatter_kw : dict, optional
        Additional kwargs forwarded to ``sns.scatterplot``.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes. If ``None``, creates new figure and axes.
    **render_kw
        Forwarded to the shape renderer.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The main scatter plot axes.

    Raises
    ------
    ImportError
        If matplotlib or seaborn are not installed.
    ValueError
        If required parameters cannot be resolved.

    Examples
    --------
    >>> from ktch.plot import morphospace_plot
    >>> ax = morphospace_plot(  # doctest: +SKIP
    ...     data=df_pca,
    ...     x="PC1", y="PC2", hue="genus",
    ...     reducer=pca,
    ...     descriptor=efa,
    ...     palette="Paired",
    ...     n_shapes=5,
    ...     shape_scale=0.8,
    ... )
    """
    require_dependencies("matplotlib", "seaborn")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create or reuse axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Draw scatter plot (if data provided)
    if x is not None and y is not None:
        sns.scatterplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            palette=palette,
            hue_order=hue_order,
            ax=ax,
            **(scatter_kw or {}),
        )

    # Resolve reducer/descriptor parameters (if reducer available)
    if reducer is not None or reducer_inverse_transform is not None:
        reducer_inverse_transform, _, n_components = _resolve_reducer_params(
            reducer,
            reducer_inverse_transform,
            explained_variance=None,
            n_components=n_components,
            require_variance=False,
        )
        descriptor_inverse_transform, n_dim = _resolve_descriptor_params(
            descriptor,
            descriptor_inverse_transform,
            n_dim,
            shape_type,
        )
        _validate_components(components, n_components)

        # Overlay shapes
        fig.canvas.draw()  # force layout for accurate positions

        comp_h, comp_v = components
        x_range = np.linspace(*ax.get_xlim(), n_shapes)
        y_range = np.linspace(*ax.get_ylim(), n_shapes)

        # Batch reconstruction
        grid = np.array([(h, v) for h in x_range for v in y_range])
        all_scores = np.zeros((len(grid), n_components))
        all_scores[:, comp_h] = grid[:, 0]
        all_scores[:, comp_v] = grid[:, 1]

        all_coeffs = reducer_inverse_transform(all_scores)
        if descriptor_inverse_transform is not None:
            all_coords = np.asarray(descriptor_inverse_transform(all_coeffs))
        else:
            all_coords = all_coeffs.reshape(len(grid), -1, n_dim)

        # Auto-detect shape_type if needed
        if shape_type == "auto":
            shape_type = _detect_shape_type(
                all_coords[0],
                descriptor_inverse_transform,
                n_dim,
            )

        renderer, proj = _get_renderer_and_projection(shape_type, render_fn)

        # Compute inset sizing
        ax_extent = ax.get_window_extent()
        fig_extent = fig.get_window_extent()
        fig_width = fig_extent.width
        fig_height = fig_extent.height
        inset_size = shape_scale * ax_extent.width / (fig_width * n_shapes)

        resolved = _resolve_render_kw(
            render_kw,
            color=shape_color,
            alpha=shape_alpha,
            links=links,
        )

        for idx, (score_h, score_v) in enumerate(grid):
            single = all_coords[idx]

            loc = ax.transData.transform((score_h, score_v))
            axins = fig.add_axes(
                [
                    loc[0] / fig_width - inset_size / 2,
                    loc[1] / fig_height - inset_size / 2,
                    inset_size,
                    inset_size,
                ],
                anchor="C",
                projection=proj,
            )

            renderer(single, axins, **resolved)
            axins.axis("off")

        ax.set_zorder(1)  # draw scatter on top of inset shapes
        ax.patch.set_alpha(0)  # transparent background so insets show through

    return ax
