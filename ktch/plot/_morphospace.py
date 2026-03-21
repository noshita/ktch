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

import warnings
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
    _resolve_xy_hue,
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

    Notes
    -----
    When ``shape_type="auto"`` (the default), the type is inferred from a
    single specimen (batch dimension removed) of the descriptor inverse
    transform output:

    - ``(m, n, 3)`` (ndim=3) -> ``"surface_3d"``
    - ``(t, 2)`` (ndim=2, last dim 2) -> ``"curve_2d"``
    - ``(t, k)`` (ndim=2, last dim >= 3) -> ``"curve_3d"``
    - No descriptor (identity / GPA case) with ``n_dim=2`` -> ``"landmarks_2d"``
    - No descriptor (identity / GPA case) with ``n_dim=3`` -> ``"landmarks_3d"``

    For per-specimen shapes with ``shape[-1] == 3`` and ndim=2,
    auto-detection chooses ``"curve_3d"``. If the data represents
    landmarks, specify ``shape_type="landmarks_3d"`` explicitly.

    3-D shape types (``"surface_3d"``, ``"curve_3d"``, ``"landmarks_3d"``)
    use matplotlib 3-D projection for each inset, which is significantly
    slower. For 3-D surfaces (e.g., SHA), consider using ``n_shapes <= 3``
    and reducing surface resolution via a
    ``descriptor_inverse_transform`` wrapper.

    See Also
    --------
    shape_variation_plot : Shape grid along component axes.
    explained_variance_ratio_plot : Scree plot of explained variance.

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

    if n_shapes < 1:
        raise ValueError(f"n_shapes must be >= 1, got {n_shapes}")

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


#
# Distribution overlays
#


def _iter_overlay_groups(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    hue_arr: np.ndarray | None,
    categories: list | None,
    *,
    color: Any,
    palette: str | Sequence | None,
    min_points: int,
    overlay_name: str,
):
    """Yield ``(xi, yi, group_color, category_name)`` for each drawable group.

    Handles single-group and per-hue-group iteration, color resolution
    via seaborn palette, and minimum-point-count warnings.

    Parameters
    ----------
    x_arr, y_arr : ndarray
        Coordinate arrays.
    hue_arr : ndarray or None
        Group labels (None for a single group).
    categories : list or None
        Ordered unique hue values.
    color : color
        Single color for the no-hue case (falls back to ``"C0"``).
    palette : str, sequence, or None
        Seaborn palette for hue groups.
    min_points : int
        Minimum number of points required to draw the overlay.
    overlay_name : str
        Human-readable name for warning messages (e.g. ``"confidence ellipse"``).

    Yields
    ------
    xi : ndarray
    yi : ndarray
    group_color : color
    category_name : str or None
        ``None`` for the single-group (no-hue) case, otherwise the
        category value.
    """
    if hue_arr is None:
        c = color if color is not None else "C0"
        if len(x_arr) < min_points:
            warnings.warn(
                f"Skipped: need at least {min_points} data points "
                f"for {overlay_name} (got {len(x_arr)})",
                UserWarning,
                stacklevel=3,
            )
            return
        yield x_arr, y_arr, c, None
    else:
        require_dependencies("seaborn")
        import seaborn as sns

        colors = sns.color_palette(palette, n_colors=len(categories))
        color_map = dict(zip(categories, colors))
        for cat in categories:
            mask = hue_arr == cat
            xi, yi = x_arr[mask], y_arr[mask]
            if len(xi) < min_points:
                warnings.warn(
                    f"Category {cat!r} skipped: need at least {min_points} "
                    f"data points for {overlay_name} (got {len(xi)})",
                    UserWarning,
                    stacklevel=3,
                )
                continue
            yield xi, yi, color_map[cat], cat


def _draw_confidence_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    ax: object,
    *,
    n_std: float,
    fill: bool,
    alpha: float,
    color: Any,
    linewidth: float,
    label: str | None = None,
    **kwargs: Any,
) -> bool:
    """Draw a single confidence ellipse on *ax*.

    Returns True on success, False if the ellipse is degenerate
    (zero variance in x or y).
    """
    import matplotlib.patches as mpatches
    import matplotlib.transforms as transforms

    cov = np.cov(x, y)
    var_product = cov[0, 0] * cov[1, 1]
    if var_product <= 0:
        return False

    pearson = cov[0, 1] / np.sqrt(var_product)
    pearson = np.clip(pearson, -1.0, 1.0)

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    facecolor = color if fill else "none"

    ellipse = mpatches.Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        edgecolor=color,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
        **kwargs,
    )

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)
    return True


def confidence_ellipse_plot(
    data: Any | None = None,
    *,
    x: str | npt.ArrayLike | None = None,
    y: str | npt.ArrayLike | None = None,
    hue: str | npt.ArrayLike | None = None,
    confidence: float = 0.95,
    n_std: float | None = None,
    fill: bool = False,
    alpha: float | None = None,
    palette: str | Sequence | None = None,
    hue_order: Sequence | None = None,
    color: Any | None = None,
    linewidth: float = 1.0,
    legend: bool = True,
    ax: object | None = None,
    **kwargs: Any,
) -> object:
    r"""Draw confidence ellipses for groups on a scatter plot.

    Overlays covariance-based confidence ellipses on existing axes,
    colored by *hue* groups. Designed to be combined with
    :func:`morphospace_plot` or ``seaborn.scatterplot``.

    The ellipse size is determined by *confidence* (default 0.95),
    converted internally via
    :math:`n_{\text{std}} = \sqrt{-2 \ln(1 - p)}`.
    Alternatively, *n_std* can be given directly to bypass the
    conversion.

    Parameters
    ----------
    data : DataFrame, optional
        DataFrame containing the data. When provided, *x*, *y*, and
        *hue* should be column names.
    x : str or array-like
        Horizontal axis values.
    y : str or array-like
        Vertical axis values.
    hue : str or array-like, optional
        Grouping variable. One ellipse is drawn per group.
    confidence : float
        Confidence level in the open interval (0, 1). Determines the
        ellipse size assuming a bivariate normal distribution.
        Ignored when *n_std* is given.
    n_std : float or None
        Number of standard deviations for the ellipse radii. When
        given, overrides *confidence*.
    fill : bool
        If True, fill the interior of each ellipse.
    alpha : float or None
        Opacity. Defaults to 0.25 when *fill* is True, 1.0 otherwise.
    palette : str or sequence, optional
        Seaborn color palette for hue groups.
    hue_order : sequence, optional
        Order and subset of hue levels to plot.
    color : color, optional
        Single color used when *hue* is not set.
        Defaults to ``"C0"``.
    linewidth : float
        Ellipse edge width.
    legend : bool
        If True, set *label* on each ellipse so it can appear in a
        legend created by ``ax.legend()``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Defaults to ``plt.gca()``.
    **kwargs
        Forwarded to :class:`matplotlib.patches.Ellipse`.

    Returns
    -------
    matplotlib.axes.Axes

    See Also
    --------
    convex_hull_plot : Convex hull overlay.
    morphospace_plot : Scatter plot with shape insets.

    Examples
    --------
    >>> from ktch.plot import morphospace_plot, confidence_ellipse_plot
    >>> ax = morphospace_plot(  # doctest: +SKIP
    ...     data=df, x="PC1", y="PC2", hue="species",
    ...     reducer=pca, descriptor=efa,
    ... )
    >>> confidence_ellipse_plot(  # doctest: +SKIP
    ...     data=df, x="PC1", y="PC2", hue="species",
    ...     ax=ax,
    ... )
    """
    require_dependencies("matplotlib")
    import matplotlib.pyplot as plt

    if n_std is None:
        if not 0 < confidence < 1:
            raise ValueError(
                "confidence must be in the open interval (0, 1), "
                f"got {confidence}"
            )
        n_std = float(np.sqrt(-2.0 * np.log(1.0 - confidence)))

    if ax is None:
        ax = plt.gca()

    x_arr, y_arr, hue_arr, categories = _resolve_xy_hue(data, x, y, hue, hue_order)

    if alpha is None:
        alpha = 0.25 if fill else 1.0

    for xi, yi, c, cat in _iter_overlay_groups(
        x_arr, y_arr, hue_arr, categories,
        color=color, palette=palette,
        min_points=2, overlay_name="confidence ellipse",
    ):
        label = None if cat is None else (cat if legend else "_nolegend_")
        drawn = _draw_confidence_ellipse(
            xi,
            yi,
            ax,
            n_std=n_std,
            fill=fill,
            alpha=alpha,
            color=c,
            linewidth=linewidth,
            label=label,
            **kwargs,
        )
        if not drawn:
            prefix = f"Category {cat!r} skipped" if cat is not None else "Skipped"
            warnings.warn(
                f"{prefix}: zero variance in x or y",
                UserWarning,
                stacklevel=2,
            )

    return ax


def _draw_convex_hull(
    x: np.ndarray,
    y: np.ndarray,
    ax: object,
    *,
    fill: bool,
    alpha: float,
    color: Any,
    linewidth: float,
    label: str | None = None,
    **kwargs: Any,
) -> None:
    """Draw a single convex hull on *ax*.

    May raise ``QhullError`` if the points are degenerate.
    """
    import matplotlib.patches as mpatches
    from scipy.spatial import ConvexHull

    points = np.column_stack([x, y])
    hull = ConvexHull(points)
    verts = hull.vertices

    if fill:
        polygon = mpatches.Polygon(
            points[verts],
            closed=True,
            facecolor=color,
            edgecolor=color,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            **kwargs,
        )
        ax.add_patch(polygon)
    else:
        hull_x = np.append(points[verts, 0], points[verts[0], 0])
        hull_y = np.append(points[verts, 1], points[verts[0], 1])
        ax.plot(
            hull_x,
            hull_y,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            **kwargs,
        )


def convex_hull_plot(
    data: Any | None = None,
    *,
    x: str | npt.ArrayLike | None = None,
    y: str | npt.ArrayLike | None = None,
    hue: str | npt.ArrayLike | None = None,
    fill: bool = False,
    alpha: float | None = None,
    palette: str | Sequence | None = None,
    hue_order: Sequence | None = None,
    color: Any | None = None,
    linewidth: float = 1.0,
    legend: bool = True,
    ax: object | None = None,
    **kwargs: Any,
) -> object:
    """Draw convex hulls for groups on a scatter plot.

    Overlays convex hulls on existing axes, colored by *hue* groups.
    Designed to be combined with :func:`morphospace_plot` or
    ``seaborn.scatterplot``.

    Parameters
    ----------
    data : DataFrame, optional
        DataFrame containing the data. When provided, *x*, *y*, and
        *hue* should be column names.
    x : str or array-like
        Horizontal axis values.
    y : str or array-like
        Vertical axis values.
    hue : str or array-like, optional
        Grouping variable. One hull is drawn per group.
    fill : bool
        If True, fill the interior of each hull.
    alpha : float or None
        Opacity. Defaults to 0.2 when *fill* is True, 1.0 otherwise.
    palette : str or sequence, optional
        Seaborn color palette for hue groups.
    hue_order : sequence, optional
        Order and subset of hue levels to plot.
    color : color, optional
        Single color used when *hue* is not set.
        Defaults to ``"C0"``.
    linewidth : float
        Hull edge width.
    legend : bool
        If True, set *label* on each hull so it can appear in a
        legend created by ``ax.legend()``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Defaults to ``plt.gca()``.
    **kwargs
        Forwarded to :meth:`~matplotlib.axes.Axes.plot` (when
        *fill* is False) or :class:`matplotlib.patches.Polygon`
        (when *fill* is True).

    Returns
    -------
    matplotlib.axes.Axes

    See Also
    --------
    confidence_ellipse_plot : Confidence ellipse overlay.
    morphospace_plot : Scatter plot with shape insets.

    Examples
    --------
    >>> from ktch.plot import morphospace_plot, convex_hull_plot
    >>> ax = morphospace_plot(  # doctest: +SKIP
    ...     data=df, x="PC1", y="PC2", hue="species",
    ...     reducer=pca, descriptor=efa,
    ... )
    >>> convex_hull_plot(  # doctest: +SKIP
    ...     data=df, x="PC1", y="PC2", hue="species",
    ...     ax=ax,
    ... )
    """
    require_dependencies("matplotlib")
    import matplotlib.pyplot as plt
    from scipy.spatial import QhullError

    if ax is None:
        ax = plt.gca()

    x_arr, y_arr, hue_arr, categories = _resolve_xy_hue(data, x, y, hue, hue_order)

    if alpha is None:
        alpha = 0.2 if fill else 1.0

    for xi, yi, c, cat in _iter_overlay_groups(
        x_arr, y_arr, hue_arr, categories,
        color=color, palette=palette,
        min_points=3, overlay_name="convex hull",
    ):
        label = None if cat is None else (cat if legend else "_nolegend_")
        try:
            _draw_convex_hull(
                xi,
                yi,
                ax,
                fill=fill,
                alpha=alpha,
                color=c,
                linewidth=linewidth,
                label=label,
                **kwargs,
            )
        except QhullError:
            prefix = f"Category {cat!r} skipped" if cat is not None else "Skipped"
            warnings.warn(
                f"{prefix}: unable to compute convex hull "
                f"(points may be collinear)",
                UserWarning,
                stacklevel=2,
            )

    return ax
