"""Plot functions for PCA results."""

# Copyright 2025 Koji Noshita
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

from ._base import require_dependencies
from ._params import (
    _detect_shape_type,
    _get_renderer_and_projection,
    _resolve_descriptor_params,
    _resolve_reducer_params,
    _validate_components,
)
from ._renderers import _resolve_render_kw


def explained_variance_ratio_plot(
    pca: Any,
    n_components: int | None = None,
    ax: object | None = None,
    verbose: bool = False,
) -> object:
    """Plot explained variance ratio of PCA components.

    Parameters
    ----------
    pca : sklearn.decomposition.PCA
        Fitted PCA object.
    n_components : int, optional
        Number of principal components to plot. If None, plot all components.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes are created.
    verbose : bool, optional
        If True, print explained variance ratios and their cumulative sums.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object with the plot.

    Raises
    ------
    ImportError
        If matplotlib or seaborn are not installed.
    """
    require_dependencies("matplotlib", "seaborn")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    if n_components is None:
        n_components = pca.n_components_

    max_components = len(pca.explained_variance_ratio_)
    if n_components > max_components:
        raise ValueError(
            f"n_components ({n_components}) exceeds the number of fitted "
            f"components ({max_components})."
        )

    pc_evr = pca.explained_variance_ratio_[0:n_components]
    pc_cum = np.cumsum(pc_evr)

    if verbose:
        print("Explained variance ratio:")
        print(["PC" + str(i + 1) + " " + str(val) for i, val in enumerate(pc_evr)])

        print("Cumsum of Explained variance ratio:")
        print(["PC" + str(i + 1) + " " + str(val) for i, val in enumerate(pc_cum)])

    sns.barplot(
        x=["PC" + str(i + 1) for i in range(n_components)],
        y=pc_evr,
        color="gray",
        ax=ax,
    )
    sns.lineplot(
        x=["PC" + str(i + 1) for i in range(n_components)],
        y=pc_cum,
        color="gray",
        ax=ax,
    )
    sns.scatterplot(
        x=["PC" + str(i + 1) for i in range(n_components)],
        y=pc_cum,
        color="gray",
        ax=ax,
    )
    return ax


def shape_variation_plot(
    reducer: Any | None = None,
    *,
    descriptor: Any | None = None,
    descriptor_inverse_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    reducer_inverse_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    explained_variance: np.ndarray | None = None,
    n_components: int | None = None,
    components: Sequence[int] = (0, 1, 2),
    sd_values: Sequence[float] = (-2.0, -1.0, 0.0, 1.0, 2.0),
    shape_type: str = "auto",
    render_fn: Callable[..., None] | None = None,
    n_dim: int | None = None,
    links: Sequence[Sequence[int]] | None = None,
    color: str = "gray",
    alpha: float = 1.0,
    fig: object | None = None,
    dpi: int = 150,
    figscale: float = 3.0,
    **render_kw: Any,
) -> object:
    """Plot reconstructed shapes along component axes.

    Creates a grid of subplots showing shape variation along dimensionality
    reduction component axes. Each row corresponds to a component, each
    column to a standard deviation multiplier.

    The function uses a two-stage inverse transform pipeline:
    ``scores -> [reducer_inverse_transform] -> coefficients ->
    [descriptor_inverse_transform] -> shape coordinates``.

    Parameters
    ----------
    reducer : fitted estimator, optional
        Fitted dimensionality reduction object (e.g.,
        ``sklearn.decomposition.PCA``). Convenience parameter that extracts
        ``reducer_inverse_transform`` via ``.inverse_transform``,
        ``explained_variance`` via ``.explained_variance_``, and
        ``n_components`` via ``.n_components_`` (fallback to ``.n_components``).
    descriptor : fitted estimator, optional
        Fitted shape descriptor (e.g., ``EllipticFourierAnalysis``).
        Convenience parameter that extracts ``descriptor_inverse_transform``
        via ``.inverse_transform``. For GPA (landmarks): pass ``None``.
    descriptor_inverse_transform : callable, optional
        Converts coefficient vectors to shape coordinates. Overrides
        ``descriptor.inverse_transform``. For SHA resolution control, wrap
        with a lambda: ``lambda c: sha.inverse_transform(c,
        theta_range=..., phi_range=...)``.
    reducer_inverse_transform : callable, optional
        Converts low-dimensional scores to coefficient space. Overrides
        ``reducer.inverse_transform``.
    explained_variance : ndarray, optional
        Variance per component for SD calculation. Overrides
        ``reducer.explained_variance_``.
    n_components : int, optional
        Total number of components. Overrides ``reducer.n_components_``.
    components : sequence of int
        0-indexed component indices to display as rows.
    sd_values : sequence of float
        Standard deviation multipliers for columns.
    shape_type : str
        Shape rendering type. One of ``"auto"``, ``"curve_2d"``,
        ``"curve_3d"``, ``"surface_3d"``, ``"landmarks_2d"``,
        ``"landmarks_3d"``.
    render_fn : callable, optional
        Custom renderer ``(coords, ax, **kw) -> None``. Overrides
        ``shape_type``.
    n_dim : int, optional
        Spatial dimensionality (for reshape in GPA identity case). Required
        when ``descriptor`` is not provided, unless ``shape_type`` is an
        explicit landmarks type.
    links : sequence of sequence of int, optional
        Landmark link pairs (e.g., ``[[0, 1], [1, 2]]``).
    color : str
        Shape color.
    alpha : float
        Shape transparency.
    fig : matplotlib.figure.Figure, optional
        Existing figure. If ``None``, a new one is created.
    dpi : int
        Figure resolution (used only when creating a new figure).
    figscale : float
        Scale factor for figure size.
    **render_kw
        Forwarded to the renderer.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the shape grid.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If required parameters cannot be resolved.

    Examples
    --------
    >>> from sklearn.decomposition import PCA
    >>> from ktch.harmonic import EllipticFourierAnalysis
    >>> from ktch.plot import shape_variation_plot
    >>> efa = EllipticFourierAnalysis(n_harmonics=20)
    >>> coeffs = efa.fit_transform(outlines_2d)  # doctest: +SKIP
    >>> pca = PCA(n_components=10).fit(coeffs)  # doctest: +SKIP
    >>> fig = shape_variation_plot(pca, descriptor=efa)  # doctest: +SKIP
    """
    require_dependencies("matplotlib")
    import matplotlib.pyplot as plt

    # --- Resolve parameters ---
    reducer_inverse_transform, explained_variance, n_components = (
        _resolve_reducer_params(
            reducer,
            reducer_inverse_transform,
            explained_variance,
            n_components,
            require_variance=True,
        )
    )
    descriptor_inverse_transform, n_dim = _resolve_descriptor_params(
        descriptor,
        descriptor_inverse_transform,
        n_dim,
        shape_type,
    )
    _validate_components(components, n_components)

    # --- Auto-detect shape_type if needed ---
    if shape_type == "auto":
        probe_score = np.zeros((1, n_components))
        probe_coeffs = reducer_inverse_transform(probe_score)
        if descriptor_inverse_transform is not None:
            probe_coords = np.asarray(descriptor_inverse_transform(probe_coeffs))
            sample = probe_coords[0]
        else:
            sample = probe_coeffs[0].reshape(-1, n_dim)
        shape_type = _detect_shape_type(
            sample,
            descriptor_inverse_transform,
            n_dim,
        )

    renderer, proj = _get_renderer_and_projection(shape_type, render_fn)

    # --- Build figure ---
    n_rows = len(components)
    n_cols = len(sd_values)

    if fig is None:
        fig = plt.figure(
            figsize=(figscale * n_cols, figscale * n_rows),
            dpi=dpi,
        )

    axes_grid: list[list[Any]] = []
    resolved = _resolve_render_kw(
        render_kw,
        color=color,
        alpha=alpha,
        links=links,
    )

    for i, comp_idx in enumerate(components):
        sd = np.sqrt(explained_variance[comp_idx])
        row_axes = []
        for j, sd_val in enumerate(sd_values):
            score = np.zeros(n_components)
            score[comp_idx] = sd_val * sd

            coeffs = reducer_inverse_transform(score.reshape(1, -1))

            if descriptor_inverse_transform is not None:
                coords = np.asarray(descriptor_inverse_transform(coeffs))
                single = coords[0]
            else:
                single = coeffs[0].reshape(-1, n_dim)

            ax = fig.add_subplot(
                n_rows,
                n_cols,
                n_cols * i + j + 1,
                projection=proj,
            )
            row_axes.append(ax)
            renderer(single, ax, **resolved)
            ax.axis("off")

        axes_grid.append(row_axes)

    # Row labels on leftmost subplots
    for i, comp_idx in enumerate(components):
        axes_grid[i][0].set_ylabel(f"PC{comp_idx + 1}")

    # Column labels on topmost subplots
    for j, sd_val in enumerate(sd_values):
        axes_grid[0][j].set_title(f"{sd_val:+g} SD")

    return fig
