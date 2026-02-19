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

from collections.abc import Sequence
from typing import Any

import numpy as np

from ._base import require_dependencies


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


def plot_shapes_along_pcs(
    descriptor_inverse_transform: Any,
    pca: Any,
    n_dim: int = 2,
    n_pcs: Sequence[int] = (0, 1, 2),
    sd_values: Sequence[float] = (-2, -1, 0, 1, 2),
    morph_color: str = "gray",
    morph_alpha: float = 1.0,
    fig: object | None = None,
    dpi: int = 150,
    figscale: float = 3.0,
) -> None:
    """Plot reconstructed shapes along principal components.

    This function is a placeholder for the actual implementation.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.
    """
    raise NotImplementedError("plot_shapes_along_pcs is not yet implemented.")
