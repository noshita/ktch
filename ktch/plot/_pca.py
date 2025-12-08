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

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def explained_variance_ratio_plot(pca, n_components=None, ax=None, verbose=False):
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

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    if n_components is None:
        n_components = pca.n_components_

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
    descriptor_inverse_transform,
    pca,
    n_dim=2,
    n_PCs=(0, 1, 2),
    sd_values=(-2, -1, 0, 1, 2),
    morph_color="gray",
    morph_alpha=1.0,
    fig=None,
    dpi=150,
    figscale=3.0,
):
    """Plot reconstructed shapes along principal components.

    This function is a placeholder for the actual implementation.
    """
    pass
