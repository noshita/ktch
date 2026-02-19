"""End-to-end tests for plot._pca module."""

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

import warnings

import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
pytest.importorskip("seaborn")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from ktch.plot import explained_variance_ratio_plot
from ktch.plot._pca import plot_shapes_along_pcs


@pytest.fixture
def fitted_pca():
    """A PCA object fitted on random data with 5 components."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 5))
    pca = PCA(n_components=5)
    pca.fit(X)
    return pca


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


class TestExplainedVarianceRatioPlot:
    """End-to-end tests for explained_variance_ratio_plot."""

    def test_returns_axes(self, fitted_pca):
        """Test that the function returns a matplotlib Axes object."""
        ax = explained_variance_ratio_plot(fitted_pca)
        assert isinstance(ax, mpl.axes.Axes)

    def test_with_existing_axes(self, fitted_pca):
        """Test that the function draws on a provided axes."""
        fig, ax_in = plt.subplots()
        ax_out = explained_variance_ratio_plot(fitted_pca, ax=ax_in)
        assert ax_out is ax_in

    def test_creates_new_figure_when_no_axes(self, fitted_pca):
        """Test that a new figure is created when ax is not provided."""
        plt.close("all")
        explained_variance_ratio_plot(fitted_pca)
        assert len(plt.get_fignums()) == 1

    def test_default_n_components(self, fitted_pca):
        """Test that all components are plotted by default."""
        ax = explained_variance_ratio_plot(fitted_pca)
        # barplot creates one patch per component
        n_bars = len(ax.patches)
        assert n_bars == fitted_pca.n_components_

    def test_explicit_n_components(self, fitted_pca):
        """Test with a subset of components."""
        ax = explained_variance_ratio_plot(fitted_pca, n_components=3)
        n_bars = len(ax.patches)
        assert n_bars == 3

    def test_n_components_one(self, fitted_pca):
        """Test with a single component."""
        ax = explained_variance_ratio_plot(fitted_pca, n_components=1)
        n_bars = len(ax.patches)
        assert n_bars == 1

    def test_n_components_exceeds_fitted_raises_error(self, fitted_pca):
        """Test that n_components > fitted components raises ValueError."""
        with pytest.raises(ValueError, match="exceeds"):
            explained_variance_ratio_plot(fitted_pca, n_components=10)

    def test_verbose_output(self, fitted_pca, capsys):
        """Test that verbose mode prints variance information."""
        explained_variance_ratio_plot(fitted_pca, verbose=True)
        captured = capsys.readouterr()
        assert "Explained variance ratio:" in captured.out
        assert "Cumsum of Explained variance ratio:" in captured.out

    def test_axes_has_plot_elements(self, fitted_pca):
        """Test that the axes contains bars, lines, and scatter points."""
        ax = explained_variance_ratio_plot(fitted_pca)
        assert len(ax.patches) > 0, "Expected bar patches"
        assert len(ax.lines) > 0, "Expected line plot"
        assert len(ax.collections) > 0, "Expected scatter points"


class TestDeprecatedAlias:
    """Test the deprecated plot_explained_variance_ratio alias."""

    def test_deprecated_alias_works(self, fitted_pca):
        """Test that the old name still works with a DeprecationWarning."""
        from ktch import plot

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ax = plot.plot_explained_variance_ratio(fitted_pca)

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "renamed" in str(w[0].message)
        assert isinstance(ax, mpl.axes.Axes)


class TestPlotShapesAlongPcs:
    """Test the plot_shapes_along_pcs placeholder."""

    def test_raises_not_implemented(self):
        """Test that the placeholder raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            plot_shapes_along_pcs(None, None)
