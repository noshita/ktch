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

from ktch.plot import explained_variance_ratio_plot, shape_variation_plot


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


# ---------------------------------------------------------------------------
# Mock descriptors for shape_variation_plot tests
# ---------------------------------------------------------------------------


class _MockDescriptor2D:
    """Mock 2D curve descriptor (EFA-like)."""

    def inverse_transform(self, X):
        n = X.shape[0]
        t = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        coords = np.zeros((n, 50, 2))
        for i in range(n):
            coords[i, :, 0] = np.cos(t)
            coords[i, :, 1] = np.sin(t)
        return coords


class _MockDescriptor3DCurve:
    """Mock 3D curve descriptor."""

    def inverse_transform(self, X):
        n = X.shape[0]
        t = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        coords = np.zeros((n, 50, 3))
        for i in range(n):
            coords[i, :, 0] = np.cos(t)
            coords[i, :, 1] = np.sin(t)
            coords[i, :, 2] = t / (2 * np.pi)
        return coords


class _MockDescriptor3DSurface:
    """Mock 3D surface descriptor (SHA-like)."""

    def inverse_transform(self, X):
        n = X.shape[0]
        m, k = 10, 20
        theta = np.linspace(0, np.pi, m)
        phi = np.linspace(0, 2 * np.pi, k)
        T, P = np.meshgrid(theta, phi, indexing="ij")
        coords = np.zeros((n, m, k, 3))
        for i in range(n):
            coords[i, :, :, 0] = np.sin(T) * np.cos(P)
            coords[i, :, :, 1] = np.sin(T) * np.sin(P)
            coords[i, :, :, 2] = np.cos(T)
        return coords


# ===========================================================================
# explained_variance_ratio_plot
# ===========================================================================


class TestExplainedVarianceRatioPlot:
    """End-to-end tests for explained_variance_ratio_plot."""

    def test_returns_axes(self, fitted_pca):
        ax = explained_variance_ratio_plot(fitted_pca)
        assert isinstance(ax, mpl.axes.Axes)

    def test_with_existing_axes(self, fitted_pca):
        fig, ax_in = plt.subplots()
        ax_out = explained_variance_ratio_plot(fitted_pca, ax=ax_in)
        assert ax_out is ax_in

    def test_creates_new_figure_when_no_axes(self, fitted_pca):
        plt.close("all")
        explained_variance_ratio_plot(fitted_pca)
        assert len(plt.get_fignums()) == 1

    def test_default_n_components(self, fitted_pca):
        ax = explained_variance_ratio_plot(fitted_pca)
        n_bars = len(ax.patches)
        assert n_bars == fitted_pca.n_components_

    def test_explicit_n_components(self, fitted_pca):
        ax = explained_variance_ratio_plot(fitted_pca, n_components=3)
        n_bars = len(ax.patches)
        assert n_bars == 3

    def test_n_components_one(self, fitted_pca):
        ax = explained_variance_ratio_plot(fitted_pca, n_components=1)
        n_bars = len(ax.patches)
        assert n_bars == 1

    def test_n_components_exceeds_fitted_raises_error(self, fitted_pca):
        with pytest.raises(ValueError, match="exceeds"):
            explained_variance_ratio_plot(fitted_pca, n_components=10)

    def test_verbose_output(self, fitted_pca, capsys):
        explained_variance_ratio_plot(fitted_pca, verbose=True)
        captured = capsys.readouterr()
        assert "EVR" in captured.out
        assert "Cumsum" in captured.out
        assert "PC1" in captured.out

    def test_axes_has_plot_elements(self, fitted_pca):
        ax = explained_variance_ratio_plot(fitted_pca)
        assert len(ax.patches) > 0, "Expected bar patches"
        assert len(ax.lines) > 0, "Expected line plot"
        assert len(ax.collections) > 0, "Expected scatter points"


# ===========================================================================
# Deprecated aliases
# ===========================================================================


class TestDeprecatedAlias:
    def test_plot_explained_variance_ratio(self, fitted_pca):
        from ktch import plot

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ax = plot.plot_explained_variance_ratio(fitted_pca)

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "renamed" in str(w[0].message)
        assert isinstance(ax, mpl.axes.Axes)


# ===========================================================================
# shape_variation_plot
# ===========================================================================


class TestShapeVariationPlotSmoke:
    """Smoke tests: each shape_type completes without error."""

    def test_curve_2d_auto(self, fitted_pca):
        desc = _MockDescriptor2D()
        fig = shape_variation_plot(
            fitted_pca,
            descriptor=desc,
            components=(0, 1),
        )
        assert isinstance(fig, mpl.figure.Figure)

    def test_curve_2d_explicit(self, fitted_pca):
        desc = _MockDescriptor2D()
        fig = shape_variation_plot(
            fitted_pca,
            descriptor=desc,
            shape_type="curve_2d",
            components=(0,),
        )
        assert isinstance(fig, mpl.figure.Figure)

    def test_curve_3d(self, fitted_pca):
        desc = _MockDescriptor3DCurve()
        fig = shape_variation_plot(
            fitted_pca,
            descriptor=desc,
            shape_type="curve_3d",
            components=(0,),
        )
        assert isinstance(fig, mpl.figure.Figure)

    def test_surface_3d(self, fitted_pca):
        desc = _MockDescriptor3DSurface()
        fig = shape_variation_plot(
            fitted_pca,
            descriptor=desc,
            shape_type="surface_3d",
            components=(0,),
        )
        assert isinstance(fig, mpl.figure.Figure)

    def test_landmarks_2d(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 10))  # 5 landmarks * 2 dims
        pca = PCA(n_components=5).fit(X)
        fig = shape_variation_plot(
            pca,
            n_dim=2,
            shape_type="landmarks_2d",
            components=(0,),
        )
        assert isinstance(fig, mpl.figure.Figure)

    def test_landmarks_2d_with_links(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 10))  # 5 landmarks * 2 dims
        pca = PCA(n_components=5).fit(X)
        fig = shape_variation_plot(
            pca,
            n_dim=2,
            shape_type="landmarks_2d",
            links=[[0, 1], [1, 2]],
            components=(0,),
        )
        assert isinstance(fig, mpl.figure.Figure)

    def test_landmarks_3d(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 12))  # 4 landmarks * 3 dims
        pca = PCA(n_components=5).fit(X)
        fig = shape_variation_plot(
            pca,
            n_dim=3,
            shape_type="landmarks_3d",
            components=(0,),
        )
        assert isinstance(fig, mpl.figure.Figure)


class TestShapeVariationPlotStructure:
    def test_subplot_count(self, fitted_pca):
        desc = _MockDescriptor2D()
        components = (0, 1)
        sd_values = (-1.0, 0.0, 1.0)
        fig = shape_variation_plot(
            fitted_pca,
            descriptor=desc,
            components=components,
            sd_values=sd_values,
        )
        expected = len(components) * len(sd_values)
        assert len(fig.axes) == expected

    def test_returns_provided_figure(self, fitted_pca):
        desc = _MockDescriptor2D()
        fig_in = plt.figure()
        fig_out = shape_variation_plot(
            fitted_pca,
            descriptor=desc,
            components=(0,),
            fig=fig_in,
        )
        assert fig_out is fig_in

    def test_3d_projection_for_surface(self, fitted_pca):
        desc = _MockDescriptor3DSurface()
        fig = shape_variation_plot(
            fitted_pca,
            descriptor=desc,
            shape_type="surface_3d",
            components=(0,),
            sd_values=(0.0,),
        )
        assert fig.axes[0].name == "3d"


class TestShapeVariationPlotOverrides:
    def test_explicit_reducer_overrides(self, fitted_pca):
        desc = _MockDescriptor2D()
        fig = shape_variation_plot(
            reducer_inverse_transform=fitted_pca.inverse_transform,
            explained_variance=fitted_pca.explained_variance_,
            n_components=fitted_pca.n_components_,
            descriptor=desc,
            components=(0,),
        )
        assert isinstance(fig, mpl.figure.Figure)

    def test_custom_render_fn(self, fitted_pca):
        desc = _MockDescriptor2D()

        def my_renderer(coords, ax, **kw):
            kw.pop("links", None)
            ax.fill(coords[:, 0], coords[:, 1], color=kw.get("color", "gray"))

        fig = shape_variation_plot(
            fitted_pca,
            descriptor=desc,
            render_fn=my_renderer,
            components=(0,),
        )
        assert isinstance(fig, mpl.figure.Figure)

    def test_descriptor_inverse_transform_override(self, fitted_pca):
        desc = _MockDescriptor2D()
        fig = shape_variation_plot(
            fitted_pca,
            descriptor_inverse_transform=desc.inverse_transform,
            components=(0,),
        )
        assert isinstance(fig, mpl.figure.Figure)


class TestShapeVariationPlotErrors:
    def test_missing_reducer(self):
        with pytest.raises(ValueError, match="reducer"):
            shape_variation_plot(components=(0,))

    def test_missing_explained_variance(self):
        with pytest.raises(ValueError, match="explained_variance"):
            shape_variation_plot(
                reducer_inverse_transform=lambda x: x,
                n_components=5,
                components=(0,),
            )

    def test_missing_n_dim_for_identity(self, fitted_pca):
        with pytest.raises(ValueError, match="n_dim is required"):
            shape_variation_plot(fitted_pca, components=(0,))

    def test_component_out_of_range(self, fitted_pca):
        desc = _MockDescriptor2D()
        with pytest.raises(ValueError, match="Component index"):
            shape_variation_plot(
                fitted_pca,
                descriptor=desc,
                components=(0, 10),
            )

    def test_invalid_shape_type(self, fitted_pca):
        desc = _MockDescriptor2D()
        with pytest.raises(ValueError, match="Invalid shape_type"):
            shape_variation_plot(
                fitted_pca,
                descriptor=desc,
                shape_type="invalid",
                components=(0,),
            )


class TestShapeVariationPlotWarnings:
    def test_render_kw_conflict(self):
        from ktch.plot._renderers import _resolve_render_kw

        with pytest.warns(UserWarning, match="render_kw ignored"):
            _resolve_render_kw(
                {"color": "blue"},
                color="red",
                alpha=1.0,
                links=None,
            )
