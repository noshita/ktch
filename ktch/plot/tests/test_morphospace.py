"""End-to-end tests for plot._morphospace module."""

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

import numpy as np
import pandas as pd
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
pytest.importorskip("seaborn")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from ktch.plot import morphospace_plot


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class _MockDescriptor2D:
    """Mock 2D curve descriptor (EFA-like)."""

    def inverse_transform(self, X):
        rng = np.random.default_rng(42)
        n = X.shape[0]
        t = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        coords = np.zeros((n, 50, 2))
        for i in range(n):
            coords[i, :, 0] = np.cos(t) + rng.normal(0, 0.01, 50)
            coords[i, :, 1] = np.sin(t) + rng.normal(0, 0.01, 50)
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


@pytest.fixture
def pca_data():
    """Return (fitted PCA, DataFrame with PC scores)."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 10))
    pca = PCA(n_components=5).fit(X)
    scores = pca.transform(X)
    df = pd.DataFrame(
        {f"PC{i + 1}": scores[:, i] for i in range(5)},
    )
    df["group"] = rng.choice(["A", "B", "C"], 30)
    return pca, df


class TestMorphospacePlotScatterOnly:
    def test_returns_axes(self, pca_data):
        pca, df = pca_data
        ax = morphospace_plot(data=df, x="PC1", y="PC2", hue="group")
        assert isinstance(ax, mpl.axes.Axes)

    def test_with_existing_axes(self, pca_data):
        pca, df = pca_data
        fig, ax_in = plt.subplots()
        ax_out = morphospace_plot(
            data=df,
            x="PC1",
            y="PC2",
            ax=ax_in,
        )
        assert ax_out is ax_in


class TestMorphospacePlotWithShapes:
    def test_curve_2d_auto(self, pca_data):
        pca, df = pca_data
        desc = _MockDescriptor2D()
        ax = morphospace_plot(
            data=df,
            x="PC1",
            y="PC2",
            reducer=pca,
            descriptor=desc,
            n_shapes=2,
        )
        assert isinstance(ax, mpl.axes.Axes)
        # Should have inset axes (2*2 = 4 insets + 1 main)
        assert len(ax.figure.axes) > 1

    def test_surface_3d_explicit(self, pca_data):
        pca, df = pca_data
        desc = _MockDescriptor3DSurface()
        ax = morphospace_plot(
            data=df,
            x="PC1",
            y="PC2",
            reducer=pca,
            descriptor=desc,
            shape_type="surface_3d",
            n_shapes=2,
        )
        assert isinstance(ax, mpl.axes.Axes)

    def test_landmarks_2d(self, pca_data):
        pca, df = pca_data
        ax = morphospace_plot(
            data=df,
            x="PC1",
            y="PC2",
            reducer=pca,
            n_dim=2,
            shape_type="landmarks_2d",
            n_shapes=2,
        )
        assert isinstance(ax, mpl.axes.Axes)

    def test_explicit_reducer_inverse_transform(self, pca_data):
        pca, df = pca_data
        desc = _MockDescriptor2D()
        ax = morphospace_plot(
            data=df,
            x="PC1",
            y="PC2",
            reducer_inverse_transform=pca.inverse_transform,
            n_components=pca.n_components_,
            descriptor=desc,
            n_shapes=2,
        )
        assert isinstance(ax, mpl.axes.Axes)

    def test_no_scatter_shape_overlay_only(self, pca_data):
        pca, _ = pca_data
        desc = _MockDescriptor2D()
        fig, ax_in = plt.subplots()
        ax_out = morphospace_plot(
            reducer=pca,
            descriptor=desc,
            n_shapes=2,
            ax=ax_in,
        )
        assert ax_out is ax_in


class TestMorphospacePlotErrors:
    def test_missing_n_dim_for_identity(self, pca_data):
        pca, df = pca_data
        with pytest.raises(ValueError, match="n_dim is required"):
            morphospace_plot(
                data=df,
                x="PC1",
                y="PC2",
                reducer=pca,
                n_shapes=2,
            )

    def test_component_index_out_of_range(self, pca_data):
        pca, df = pca_data
        desc = _MockDescriptor2D()
        with pytest.raises(ValueError, match="Component index"):
            morphospace_plot(
                data=df,
                x="PC1",
                y="PC2",
                reducer=pca,
                descriptor=desc,
                components=(0, 10),
                n_shapes=2,
            )
