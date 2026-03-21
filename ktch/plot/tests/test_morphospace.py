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

from ktch.plot import confidence_ellipse_plot, convex_hull_plot, morphospace_plot


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


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
    def test_curve_2d_auto(self, pca_data, mock_descriptor_2d):
        pca, df = pca_data
        ax = morphospace_plot(
            data=df,
            x="PC1",
            y="PC2",
            reducer=pca,
            descriptor=mock_descriptor_2d,
            n_shapes=2,
        )
        assert isinstance(ax, mpl.axes.Axes)
        # Should have inset axes (2*2 = 4 insets + 1 main)
        assert len(ax.figure.axes) > 1

    def test_surface_3d_explicit(self, pca_data, mock_descriptor_3d_surface):
        pca, df = pca_data
        ax = morphospace_plot(
            data=df,
            x="PC1",
            y="PC2",
            reducer=pca,
            descriptor=mock_descriptor_3d_surface,
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

    def test_explicit_reducer_inverse_transform(self, pca_data, mock_descriptor_2d):
        pca, df = pca_data
        ax = morphospace_plot(
            data=df,
            x="PC1",
            y="PC2",
            reducer_inverse_transform=pca.inverse_transform,
            n_components=pca.n_components_,
            descriptor=mock_descriptor_2d,
            n_shapes=2,
        )
        assert isinstance(ax, mpl.axes.Axes)

    def test_no_scatter_shape_overlay_only(self, pca_data, mock_descriptor_2d):
        pca, _ = pca_data
        fig, ax_in = plt.subplots()
        ax_out = morphospace_plot(
            reducer=pca,
            descriptor=mock_descriptor_2d,
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

    def test_component_index_out_of_range(self, pca_data, mock_descriptor_2d):
        pca, df = pca_data
        with pytest.raises(ValueError, match="Component index"):
            morphospace_plot(
                data=df,
                x="PC1",
                y="PC2",
                reducer=pca,
                descriptor=mock_descriptor_2d,
                components=(0, 10),
                n_shapes=2,
            )

    def test_n_shapes_zero_raises(self, pca_data, mock_descriptor_2d):
        pca, df = pca_data
        with pytest.raises(ValueError, match="n_shapes must be >= 1"):
            morphospace_plot(
                data=df,
                x="PC1",
                y="PC2",
                reducer=pca,
                descriptor=mock_descriptor_2d,
                n_shapes=0,
            )

    def test_n_shapes_negative_raises(self, pca_data, mock_descriptor_2d):
        pca, df = pca_data
        with pytest.raises(ValueError, match="n_shapes must be >= 1"):
            morphospace_plot(
                data=df,
                x="PC1",
                y="PC2",
                reducer=pca,
                descriptor=mock_descriptor_2d,
                n_shapes=-1,
            )


#
# confidence_ellipse_plot
#


class TestConfidenceEllipsePlot:
    def test_returns_axes_no_hue(self):
        rng = np.random.default_rng(42)
        x, y = rng.standard_normal(20), rng.standard_normal(20)
        ax = confidence_ellipse_plot(x=x, y=y)
        assert isinstance(ax, mpl.axes.Axes)

    def test_returns_axes_with_hue(self, pca_data):
        _, df = pca_data
        ax = confidence_ellipse_plot(data=df, x="PC1", y="PC2", hue="group")
        assert isinstance(ax, mpl.axes.Axes)

    def test_with_existing_axes(self, pca_data):
        _, df = pca_data
        fig, ax_in = plt.subplots()
        ax_out = confidence_ellipse_plot(data=df, x="PC1", y="PC2", ax=ax_in)
        assert ax_out is ax_in

    def test_adds_patches_per_group(self, pca_data):
        _, df = pca_data
        fig, ax = plt.subplots()
        confidence_ellipse_plot(data=df, x="PC1", y="PC2", hue="group", ax=ax)
        n_groups = df["group"].nunique()
        assert len(ax.patches) == n_groups

    def test_single_ellipse_no_hue(self):
        rng = np.random.default_rng(42)
        x, y = rng.standard_normal(20), rng.standard_normal(20)
        fig, ax = plt.subplots()
        confidence_ellipse_plot(x=x, y=y, ax=ax)
        assert len(ax.patches) == 1

    def test_fill(self, pca_data):
        _, df = pca_data
        ax = confidence_ellipse_plot(data=df, x="PC1", y="PC2", hue="group", fill=True)
        assert isinstance(ax, mpl.axes.Axes)

    def test_warns_few_points(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        hue = np.array(["A", "B", "B"])
        with pytest.warns(UserWarning, match="at least 2"):
            confidence_ellipse_plot(x=x, y=y, hue=hue)

    def test_confidence_parameter(self, pca_data):
        _, df = pca_data
        ax = confidence_ellipse_plot(
            data=df, x="PC1", y="PC2", hue="group", confidence=0.99
        )
        assert isinstance(ax, mpl.axes.Axes)

    def test_n_std_overrides_confidence(self, pca_data):
        _, df = pca_data
        ax = confidence_ellipse_plot(data=df, x="PC1", y="PC2", hue="group", n_std=1.0)
        assert isinstance(ax, mpl.axes.Axes)

    def test_invalid_confidence_raises(self):
        rng = np.random.default_rng(42)
        x, y = rng.standard_normal(20), rng.standard_normal(20)
        with pytest.raises(ValueError, match="open interval"):
            confidence_ellipse_plot(x=x, y=y, confidence=1.5)

    def test_legend_false_no_labels(self, pca_data):
        _, df = pca_data
        fig, ax = plt.subplots()
        confidence_ellipse_plot(
            data=df, x="PC1", y="PC2", hue="group", legend=False, ax=ax
        )
        labels = [p.get_label() for p in ax.patches]
        assert all(lab is None or lab.startswith("_") or lab == "" for lab in labels)

    def test_categorical_hue_preserves_order(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.standard_normal(30), "y": rng.standard_normal(30)})
        cat_order = ["C", "A", "B"]
        df["grp"] = pd.Categorical(
            rng.choice(cat_order, 30), categories=cat_order, ordered=True
        )
        fig, ax = plt.subplots()
        confidence_ellipse_plot(
            data=df, x="x", y="y", hue="grp", palette="tab10", ax=ax
        )
        import seaborn as sns

        expected_colors = sns.color_palette("tab10", n_colors=3)
        labels = [p.get_label() for p in ax.patches]
        colors = [p.get_edgecolor() for p in ax.patches]
        for lab, col in zip(labels, colors):
            idx = cat_order.index(lab)
            assert np.allclose(col[:3], expected_colors[idx], atol=1e-2)


#
# convex_hull_plot
#


class TestConvexHullPlot:
    def test_returns_axes_no_hue(self):
        rng = np.random.default_rng(42)
        ax = convex_hull_plot(x=rng.standard_normal(20), y=rng.standard_normal(20))
        assert isinstance(ax, mpl.axes.Axes)

    def test_returns_axes_with_hue(self, pca_data):
        _, df = pca_data
        ax = convex_hull_plot(data=df, x="PC1", y="PC2", hue="group")
        assert isinstance(ax, mpl.axes.Axes)

    def test_with_existing_axes(self, pca_data):
        _, df = pca_data
        fig, ax_in = plt.subplots()
        ax_out = convex_hull_plot(data=df, x="PC1", y="PC2", ax=ax_in)
        assert ax_out is ax_in

    def test_draws_lines_no_fill(self):
        rng = np.random.default_rng(42)
        fig, ax = plt.subplots()
        convex_hull_plot(x=rng.standard_normal(20), y=rng.standard_normal(20), ax=ax)
        assert len(ax.lines) > 0

    def test_fill_adds_patches(self, pca_data):
        _, df = pca_data
        fig, ax = plt.subplots()
        convex_hull_plot(data=df, x="PC1", y="PC2", hue="group", fill=True, ax=ax)
        n_groups = df["group"].nunique()
        assert len(ax.patches) == n_groups

    def test_warns_few_points(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        hue = np.array(["A", "A", "B", "B"])
        with pytest.warns(UserWarning, match="at least 3"):
            convex_hull_plot(x=x, y=y, hue=hue)

    def test_legend_false_no_labels(self, pca_data):
        _, df = pca_data
        fig, ax = plt.subplots()
        convex_hull_plot(
            data=df, x="PC1", y="PC2", hue="group", fill=True, legend=False, ax=ax
        )
        labels = [p.get_label() for p in ax.patches]
        assert all(lab is None or lab.startswith("_") or lab == "" for lab in labels)
