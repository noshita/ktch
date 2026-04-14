"""Unit tests for plot._configuration module."""

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
import matplotlib.pyplot as plt

pytest.importorskip("seaborn")

from ktch.plot import configuration_plot


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _make_df_2d(n_specimens=3, n_landmarks=5, seed=42):
    """Create a 2D landmark DataFrame with MultiIndex."""
    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(n_specimens):
        for lid in range(n_landmarks):
            rows.append((sid, lid, rng.standard_normal(), rng.standard_normal()))
    df = pd.DataFrame(rows, columns=["specimen_id", "coord_id", "x", "y"])
    df = df.set_index(["specimen_id", "coord_id"])
    return df


def _make_df_3d(n_specimens=3, n_landmarks=5, seed=42):
    """Create a 3D landmark DataFrame with MultiIndex."""
    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(n_specimens):
        for lid in range(n_landmarks):
            rows.append(
                (
                    sid,
                    lid,
                    rng.standard_normal(),
                    rng.standard_normal(),
                    rng.standard_normal(),
                )
            )
    df = pd.DataFrame(rows, columns=["specimen_id", "coord_id", "x", "y", "z"])
    df = df.set_index(["specimen_id", "coord_id"])
    return df


class TestConfigurationPlot2D:
    def test_single_specimen(self):
        df = _make_df_2d(n_specimens=1)
        ax = configuration_plot(df.loc[0])
        assert ax is not None

    def test_single_specimen_with_links(self):
        df = _make_df_2d(n_specimens=1)
        links = [[0, 1], [1, 2], [2, 3], [3, 4]]
        ax = configuration_plot(df.loc[0], links=links, alpha=0.5, s=20)
        assert ax is not None

    def test_multiple_specimens_hue(self):
        df = _make_df_2d()
        links = [[0, 1], [1, 2]]
        ax = configuration_plot(df, links=links, hue="specimen_id", alpha=0.3)
        assert ax is not None

    def test_no_links(self):
        df = _make_df_2d()
        ax = configuration_plot(df, hue="specimen_id")
        assert ax is not None

    def test_palette(self):
        df = _make_df_2d()
        ax = configuration_plot(df, hue="specimen_id", palette="Set2")
        assert ax is not None

    def test_style(self):
        df = _make_df_2d()
        ax = configuration_plot(df, hue="specimen_id", style="coord_id")
        assert ax is not None

    def test_hue_order(self):
        df = _make_df_2d()
        ax = configuration_plot(df, hue="specimen_id", hue_order=[2, 0, 1])
        assert ax is not None

    def test_color_links(self):
        df = _make_df_2d(n_specimens=1)
        links = [[0, 1], [1, 2]]
        ax = configuration_plot(df.loc[0], links=links, color="blue", color_links="red")
        assert ax is not None

    def test_returns_ax(self):
        df = _make_df_2d(n_specimens=1)
        result = configuration_plot(df.loc[0])
        assert isinstance(result, mpl.axes.Axes)

    def test_ax_passthrough(self):
        df = _make_df_2d(n_specimens=1)
        fig, ax = plt.subplots()
        returned = configuration_plot(df.loc[0], ax=ax)
        assert returned is ax

    def test_aspect_equal(self):
        df = _make_df_2d(n_specimens=1)
        ax = configuration_plot(df.loc[0])
        assert ax.get_aspect() in ("equal", 1.0)


class TestConfigurationPlot3D:
    def test_basic(self):
        df = _make_df_3d(n_specimens=1)
        ax = configuration_plot(df.loc[0], z="z")
        assert ax is not None
        assert ax.name == "3d"

    def test_with_links(self):
        df = _make_df_3d(n_specimens=1)
        links = [[0, 1], [1, 2], [2, 3]]
        ax = configuration_plot(df.loc[0], links=links, z="z")
        assert ax is not None

    def test_hue(self):
        df = _make_df_3d()
        ax = configuration_plot(df, z="z", hue="specimen_id", alpha=0.5)
        assert ax is not None

    def test_hue_with_links(self):
        df = _make_df_3d()
        links = [[0, 1], [1, 2]]
        ax = configuration_plot(df, z="z", links=links, hue="specimen_id", alpha=0.3)
        assert ax is not None

    def test_ax_creation(self):
        df = _make_df_3d(n_specimens=1)
        ax = configuration_plot(df.loc[0], z="z")
        assert ax.name == "3d"

    def test_ax_passthrough_3d(self):
        df = _make_df_3d(n_specimens=1)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        returned = configuration_plot(df.loc[0], z="z", ax=ax)
        assert returned is ax


class TestNumpyArrayInput:
    def test_2d_single_specimen(self):
        arr = np.random.default_rng(42).standard_normal((5, 2))
        ax = configuration_plot(arr)
        assert ax is not None

    def test_2d_with_links(self):
        arr = np.random.default_rng(42).standard_normal((5, 2))
        links = [[0, 1], [1, 2], [2, 3]]
        ax = configuration_plot(arr, links=links)
        assert ax is not None

    def test_3d_multiple_specimens(self):
        arr = np.random.default_rng(42).standard_normal((3, 5, 2))
        ax = configuration_plot(arr, hue="specimen_id", alpha=0.3)
        assert ax is not None

    def test_3d_multiple_specimens_with_links(self):
        arr = np.random.default_rng(42).standard_normal((3, 5, 2))
        links = [[0, 1], [1, 2]]
        ax = configuration_plot(arr, links=links, hue="specimen_id")
        assert ax is not None

    def test_3d_array_for_3d_plot(self):
        arr = np.random.default_rng(42).standard_normal((5, 3))
        ax = configuration_plot(arr, z="z")
        assert ax.name == "3d"

    def test_3d_array_multi_specimen_3d_plot(self):
        arr = np.random.default_rng(42).standard_normal((3, 5, 3))
        ax = configuration_plot(arr, z="z", hue="specimen_id")
        assert ax is not None

    def test_1d_array_raises(self):
        arr = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="2D or 3D"):
            configuration_plot(arr)

    def test_insufficient_columns_2d(self):
        arr = np.random.default_rng(42).standard_normal((5, 1))
        with pytest.raises(ValueError, match="at least 2 columns"):
            configuration_plot(arr)

    def test_insufficient_columns_3d_plot(self):
        arr = np.random.default_rng(42).standard_normal((5, 2))
        with pytest.raises(ValueError, match="at least 3 columns"):
            configuration_plot(arr, z="z")


class TestUnnamedIndex:
    def test_unnamed_index_no_links(self):
        df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 0]})
        ax = configuration_plot(df)
        assert ax is not None

    def test_unnamed_index_with_links(self):
        df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 0]})
        links = [[0, 1], [1, 2]]
        ax = configuration_plot(df, links=links)
        assert ax is not None


class TestPartialDictPalette:
    def test_complete_dict_palette(self):
        df = _make_df_2d()
        palette = {0: "red", 1: "green", 2: "blue"}
        ax = configuration_plot(df, hue="specimen_id", palette=palette)
        assert ax is not None

    def test_partial_dict_palette_raises(self):
        df = _make_df_2d()
        palette = {0: "red"}  # missing 1 and 2
        with pytest.raises(ValueError, match="missing keys"):
            configuration_plot(df, hue="specimen_id", palette=palette)


class TestAxesDimensionValidation:
    def test_2d_ax_with_z_raises(self):
        df = _make_df_3d(n_specimens=1)
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="not a 3D axes"):
            configuration_plot(df.loc[0], z="z", ax=ax)

    def test_3d_ax_without_z_raises(self):
        df = _make_df_2d(n_specimens=1)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        with pytest.raises(ValueError, match="not specified but ax is a 3D"):
            configuration_plot(df.loc[0], ax=ax)
