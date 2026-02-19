"""End-to-end tests for plot._kriging module."""

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
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
import matplotlib.pyplot as plt

from ktch.plot import tps_grid_2d_plot


@pytest.fixture
def triangle_configs():
    """Reference and slightly deformed triangle configurations."""
    x_reference = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    x_target = np.array([[0.05, 0.0], [1.0, 0.05], [0.5, 0.95]])
    return x_reference, x_target


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


class TestTpsGrid2dPlot:
    """End-to-end tests for tps_grid_2d_plot."""

    def test_returns_axes(self, triangle_configs):
        """Test that the function returns a matplotlib Axes object."""
        x_ref, x_tgt = triangle_configs
        ax = tps_grid_2d_plot(x_ref, x_tgt)
        assert isinstance(ax, mpl.axes.Axes)

    def test_with_existing_axes(self, triangle_configs):
        """Test that the function draws on a provided axes."""
        x_ref, x_tgt = triangle_configs
        fig, ax_in = plt.subplots()
        ax_out = tps_grid_2d_plot(x_ref, x_tgt, ax=ax_in)
        assert ax_out is ax_in

    def test_creates_new_figure_when_no_axes(self, triangle_configs):
        """Test that a new figure is created when ax is not provided."""
        plt.close("all")
        x_ref, x_tgt = triangle_configs
        tps_grid_2d_plot(x_ref, x_tgt)
        assert len(plt.get_fignums()) == 1

    def test_axes_has_plot_elements(self, triangle_configs):
        """Test that the axes contains grid lines and scatter points."""
        x_ref, x_tgt = triangle_configs
        ax = tps_grid_2d_plot(x_ref, x_tgt)

        # Should have Line2D objects (grid lines) and PathCollection (scatter)
        assert len(ax.lines) > 0, "Expected grid lines"
        assert len(ax.collections) > 0, "Expected scatter points"

    def test_explicit_grid_size(self, triangle_configs):
        """Test with an explicit numeric grid_size."""
        x_ref, x_tgt = triangle_configs
        ax = tps_grid_2d_plot(x_ref, x_tgt, grid_size=0.2)
        assert isinstance(ax, mpl.axes.Axes)

    def test_custom_n_grid_inner(self, triangle_configs):
        """Test with a custom n_grid_inner value."""
        x_ref, x_tgt = triangle_configs
        ax = tps_grid_2d_plot(x_ref, x_tgt, n_grid_inner=5)
        assert isinstance(ax, mpl.axes.Axes)

    def test_square_bounding_box(self):
        """Test with a square landmark configuration (w == h path)."""
        x_ref = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        x_tgt = np.array([[0.05, 0.05], [0.95, 0.0], [1.0, 0.95], [0.0, 1.0]])
        ax = tps_grid_2d_plot(x_ref, x_tgt)
        assert isinstance(ax, mpl.axes.Axes)

    def test_wide_bounding_box(self):
        """Test with a wide (w > h) configuration."""
        x_ref = np.array([[0.0, 0.0], [3.0, 0.0], [1.5, 0.5]])
        x_tgt = np.array([[0.0, 0.05], [3.0, 0.0], [1.5, 0.55]])
        ax = tps_grid_2d_plot(x_ref, x_tgt)
        assert isinstance(ax, mpl.axes.Axes)

    def test_tall_bounding_box(self):
        """Test with a tall (h > w) configuration."""
        x_ref = np.array([[0.0, 0.0], [0.5, 0.0], [0.25, 3.0]])
        x_tgt = np.array([[0.05, 0.0], [0.5, 0.05], [0.25, 2.95]])
        ax = tps_grid_2d_plot(x_ref, x_tgt)
        assert isinstance(ax, mpl.axes.Axes)

    def test_equal_aspect_ratio(self, triangle_configs):
        """Test that axes have equal aspect ratio."""
        x_ref, x_tgt = triangle_configs
        ax = tps_grid_2d_plot(x_ref, x_tgt)
        assert ax.get_aspect() in ("equal", 1.0)
