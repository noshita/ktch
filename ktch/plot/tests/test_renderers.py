"""Unit tests for plot._renderers module."""

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
import matplotlib.pyplot as plt

from ktch.plot._renderers import (
    _render_curve_2d,
    _render_curve_3d,
    _render_landmarks_2d,
    _render_landmarks_3d,
    _render_surface_3d,
    _resolve_render_kw,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestResolveRenderKw:
    def test_no_conflict(self):
        result = _resolve_render_kw(
            {"linewidth": 2},
            color="red",
            alpha=0.5,
            links=None,
        )
        assert result == {
            "linewidth": 2,
            "color": "red",
            "alpha": 0.5,
            "links": None,
        }

    def test_conflict_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _resolve_render_kw(
                {"color": "blue", "linewidth": 2},
                color="red",
                alpha=0.5,
                links=None,
            )
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "color" in str(w[0].message)
        assert result["color"] == "red"
        assert result["linewidth"] == 2

    def test_empty_render_kw(self):
        result = _resolve_render_kw(
            {},
            color="gray",
            alpha=1.0,
            links=None,
        )
        assert result == {"color": "gray", "alpha": 1.0, "links": None}


class TestRenderCurve2d:
    def test_smoke(self):
        fig, ax = plt.subplots()
        coords = np.column_stack(
            [
                np.cos(np.linspace(0, 2 * np.pi, 50)),
                np.sin(np.linspace(0, 2 * np.pi, 50)),
            ]
        )
        _render_curve_2d(coords, ax)
        assert len(ax.lines) == 1

    def test_3_plus_columns_uses_first_2(self):
        fig, ax = plt.subplots()
        coords = np.random.default_rng(0).standard_normal((50, 4))
        _render_curve_2d(coords, ax)
        assert len(ax.lines) == 1


class TestRenderCurve3d:
    def test_smoke(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        t = np.linspace(0, 2 * np.pi, 50)
        coords = np.column_stack([np.cos(t), np.sin(t), t / (2 * np.pi)])
        _render_curve_3d(coords, ax)
        assert len(ax.lines) > 0


class TestRenderSurface3d:
    def test_smoke(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        theta = np.linspace(0, np.pi, 10)
        phi = np.linspace(0, 2 * np.pi, 20)
        T, P = np.meshgrid(theta, phi, indexing="ij")
        coords = np.stack(
            [
                np.sin(T) * np.cos(P),
                np.sin(T) * np.sin(P),
                np.cos(T),
            ],
            axis=-1,
        )
        _render_surface_3d(coords, ax)


class TestRenderLandmarks2d:
    def test_smoke_no_links(self):
        fig, ax = plt.subplots()
        coords = np.random.default_rng(0).standard_normal((10, 2))
        _render_landmarks_2d(coords, ax)
        assert len(ax.collections) > 0

    def test_with_links(self):
        fig, ax = plt.subplots()
        coords = np.random.default_rng(0).standard_normal((10, 2))
        links = [[0, 1], [1, 2], [2, 3]]
        _render_landmarks_2d(coords, ax, links=links)
        # LineCollection + scatter PathCollection
        assert len(ax.collections) == 2


class TestRenderLandmarks3d:
    def test_smoke_no_links(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        coords = np.random.default_rng(0).standard_normal((10, 3))
        _render_landmarks_3d(coords, ax)

    def test_with_links(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        coords = np.random.default_rng(0).standard_normal((10, 3))
        links = [[0, 1], [1, 2]]
        _render_landmarks_3d(coords, ax, links=links)
        assert len(ax.lines) == 2
