"""Unit tests for plot._params module."""

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

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from ktch.plot._params import (
    _detect_shape_type,
    _get_renderer_and_projection,
    _resolve_descriptor_params,
    _resolve_reducer_params,
    _resolve_xy_hue,
    _validate_components,
)

# ===========================================================================
# _resolve_reducer_params
# ===========================================================================


class TestResolveReducerParams:
    def _make_reducer(self, *, n_components_=5, has_n_components_=True):
        """Create a mock reducer with configurable attributes."""
        reducer = Mock()
        reducer.inverse_transform = Mock(return_value=np.zeros((1, 10)))
        reducer.explained_variance_ = np.arange(1, 6, dtype=float)
        if has_n_components_:
            reducer.n_components_ = n_components_
        else:
            del reducer.n_components_
            reducer.n_components = n_components_
        return reducer

    def test_extracts_from_reducer(self):
        reducer = self._make_reducer()
        inv, ev, nc = _resolve_reducer_params(
            reducer, None, None, None, require_variance=True
        )
        assert inv is reducer.inverse_transform
        assert np.array_equal(ev, reducer.explained_variance_)
        assert nc == 5

    def test_explicit_overrides_reducer(self):
        reducer = self._make_reducer()
        custom_inv = Mock()
        custom_ev = np.array([1.0, 2.0])
        inv, ev, nc = _resolve_reducer_params(
            reducer, custom_inv, custom_ev, 3, require_variance=True
        )
        assert inv is custom_inv
        assert np.array_equal(ev, custom_ev)
        assert nc == 3

    def test_fallback_n_components(self):
        """Falls back from n_components_ to n_components."""
        reducer = self._make_reducer(n_components_=7, has_n_components_=False)
        _, _, nc = _resolve_reducer_params(
            reducer, None, None, None, require_variance=True
        )
        assert nc == 7

    def test_missing_inverse_transform_raises(self):
        with pytest.raises(ValueError, match="reducer"):
            _resolve_reducer_params(None, None, None, 5, require_variance=False)

    def test_missing_explained_variance_raises_when_required(self):
        reducer = Mock()
        reducer.inverse_transform = Mock()
        reducer.explained_variance_ = None
        del reducer.explained_variance_
        reducer.n_components_ = 3
        with pytest.raises(ValueError, match="explained_variance"):
            _resolve_reducer_params(reducer, None, None, None, require_variance=True)

    def test_missing_explained_variance_ok_when_not_required(self):
        inv_fn = Mock()
        inv, ev, nc = _resolve_reducer_params(
            None, inv_fn, None, 5, require_variance=False
        )
        assert inv is inv_fn
        assert ev is None
        assert nc == 5

    def test_missing_n_components_raises(self):
        inv_fn = Mock()
        ev = np.array([1.0])
        with pytest.raises(ValueError, match="n_components"):
            _resolve_reducer_params(None, inv_fn, ev, None, require_variance=True)


# ===========================================================================
# _resolve_descriptor_params
# ===========================================================================


class TestResolveDescriptorParams:
    def test_extracts_from_descriptor(self):
        descriptor = Mock()
        descriptor.inverse_transform = Mock()
        inv, n_dim = _resolve_descriptor_params(descriptor, None, None, "auto")
        assert inv is descriptor.inverse_transform
        assert n_dim is None

    def test_explicit_override(self):
        descriptor = Mock()
        descriptor.inverse_transform = Mock()
        custom_inv = Mock()
        inv, n_dim = _resolve_descriptor_params(descriptor, custom_inv, 2, "auto")
        assert inv is custom_inv
        assert n_dim == 2

    def test_identity_case_with_n_dim(self):
        inv, n_dim = _resolve_descriptor_params(None, None, 2, "auto")
        assert inv is None
        assert n_dim == 2

    def test_identity_infers_n_dim_landmarks_2d(self):
        inv, n_dim = _resolve_descriptor_params(None, None, None, "landmarks_2d")
        assert inv is None
        assert n_dim == 2

    def test_identity_infers_n_dim_landmarks_3d(self):
        inv, n_dim = _resolve_descriptor_params(None, None, None, "landmarks_3d")
        assert inv is None
        assert n_dim == 3

    def test_identity_missing_n_dim_raises(self):
        with pytest.raises(ValueError, match="n_dim is required"):
            _resolve_descriptor_params(None, None, None, "auto")


# ===========================================================================
# _detect_shape_type
# ===========================================================================


class TestDetectShapeType:
    def test_surface_3d(self):
        sample = np.zeros((10, 20, 3))
        result = _detect_shape_type(sample, Mock(), None)
        assert result == "surface_3d"

    def test_curve_2d(self):
        sample = np.zeros((50, 2))
        result = _detect_shape_type(sample, Mock(), None)
        assert result == "curve_2d"

    def test_curve_3d(self):
        sample = np.zeros((50, 3))
        result = _detect_shape_type(sample, Mock(), None)
        assert result == "curve_3d"

    def test_curve_high_dim(self):
        """ndim=2 with last dim > 3 is still curve_3d."""
        sample = np.zeros((50, 5))
        result = _detect_shape_type(sample, Mock(), None)
        assert result == "curve_3d"

    def test_landmarks_2d(self):
        sample = np.zeros((10, 2))
        result = _detect_shape_type(sample, None, 2)
        assert result == "landmarks_2d"

    def test_landmarks_3d(self):
        sample = np.zeros((10, 3))
        result = _detect_shape_type(sample, None, 3)
        assert result == "landmarks_3d"

    def test_1d_array_raises(self):
        sample = np.zeros(10)
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            _detect_shape_type(sample, Mock(), None)

    def test_single_column_raises(self):
        sample = np.zeros((10, 1))
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            _detect_shape_type(sample, Mock(), None)

    def test_identity_unknown_n_dim_raises(self):
        sample = np.zeros((10, 2))
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            _detect_shape_type(sample, None, 4)


# ===========================================================================
# _get_renderer_and_projection
# ===========================================================================


class TestGetRendererAndProjection:
    def test_valid_shape_types(self):
        from ktch.plot._renderers import _SHAPE_TYPE_REGISTRY

        for st in _SHAPE_TYPE_REGISTRY:
            renderer, proj = _get_renderer_and_projection(st, None)
            assert callable(renderer)

    def test_invalid_shape_type_raises(self):
        with pytest.raises(ValueError, match="Invalid shape_type"):
            _get_renderer_and_projection("invalid_type", None)

    def test_custom_render_fn_with_valid_type(self):
        custom_fn = Mock()
        renderer, proj = _get_renderer_and_projection("curve_2d", custom_fn)
        assert renderer is custom_fn
        assert proj is None  # curve_2d has no 3D projection

    def test_custom_render_fn_with_3d_type(self):
        custom_fn = Mock()
        renderer, proj = _get_renderer_and_projection("surface_3d", custom_fn)
        assert renderer is custom_fn
        assert proj == "3d"

    def test_custom_render_fn_with_unknown_type(self):
        custom_fn = Mock()
        renderer, proj = _get_renderer_and_projection("unknown", custom_fn)
        assert renderer is custom_fn
        assert proj is None


# ===========================================================================
# _validate_components
# ===========================================================================


class TestValidateComponents:
    def test_valid_indices(self):
        _validate_components((0, 1, 2), 5)  # should not raise

    def test_single_valid_index(self):
        _validate_components((0,), 1)  # should not raise

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="Component index 5"):
            _validate_components((0, 5), 5)

    def test_negative_index_raises(self):
        with pytest.raises(ValueError, match="Component index -1"):
            _validate_components((-1,), 5)

    def test_boundary_index_raises(self):
        with pytest.raises(ValueError, match="Component index 3"):
            _validate_components((3,), 3)

    def test_boundary_index_valid(self):
        _validate_components((2,), 3)  # should not raise


# ===========================================================================
# _resolve_xy_hue
# ===========================================================================


class TestResolveXyHue:
    def test_raw_arrays(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        x_arr, y_arr, hue_arr, cats = _resolve_xy_hue(None, x, y, None)
        assert np.array_equal(x_arr, x)
        assert np.array_equal(y_arr, y)
        assert hue_arr is None
        assert cats is None

    def test_raw_arrays_with_hue(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        hue = np.array(["A", "B", "A"])
        x_arr, y_arr, hue_arr, cats = _resolve_xy_hue(None, x, y, hue)
        assert np.array_equal(x_arr, x)
        assert hue_arr is not None
        assert set(cats) == {"A", "B"}

    def test_dataframe_columns(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "g": ["A", "B", "A"]})
        x_arr, y_arr, hue_arr, cats = _resolve_xy_hue(df, "x", "y", "g")
        assert np.array_equal(x_arr, [1, 2, 3])
        assert np.array_equal(y_arr, [4, 5, 6])
        assert list(hue_arr) == ["A", "B", "A"]

    def test_dataframe_no_hue(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        x_arr, y_arr, hue_arr, cats = _resolve_xy_hue(df, "x", "y", None)
        assert hue_arr is None
        assert cats is None

    def test_categorical_hue_order_preserved(self):
        cat_order = ["C", "A", "B"]
        df = pd.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [4, 5, 6],
                "g": pd.Categorical(
                    ["A", "B", "C"], categories=cat_order, ordered=True
                ),
            }
        )
        _, _, _, cats = _resolve_xy_hue(df, "x", "y", "g")
        assert cats == cat_order

    def test_hue_order_override(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        hue = np.array(["A", "B", "C"])
        _, _, _, cats = _resolve_xy_hue(None, x, y, hue, hue_order=["C", "B", "A"])
        assert cats == ["C", "B", "A"]

    def test_x_none_raises(self):
        with pytest.raises(ValueError, match="x and y are required"):
            _resolve_xy_hue(None, None, np.array([1.0]), None)

    def test_y_none_raises(self):
        with pytest.raises(ValueError, match="x and y are required"):
            _resolve_xy_hue(None, np.array([1.0]), None, None)

    def test_both_none_raises(self):
        with pytest.raises(ValueError, match="x and y are required"):
            _resolve_xy_hue(None, None, None, None)

    def test_lists_converted_to_arrays(self):
        x_arr, y_arr, _, _ = _resolve_xy_hue(None, [1, 2, 3], [4, 5, 6], None)
        assert isinstance(x_arr, np.ndarray)
        assert isinstance(y_arr, np.ndarray)

    def test_unique_hue_order_from_raw_arrays(self):
        """Without hue_order, categories follow insertion order."""
        hue = np.array(["B", "A", "B", "C"])
        _, _, _, cats = _resolve_xy_hue(None, np.arange(4.0), np.arange(4.0), hue)
        assert cats == ["B", "A", "C"]
