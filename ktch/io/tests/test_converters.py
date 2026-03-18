"""Tests for format conversion functions."""

import numpy as np
import pandas as pd
import pytest

from ktch.io._converters import (
    convert_coords_df_to_list,
    convert_coords_df_to_df_sklearn_transform,
    efa_coeffs_to_nef,
    nef_to_efa_coeffs,
)
from ktch.io._nef import NefData


def _make_nef(n_harmonics=3, seed=42):
    rng = np.random.RandomState(seed)
    coeffs = rng.randn(n_harmonics, 4) * 0.1
    coeffs[0] = [1.0, 0.0, 0.0, rng.rand() * 0.3]
    return NefData(specimen_name="S1", coeffs=coeffs)


# --- nef_to_efa_coeffs ---


class TestNefToEfaCoeffs:
    def test_layout(self):
        coeffs = np.array([[1.0, 0.0, 0.0, 0.2], [0.01, 0.02, 0.03, 0.04]])
        nef = NefData(specimen_name="T", coeffs=coeffs)
        result = nef_to_efa_coeffs(nef)

        assert result.shape == (1, 4 * 3)  # 4 axes * (2 harmonics + 1 DC)
        # Layout: [a_0, a_1, a_2, b_0, b_1, b_2, c_0, c_1, c_2, d_0, d_1, d_2]
        # DC offset = 0
        assert result[0, 0] == 0.0  # a_0 (DC)
        assert result[0, 1] == 1.0  # a_1
        assert result[0, 2] == 0.01  # a_2
        assert result[0, 3] == 0.0  # b_0 (DC)
        assert result[0, 4] == 0.0  # b_1
        assert result[0, 5] == 0.02  # b_2
        assert result[0, 6] == 0.0  # c_0 (DC)
        assert result[0, 7] == 0.0  # c_1
        assert result[0, 8] == 0.03  # c_2

    def test_dc_offset(self):
        coeffs = np.array([[1.0, 0.0, 0.0, 0.2]])
        nef = NefData(specimen_name="T", coeffs=coeffs)
        dc = np.array([10.0, 0.0, 20.0, 0.0])
        result = nef_to_efa_coeffs(nef, dc_offset=dc)

        assert result[0, 0] == 10.0  # a_0
        assert result[0, 2] == 0.0  # b_0
        assert result[0, 4] == 20.0  # c_0

    def test_batch(self):
        nef1 = _make_nef(n_harmonics=2, seed=1)
        nef2 = _make_nef(n_harmonics=2, seed=2)
        result = nef_to_efa_coeffs([nef1, nef2])

        assert result.shape == (2, 4 * 3)


# --- efa_coeffs_to_nef ---


class TestEfaCoeffsToNef:
    def test_roundtrip(self):
        nef_orig = _make_nef(n_harmonics=5)
        efa_flat = nef_to_efa_coeffs(nef_orig)
        nef_list = efa_coeffs_to_nef(efa_flat, specimen_names=["Roundtrip"])

        assert len(nef_list) == 1
        assert nef_list[0].specimen_name == "Roundtrip"
        np.testing.assert_allclose(nef_list[0].coeffs, nef_orig.coeffs, atol=1e-12)

    def test_strips_orientation_scale(self):
        nef_orig = _make_nef(n_harmonics=3)
        efa_flat = nef_to_efa_coeffs(nef_orig)
        # Append psi and scale (2 extra columns)
        extra = np.array([[0.5, 1.2]])
        efa_with_extras = np.hstack([efa_flat, extra])
        nef_list = efa_coeffs_to_nef(efa_with_extras)

        np.testing.assert_allclose(nef_list[0].coeffs, nef_orig.coeffs, atol=1e-12)

    def test_3d_raises(self):
        with pytest.raises(ValueError, match="Only 2D"):
            efa_coeffs_to_nef(np.zeros((1, 24)), n_dim=3)

    def test_from_efa_coeffs_classmethod(self):
        nef_orig = _make_nef(n_harmonics=3)
        efa_flat = nef_to_efa_coeffs(nef_orig)
        nef_result = NefData.from_efa_coeffs(efa_flat[0], specimen_name="cls")

        assert nef_result.specimen_name == "cls"
        np.testing.assert_allclose(nef_result.coeffs, nef_orig.coeffs, atol=1e-12)


# --- Coordinate conversion utilities ---


class TestCoordConversion:
    def _make_coords_df(self):
        index = pd.MultiIndex.from_tuples(
            [("A", 0), ("A", 1), ("A", 2), ("B", 0), ("B", 1), ("B", 2)],
            names=["specimen_id", "coord_id"],
        )
        data = {"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "y": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]}
        return pd.DataFrame(data, index=index)

    def test_convert_coords_df_to_list(self):
        df = self._make_coords_df()
        result = convert_coords_df_to_list(df)

        assert len(result) == 2
        assert result[0].shape == (3, 2)
        assert result[1].shape == (3, 2)
        np.testing.assert_array_equal(result[0], [[1, 7], [2, 8], [3, 9]])

    def test_convert_coords_df_to_df_sklearn_transform(self):
        df = self._make_coords_df()
        result = convert_coords_df_to_df_sklearn_transform(df)

        assert result.shape == (2, 6)  # 2 specimens, 3 coords * 2 dims
        assert "A" in result.index
        assert "B" in result.index
