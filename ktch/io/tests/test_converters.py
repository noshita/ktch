"""Tests for format conversion functions."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ktch.io import read_spharmpdm_coef
from ktch.io._converters import (
    _cvt_spharm_coef_list_to_spharmpdm,
    _cvt_spharm_coef_spharmpdm_to_list,
    convert_coords_df_to_df_sklearn_transform,
    convert_coords_df_to_list,
    efa_coeffs_to_nef,
    nef_to_efa_coeffs,
    sha_coeffs_to_spharmpdm,
    spharmpdm_to_sha_coeffs,
)
from ktch.io._nef import NefData
from ktch.io._spharm_pdm import SpharmPdmData


def _make_nef(n_harmonics=3, seed=42):
    rng = np.random.RandomState(seed)
    coeffs = rng.randn(n_harmonics, 4) * 0.1
    coeffs[0] = [1.0, 0.0, 0.0, rng.rand() * 0.3]
    return NefData(specimen_name="S1", coeffs=coeffs)


#
# nef_to_efa_coeffs
#


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
        data = {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        }
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


#
# SPHARM-PDM format packing
#


class TestSpharmCoefFormatPacking:
    @pytest.fixture()
    def spharmpdm_data(self):
        path = Path(__file__).parent / "data" / "andesred_07_allSegments_SPHARM.coef"
        return read_spharmpdm_coef(path)

    def test_roundtrip_list(self, spharmpdm_data):
        """list -> spharmpdm -> list produces identical coefficients."""
        coef_spharmpdm = _cvt_spharm_coef_list_to_spharmpdm(spharmpdm_data.coeffs)
        coef_list_rt = _cvt_spharm_coef_spharmpdm_to_list(coef_spharmpdm)

        for l in range(len(spharmpdm_data.coeffs)):
            np.testing.assert_array_almost_equal(
                spharmpdm_data.coeffs[l], coef_list_rt[l]
            )

    def test_roundtrip_spharmpdm(self, spharmpdm_data):
        """spharmpdm -> list -> spharmpdm produces identical array."""
        coef_pdm = _cvt_spharm_coef_list_to_spharmpdm(spharmpdm_data.coeffs)
        coef_list_rt = _cvt_spharm_coef_spharmpdm_to_list(coef_pdm)
        coef_pdm_rt = _cvt_spharm_coef_list_to_spharmpdm(coef_list_rt)

        np.testing.assert_array_almost_equal(coef_pdm, coef_pdm_rt)


# --- SPHARM-PDM <-> SHA bridge converters ---


class TestSpharmpdmToShaCoeffs:
    @pytest.fixture()
    def spharmpdm_data(self):
        path = Path(__file__).parent / "data" / "andesred_07_allSegments_SPHARM.coef"
        return read_spharmpdm_coef(path)

    def test_output_shape(self, spharmpdm_data):
        result = spharmpdm_to_sha_coeffs(spharmpdm_data)
        l_max = spharmpdm_data.l_max
        assert result.shape == (1, 3 * (l_max + 1) ** 2)

    def test_output_dtype(self, spharmpdm_data):
        result = spharmpdm_to_sha_coeffs(spharmpdm_data)
        assert result.dtype == np.float64

    def test_layout_lmax0_real(self):
        """l=0 coefficients: m=0 is real, same in complex and real basis."""
        # For l=0, m=0: a_0^0 = Re(c_0^0). Use real c_0^0.
        coeffs = [
            np.array([[1.0 + 0j, 3.0 + 0j, 5.0 + 0j]]),  # l=0
        ]
        data = SpharmPdmData(specimen_name="T", coeffs=coeffs)
        result = spharmpdm_to_sha_coeffs(data)

        assert result.shape == (1, 3)
        assert result.dtype == np.float64
        np.testing.assert_allclose(result[0], [1.0, 3.0, 5.0])

    def test_roundtrip_preserves_shape(self, spharmpdm_data):
        """SHA flat → SpharmPdmData → SHA flat round-trip."""
        sha_flat = spharmpdm_to_sha_coeffs(spharmpdm_data)
        rt_list = sha_coeffs_to_spharmpdm(
            sha_flat, specimen_names=[spharmpdm_data.specimen_name]
        )
        sha_flat_rt = spharmpdm_to_sha_coeffs(rt_list)
        np.testing.assert_allclose(sha_flat_rt, sha_flat, atol=1e-12)

    def test_batch(self, spharmpdm_data):
        result = spharmpdm_to_sha_coeffs([spharmpdm_data, spharmpdm_data])
        l_max = spharmpdm_data.l_max
        assert result.shape == (2, 3 * (l_max + 1) ** 2)
        np.testing.assert_array_equal(result[0], result[1])


class TestShaCoeffsToSpharmpdm:
    @pytest.fixture()
    def spharmpdm_data(self):
        path = Path(__file__).parent / "data" / "andesred_07_allSegments_SPHARM.coef"
        return read_spharmpdm_coef(path)

    def test_roundtrip(self, spharmpdm_data):
        """SpharmPdmData -> SHA flat -> SpharmPdmData round-trip."""
        sha_flat = spharmpdm_to_sha_coeffs(spharmpdm_data)
        result_list = sha_coeffs_to_spharmpdm(
            sha_flat, specimen_names=[spharmpdm_data.specimen_name]
        )

        assert len(result_list) == 1
        result = result_list[0]
        assert result.specimen_name == spharmpdm_data.specimen_name
        assert result.l_max == spharmpdm_data.l_max
        # SpharmPdmData.coeffs remains complex (format-faithful)
        for l in range(result.l_max + 1):
            assert result.coeffs[l].dtype == np.complex128
            np.testing.assert_array_almost_equal(
                result.coeffs[l], spharmpdm_data.coeffs[l]
            )

    def test_default_specimen_names(self):
        coeffs = np.zeros((2, 3))  # l_max=0, 2 samples
        result = sha_coeffs_to_spharmpdm(coeffs)
        assert result[0].specimen_name == "Specimen_0"
        assert result[1].specimen_name == "Specimen_1"

    def test_invalid_length_not_divisible_by_3(self):
        with pytest.raises(ValueError, match="not divisible by 3"):
            sha_coeffs_to_spharmpdm(np.zeros((1, 5)))

    def test_invalid_length_not_perfect_square(self):
        with pytest.raises(ValueError, match="not a perfect square"):
            sha_coeffs_to_spharmpdm(np.zeros((1, 6)))  # 6/3=2, not a square

    def test_single_sample_1d_input(self, spharmpdm_data):
        """1D input is promoted to 2D."""
        sha_flat = spharmpdm_to_sha_coeffs(spharmpdm_data)
        result = sha_coeffs_to_spharmpdm(sha_flat[0])
        assert len(result) == 1
        assert result[0].l_max == spharmpdm_data.l_max
