"""Tests for normalized EFD (.nef) I/O functions."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from ktch.io import read_nef, write_nef
from ktch.io._nef import NefData


def _make_coeffs(n_harmonics=3, seed=42):
    """Create a reproducible coefficient matrix."""
    rng = np.random.RandomState(seed)
    coeffs = rng.randn(n_harmonics, 4) * 0.1
    coeffs[0] = [1.0, 0.0, 0.0, rng.rand() * 0.3]
    return coeffs


#
# NefData
#


class TestNefData:
    def test_basic(self):
        coeffs = _make_coeffs()
        data = NefData(sample_name="S1", coeffs=coeffs)
        assert data.n_harmonics == 3
        assert data.const_flags == (1, 1, 1, 0)

    def test_list_input_converted(self):
        data = NefData(
            sample_name="S1", coeffs=[[1, 0, 0, 0.2], [0.01, 0.02, 0.03, 0.04]]
        )
        assert isinstance(data.coeffs, np.ndarray)
        assert data.coeffs.shape == (2, 4)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="n_harmonics, 4"):
            NefData(sample_name="S1", coeffs=np.zeros((3, 5)))

    def test_to_numpy(self):
        coeffs = _make_coeffs()
        data = NefData(sample_name="S1", coeffs=coeffs)
        np.testing.assert_array_equal(data.to_numpy(), coeffs)

    def test_to_dataframe(self):
        coeffs = _make_coeffs(n_harmonics=2)
        data = NefData(sample_name="S1", coeffs=coeffs)
        df = data.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["a", "b", "c", "d"]
        assert df.index.names == ["specimen_id", "harmonic"]
        assert df.shape == (2, 4)
        assert df.loc[("S1", 1), "a"] == coeffs[0, 0]


#
# read / write round-trip
#


class TestReadWriteNef:
    def test_single_record_roundtrip(self):
        coeffs = _make_coeffs()

        with tempfile.NamedTemporaryFile(suffix=".nef", delete=False) as f:
            temp = f.name

        try:
            write_nef(temp, coeffs, sample_names="Test1")
            result = read_nef(temp)

            assert isinstance(result, NefData)
            assert result.sample_name == "Test1"
            assert result.n_harmonics == 3
            np.testing.assert_allclose(result.coeffs, coeffs, atol=1e-7)
        finally:
            os.unlink(temp)

    def test_multiple_records_roundtrip(self):
        c1 = _make_coeffs(seed=1)
        c2 = _make_coeffs(seed=2)

        with tempfile.NamedTemporaryFile(suffix=".nef", delete=False) as f:
            temp = f.name

        try:
            write_nef(temp, [c1, c2], sample_names=["A", "B"])
            result = read_nef(temp)

            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0].sample_name == "A"
            assert result[1].sample_name == "B"
            np.testing.assert_allclose(result[0].coeffs, c1, atol=1e-7)
            np.testing.assert_allclose(result[1].coeffs, c2, atol=1e-7)
        finally:
            os.unlink(temp)

    def test_as_frame_single(self):
        coeffs = _make_coeffs(n_harmonics=2)

        with tempfile.NamedTemporaryFile(suffix=".nef", delete=False) as f:
            temp = f.name

        try:
            write_nef(temp, coeffs, sample_names="S")
            df = read_nef(temp, as_frame=True)

            assert isinstance(df, pd.DataFrame)
            assert df.shape == (2, 4)
            assert df.index.names == ["specimen_id", "harmonic"]
        finally:
            os.unlink(temp)

    def test_as_frame_multiple(self):
        c1 = _make_coeffs(n_harmonics=2, seed=1)
        c2 = _make_coeffs(n_harmonics=2, seed=2)

        with tempfile.NamedTemporaryFile(suffix=".nef", delete=False) as f:
            temp = f.name

        try:
            write_nef(temp, [c1, c2], sample_names=["A", "B"])
            df = read_nef(temp, as_frame=True)

            assert isinstance(df, pd.DataFrame)
            assert df.shape == (4, 4)
            assert len(df.index.get_level_values("specimen_id").unique()) == 2
        finally:
            os.unlink(temp)

    def test_const_flags_preserved(self):
        coeffs = _make_coeffs()
        flags = (0, 0, 0, 0)

        with tempfile.NamedTemporaryFile(suffix=".nef", delete=False) as f:
            temp = f.name

        try:
            write_nef(temp, coeffs, sample_names="T", const_flags=flags)
            result = read_nef(temp)
            assert result.const_flags == flags
        finally:
            os.unlink(temp)


class TestHeaderParsing:
    def test_file_without_headers(self):
        content = "Sample_1\n  1.0  0.0  0.0  0.2\n  0.01  0.02  0.03  0.04\n"
        with tempfile.NamedTemporaryFile(suffix=".nef", mode="w", delete=False) as f:
            f.write(content)
            temp = f.name

        try:
            result = read_nef(temp)
            assert result.sample_name == "Sample_1"
            assert result.n_harmonics == 2
            assert result.const_flags == (1, 1, 1, 0)  # default
        finally:
            os.unlink(temp)

    def test_harmo_mismatch_raises(self):
        content = "#HARMO 5\nSample_1\n  1.0  0.0  0.0  0.2\n  0.01  0.02  0.03  0.04\n"
        with tempfile.NamedTemporaryFile(suffix=".nef", mode="w", delete=False) as f:
            f.write(content)
            temp = f.name

        try:
            with pytest.raises(ValueError, match="Expected 5 harmonics"):
                read_nef(temp)
        finally:
            os.unlink(temp)


class TestErrors:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_nef("nonexistent.nef")

    def test_wrong_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp = f.name
        try:
            with pytest.raises(ValueError, match="not a .nef file"):
                read_nef(temp)
        finally:
            os.unlink(temp)

    def test_sample_names_length_mismatch(self):
        coeffs = [_make_coeffs(), _make_coeffs()]
        with pytest.raises(ValueError, match="does not match"):
            write_nef("dummy.nef", coeffs, sample_names=["only_one"])

    def test_invalid_coeffs_shape(self):
        with pytest.raises(ValueError):
            write_nef("dummy.nef", np.zeros((3,)))
