"""Tests for chain code I/O functions."""

import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

from ktch.io import read_chc, write_chc


def test_read_chc():
    """Test read_chc function."""
    sample_data = "Sample1 100 200 1.5 1000 0 1 2 3 4 5 6 7 0 -1"

    with tempfile.NamedTemporaryFile(suffix=".chc", delete=False) as f:
        f.write(sample_data.encode())
        temp_file = f.name

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            chain_code = read_chc(temp_file, as_coordinates=False)
            assert len(w) == 1
            assert "area_pixels" in str(w[0].message)

        assert isinstance(chain_code, np.ndarray)
        assert np.array_equal(chain_code, np.array([0, 1, 2, 3, 4, 5, 6, 7, 0]))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coords = read_chc(temp_file, as_coordinates=True)
        assert isinstance(coords, np.ndarray)
        assert coords.shape[0] == 10  # 10 points (including start and end)
        assert coords.shape[1] == 2  # 2D coordinates (x, y)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = read_chc(temp_file, as_frame=True)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == 10  # 10 coordinate points
        assert len(df.index.levels[0]) == 1  # 1 sample
        assert "x" in df.columns
        assert "y" in df.columns
        assert "chain_code" in df.columns
    finally:
        os.unlink(temp_file)


def test_write_chc():
    """Test write_chc function."""
    chain_code = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0])

    with tempfile.NamedTemporaryFile(suffix=".chc", delete=False) as f:
        temp_file = f.name

    try:
        write_chc(
            temp_file,
            chain_code,
            sample_names="Test",
            xs=100,
            ys=200,
            area_per_pixels=1.5,
        )

        chain_code_read = read_chc(temp_file, as_coordinates=False)
        assert np.array_equal(chain_code, chain_code_read)

        coords = read_chc(temp_file, as_coordinates=True)
        assert isinstance(coords, np.ndarray)
        assert coords.shape[0] == 10  # 10 points (including start and end)
        assert coords.shape[1] == 2  # 2D coordinates (x, y)
    finally:
        os.unlink(temp_file)


def test_multiple_chain_codes():
    """Test reading and writing multiple chain codes."""
    chain_code1 = np.array([0, 1, 2, 3, 4])
    chain_code2 = np.array([5, 6, 7, 0, 1])

    with tempfile.NamedTemporaryFile(suffix=".chc", delete=False) as f:
        temp_file = f.name

    try:
        write_chc(
            temp_file,
            [chain_code1, chain_code2],
            sample_names=["Test1", "Test2"],
            xs=[100, 200],
            ys=[150, 250],
            area_per_pixels=[1.0, 2.0],
        )

        chain_codes_read = read_chc(temp_file, as_coordinates=False)

        assert isinstance(chain_codes_read, list)
        assert len(chain_codes_read) == 2

        assert np.array_equal(chain_code1, chain_codes_read[0])
        assert np.array_equal(chain_code2, chain_codes_read[1])

        coords_list = read_chc(temp_file, as_coordinates=True)
        assert isinstance(coords_list, list)
        assert len(coords_list) == 2
        assert coords_list[0].shape[0] == 6  # 6 points (including start and end)
        assert coords_list[1].shape[0] == 6  # 6 points (including start and end)
    finally:
        os.unlink(temp_file)


def test_invalid_chain_code():
    """Test that invalid chain code values are always rejected."""
    invalid_chain_code = np.array([0, 1, 2, 8, 9])

    with pytest.raises(ValueError, match="invalid values"):
        write_chc("dummy.chc", invalid_chain_code, sample_names="Test")


def test_area_pixels_mismatch_warning():
    """Test that a warning is issued when area_pixels differs from computed value."""
    chain_code = np.array([0, 2, 4, 6])  # Square: area = 4 pixels

    with tempfile.NamedTemporaryFile(suffix=".chc", delete=False) as f:
        temp_file = f.name

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            write_chc(
                temp_file,
                chain_code,
                sample_names="Test",
                area_pixels=999,
            )
            assert len(w) == 1
            assert "area_pixels" in str(w[0].message)
            assert "Using computed value" in str(w[0].message)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            write_chc(
                temp_file,
                chain_code,
                sample_names="Test",
            )
            assert len(w) == 0
    finally:
        os.unlink(temp_file)


def test_area_pixels_auto_computed():
    """Test that area_pixels is correctly computed from chain code."""
    # Square [0, 2, 4, 6]: polygon_area=1, B=4, pixels=1+2+1=4
    chain_code = np.array([0, 2, 4, 6])

    with tempfile.NamedTemporaryFile(suffix=".chc", delete=False) as f:
        temp_file = f.name

    try:
        write_chc(temp_file, chain_code, sample_names="Square")

        with open(temp_file, "r") as f:
            line = f.read().strip()
        area_in_file = int(line.split(" ")[4])
        assert area_in_file == 4
    finally:
        os.unlink(temp_file)


def test_area_pixels_diagonal():
    """Test area computation with diagonal chain code steps."""
    # Diamond [1, 3, 5, 7]: polygon_area=2, B=4, pixels=2+2+1=5
    chain_code = np.array([1, 3, 5, 7])

    with tempfile.NamedTemporaryFile(suffix=".chc", delete=False) as f:
        temp_file = f.name

    try:
        write_chc(temp_file, chain_code, sample_names="Diamond")

        with open(temp_file, "r") as f:
            line = f.read().strip()
        area_in_file = int(line.split(" ")[4])
        assert area_in_file == 5
    finally:
        os.unlink(temp_file)


def test_area_pixels_mixed():
    """Test area computation with mixed orthogonal and diagonal steps."""
    # [0, 0, 1, 2, 4, 4, 5, 6]: polygon_area=5, B=8, pixels=5+4+1=10
    chain_code = np.array([0, 0, 1, 2, 4, 4, 5, 6])

    with tempfile.NamedTemporaryFile(suffix=".chc", delete=False) as f:
        temp_file = f.name

    try:
        write_chc(temp_file, chain_code, sample_names="Mixed")

        with open(temp_file, "r") as f:
            line = f.read().strip()
        area_in_file = int(line.split(" ")[4])
        assert area_in_file == 10
    finally:
        os.unlink(temp_file)


def test_area_pixels_mixed_complex():
    """Test area computation with mixed orthogonal and diagonal steps."""
    # [1, 7, 0, 2, 2, 2, 4, 4, 4, 5, 6, 7]: polygon_area=10, B=12, pixels=10+6+1=17
    chain_code = np.array([1, 7, 0, 2, 2, 2, 4, 4, 4, 5, 6, 7])

    with tempfile.NamedTemporaryFile(suffix=".chc", delete=False) as f:
        temp_file = f.name

    try:
        write_chc(temp_file, chain_code, sample_names="Mixed")

        with open(temp_file, "r") as f:
            line = f.read().strip()
        area_in_file = int(line.split(" ")[4])
        assert area_in_file == 17
    finally:
        os.unlink(temp_file)
