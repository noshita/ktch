"""Tests for the synthetic 3D leaf bending outline dataset."""

import numpy as np
import pandas as pd

from ktch.datasets import load_outline_leaf_bending


class TestLoadOutlineLeafBending:
    """Tests for load_outline_leaf_bending function."""

    def test_default_loading(self):
        """Test default loading returns correct types and shapes."""
        data = load_outline_leaf_bending()

        assert isinstance(data.coords, np.ndarray)
        assert data.coords.shape == (60, 200, 3)
        assert isinstance(data.meta, dict)

    def test_as_frame_true(self):
        """Test loading with as_frame=True returns DataFrames."""
        data = load_outline_leaf_bending(as_frame=True)

        assert isinstance(data.coords, pd.DataFrame)
        assert isinstance(data.meta, pd.DataFrame)

    def test_bunch_keys(self):
        """Test that Bunch contains all expected keys."""
        data = load_outline_leaf_bending()

        expected_keys = {
            "coords",
            "meta",
            "DESCR",
            "filename",
        }
        assert set(data.keys()) == expected_keys

    def test_descr_not_empty(self):
        """Test that the description is not empty."""
        data = load_outline_leaf_bending()

        assert isinstance(data.DESCR, str)
        assert len(data.DESCR) > 0

    def test_filename(self):
        """Test that filename is set correctly."""
        data = load_outline_leaf_bending()

        assert data.filename == "data_outline_leaf_bending.csv"
