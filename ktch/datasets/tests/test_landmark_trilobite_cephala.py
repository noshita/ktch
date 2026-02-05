"""Tests for the trilobite cephala landmark dataset."""

import numpy as np
import pandas as pd
import pytest

from ktch.datasets import load_landmark_trilobite_cephala


class TestLoadLandmarkTrilobiteCephala:
    """Tests for load_landmark_trilobite_cephala function."""

    def test_default_loading(self):
        """Test default loading returns correct types and shapes."""
        data = load_landmark_trilobite_cephala()

        assert isinstance(data.landmarks, np.ndarray)
        assert data.landmarks.shape == (301, 16, 2)
        assert isinstance(data.meta, dict)

    def test_curves_structure(self):
        """Test that curves have the expected structure."""
        data = load_landmark_trilobite_cephala()

        assert isinstance(data.curves, list)
        assert len(data.curves) == 301

        # Each specimen has 4 curves
        for curves in data.curves:
            assert len(curves) == 4

        # Check curve point counts for the first specimen
        assert data.curves[0][0].shape == (12, 2)
        assert data.curves[0][1].shape == (20, 2)
        assert data.curves[0][2].shape == (20, 2)
        assert data.curves[0][3].shape == (20, 2)

    def test_curve_landmarks(self):
        """Test curve_landmarks values."""
        data = load_landmark_trilobite_cephala()

        assert data.curve_landmarks == [(1, 6), (9, 11), (12, 14), (3, 14)]

    def test_bunch_keys(self):
        """Test that Bunch contains all expected keys."""
        data = load_landmark_trilobite_cephala()

        expected_keys = {
            "landmarks",
            "curves",
            "curve_landmarks",
            "meta",
            "DESCR",
            "filename",
        }
        assert set(data.keys()) == expected_keys

    def test_as_frame_true(self):
        """Test loading with as_frame=True returns DataFrames."""
        data = load_landmark_trilobite_cephala(as_frame=True)

        assert isinstance(data.landmarks, pd.DataFrame)
        assert isinstance(data.meta, pd.DataFrame)

    def test_descr_not_empty(self):
        """Test that the description is not empty."""
        data = load_landmark_trilobite_cephala()

        assert isinstance(data.DESCR, str)
        assert len(data.DESCR) > 0

    def test_filename(self):
        """Test that filename is set correctly."""
        data = load_landmark_trilobite_cephala()

        assert data.filename == "data_landmark_trilobite_cephala.tps"
