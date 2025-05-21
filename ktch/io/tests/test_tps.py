from pathlib import Path
import re

import pytest
import numpy as np

from ktch.io import read_tps


def test_read_tps_shape():
    path = Path(__file__).parent / "data" / "landmarks_triangle.tps"

    landmarks = read_tps(path)

    assert landmarks.shape == (50, 3, 2)


def test_read_tps_with_curves():
    """Test that the read_tps function correctly handles TPS files with CURVES data."""
    path = Path(__file__).parent / "data" / "sample_curve_data.tps"
    
    result = read_tps(path)
    
    assert isinstance(result, tuple), "Result should be a tuple when file contains CURVES data"
    assert len(result) == 2, "Result should contain landmarks and curves"
    
    landmarks, curves = result
    
    assert landmarks.shape == (1, 2), "Landmarks array should have one landmark"
    assert np.isclose(landmarks[0, 0], 150.0), "Landmark x coordinate should be 150.0"
    assert np.isclose(landmarks[0, 1], 250.0), "Landmark y coordinate should be 250.0"
    
    # Verify curves data
    assert isinstance(curves, list), "Curves should be a list"
    assert len(curves) == 1, "There should be 1 curve (CURVES=1)"
    assert isinstance(curves[0], np.ndarray), "Each curve should be a numpy array"
    assert curves[0].shape == (50, 2), "The curve should have 50 points with 2 coordinates each"
    
    # Verify specific coordinates
    assert np.isclose(curves[0][0, 0], 320.0), "First x coordinate should be 320.0"
    assert np.isclose(curves[0][0, 1], 145.0), "First y coordinate should be 145.0"
    assert np.isclose(curves[0][49, 0], 375.0), "Last x coordinate should be 375.0"
    assert np.isclose(curves[0][49, 1], 465.0), "Last y coordinate should be 465.0"
