from pathlib import Path

import numpy as np
import pytest

from ktch.io import read_tps, write_tps


def test_read_tps_shape():
    path = Path(__file__).parent / "data" / "landmarks_triangle.tps"

    landmarks = read_tps(path)

    assert landmarks.shape == (50, 3, 2)


def test_read_tps_with_curves():
    """Test that the read_tps function correctly handles TPS files with CURVES data."""
    path = Path(__file__).parent / "data" / "sample_curve_data.tps"

    result = read_tps(path)

    assert isinstance(result, tuple), (
        "Result should be a tuple when file contains CURVES data"
    )
    assert len(result) == 2, "Result should contain landmarks and curves"

    landmarks, curves = result

    assert landmarks.shape == (1, 2), "Landmarks array should have one landmark"
    assert np.isclose(landmarks[0, 0], 150.0), "Landmark x coordinate should be 150.0"
    assert np.isclose(landmarks[0, 1], 250.0), "Landmark y coordinate should be 250.0"

    # Verify curves data
    assert isinstance(curves, list), "Curves should be a list"
    assert len(curves) == 1, "There should be 1 curve (CURVES=1)"
    assert isinstance(curves[0], np.ndarray), "Each curve should be a numpy array"
    assert curves[0].shape == (50, 2), (
        "The curve should have 50 points with 2 coordinates each"
    )

    # Verify specific coordinates
    assert np.isclose(curves[0][0, 0], 320.0), "First x coordinate should be 320.0"
    assert np.isclose(curves[0][0, 1], 145.0), "First y coordinate should be 145.0"
    assert np.isclose(curves[0][49, 0], 375.0), "Last x coordinate should be 375.0"
    assert np.isclose(curves[0][49, 1], 465.0), "Last y coordinate should be 465.0"


def test_write_tps_single_with_semilandmarks_roundtrip(tmp_path):
    path = tmp_path / "single_semilandmarks.tps"

    landmarks = np.array([[0.0, 0.0], [1.0, 0.0]])
    semilandmarks = [np.array([[0.0, 0.0], [0.5, 0.2], [1.0, 0.0]])]

    write_tps(
        path,
        landmarks=landmarks,
        idx="specimen-1",
        semilandmarks=semilandmarks,
        comments="single specimen",
    )

    result = read_tps(path)
    assert isinstance(result, tuple)

    landmarks_out, curves_out = result
    np.testing.assert_allclose(landmarks_out, landmarks)
    assert len(curves_out) == 1
    np.testing.assert_allclose(curves_out[0], semilandmarks[0])

    text = path.read_text(encoding="utf-8")
    assert "CURVES=1\nPOINTS=3\n" in text


def test_write_tps_multi_specimens_with_semilandmarks_roundtrip(tmp_path):
    path = tmp_path / "multi_semilandmarks.tps"

    landmarks = [
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[0.0, 1.0], [1.0, 1.0]]),
    ]
    semilandmarks = [
        [
            np.array([[0.0, 0.0], [0.5, 0.2], [1.0, 0.0]]),
            np.array([[0.2, 0.1], [0.8, 0.1]]),
        ],
        [
            np.array([[0.0, 1.0], [0.5, 1.2], [1.0, 1.0]]),
            np.array([[0.2, 1.1], [0.8, 1.1]]),
        ],
    ]

    write_tps(
        path,
        landmarks=landmarks,
        image_path=["a.png", "b.png"],
        idx=["A", "B"],
        scale=[1.0, 2.0],
        semilandmarks=semilandmarks,
        comments=["first", "second"],
    )

    result = read_tps(path)
    assert isinstance(result, tuple)

    landmarks_out, curves_out = result
    assert landmarks_out.shape == (2, 2, 2)
    np.testing.assert_allclose(landmarks_out[0], landmarks[0])
    np.testing.assert_allclose(landmarks_out[1], landmarks[1])

    assert len(curves_out) == 2
    assert len(curves_out[0]) == 2
    assert len(curves_out[1]) == 2
    np.testing.assert_allclose(curves_out[0][0], semilandmarks[0][0])
    np.testing.assert_allclose(curves_out[1][1], semilandmarks[1][1])


def test_write_tps_multi_specimens_invalid_semilandmarks_length(tmp_path):
    path = tmp_path / "invalid_semilandmarks.tps"
    landmarks = [
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[0.0, 1.0], [1.0, 1.0]]),
    ]
    semilandmarks = [[np.array([[0.0, 0.0], [1.0, 0.0]])]]

    with pytest.raises(ValueError, match="same length as landmarks"):
        write_tps(path, landmarks=landmarks, semilandmarks=semilandmarks)


def test_read_tps_invalid_metadata_line_raises_value_error(tmp_path):
    path = tmp_path / "invalid_meta.tps"
    path.write_text(
        "LM=1\n0.0 0.0\nID=sample_1\nTHIS IS NOT A VALID KEY VALUE LINE\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid metadata line"):
        read_tps(path)
