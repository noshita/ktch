from pathlib import Path

import numpy as np
import pytest

from ktch.io import read_tps, write_tps
from ktch.io._tps import TPSData


def test_read_tps_returns_list_of_tpsdata():
    path = Path(__file__).parent / "data" / "landmarks_triangle.tps"

    result = read_tps(path)

    assert isinstance(result, list)
    assert len(result) == 50
    assert isinstance(result[0], TPSData)
    assert result[0].to_numpy().shape == (3, 2)


def test_read_tps_with_curves():
    """Test that read_tps correctly handles TPS files with CURVES data."""
    path = Path(__file__).parent / "data" / "sample_curve_data.tps"

    result = read_tps(path)

    assert isinstance(result, TPSData)
    assert result.to_numpy().shape == (1, 2)
    assert np.isclose(result.to_numpy()[0, 0], 150.0)
    assert np.isclose(result.to_numpy()[0, 1], 250.0)

    assert result.curves is not None
    assert len(result.curves) == 1
    assert isinstance(result.curves[0], np.ndarray)
    assert result.curves[0].shape == (50, 2)

    assert np.isclose(result.curves[0][0, 0], 320.0)
    assert np.isclose(result.curves[0][0, 1], 145.0)
    assert np.isclose(result.curves[0][49, 0], 375.0)
    assert np.isclose(result.curves[0][49, 1], 465.0)


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
    assert isinstance(result, TPSData)
    np.testing.assert_allclose(result.to_numpy(), landmarks)
    assert result.curves is not None
    assert len(result.curves) == 1
    np.testing.assert_allclose(result.curves[0], semilandmarks[0])

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
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], TPSData)

    np.testing.assert_allclose(result[0].to_numpy(), landmarks[0])
    np.testing.assert_allclose(result[1].to_numpy(), landmarks[1])

    assert result[0].curves is not None
    assert len(result[0].curves) == 2
    assert len(result[1].curves) == 2
    np.testing.assert_allclose(result[0].curves[0], semilandmarks[0][0])
    np.testing.assert_allclose(result[1].curves[1], semilandmarks[1][1])


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


def test_specimen_name_newline_raises():
    with pytest.raises(ValueError, match="specimen_name must not contain newline"):
        TPSData(specimen_name="bad\nname", landmarks=np.zeros((3, 2)))


def test_read_tps_scale_parsed_as_float(tmp_path):
    path = tmp_path / "scaled.tps"
    path.write_text("LM=1\n0.0 0.0\nID=sample_1\nSCALE=0.0123\n", encoding="utf-8")

    result = read_tps(path)
    assert result.scale == pytest.approx(0.0123)
    assert isinstance(result.scale, float)


def test_read_tps_invalid_scale_raises(tmp_path):
    path = tmp_path / "bad_scale.tps"
    path.write_text(
        "LM=1\n0.0 0.0\nID=sample_1\nSCALE=not_a_number\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Invalid SCALE value"):
        read_tps(path)


def test_read_tps_landmark_count_mismatch_raises(tmp_path):
    # LM declares 3 landmarks but only 2 coordinate rows follow.
    path = tmp_path / "count_mismatch.tps"
    path.write_text("LM=3\n0.0 0.0\n1.0 1.0\nID=sample_1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="coordinate count mismatch"):
        read_tps(path)


def test_read_tps_latin1_specimen_name(tmp_path):
    # Files saved by Windows tools may be Latin-1; reading must not depend on
    # the platform locale and must recover non-ASCII names.
    path = tmp_path / "latin1.tps"
    path.write_bytes("LM=1\n0.0 0.0\nID=sp\xe9cimen\n".encode("latin-1"))

    result = read_tps(path)
    assert result.specimen_name == "sp\xe9cimen"


def test_read_tps_lenient_count_mismatch_warns(tmp_path):
    # strict=False: warn and read the rows actually present.
    path = tmp_path / "count_mismatch_lenient.tps"
    path.write_text("LM=3\n0.0 0.0\n1.0 1.0\nID=sample_1\n", encoding="utf-8")

    with pytest.warns(UserWarning, match="coordinate count mismatch"):
        result = read_tps(path, strict=False)
    assert result.to_numpy().shape == (2, 2)


def test_read_tps_lenient_invalid_scale_warns(tmp_path):
    # strict=False: warn and fall back to scale=None.
    path = tmp_path / "bad_scale_lenient.tps"
    path.write_text(
        "LM=1\n0.0 0.0\nID=sample_1\nSCALE=not_a_number\n", encoding="utf-8"
    )

    with pytest.warns(UserWarning, match="Invalid SCALE"):
        result = read_tps(path, strict=False)
    assert result.scale is None


def test_read_tps_lenient_still_raises_on_missing_id(tmp_path):
    # A missing ID is not recoverable and must raise even when strict=False.
    path = tmp_path / "no_id_lenient.tps"
    path.write_text("LM=1\n0.0 0.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="ID"):
        read_tps(path, strict=False)


def test_read_tps_lenient_still_raises_on_bad_row(tmp_path):
    # A malformed coordinate row cannot be recovered and must raise even
    # when strict=False.
    path = tmp_path / "bad_row_lenient.tps"
    path.write_text("LM=1\nabc def\nID=sample_1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid coordinate row"):
        read_tps(path, strict=False)
