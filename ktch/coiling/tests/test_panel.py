"""Tests for the coiling panel input normalization in ``ktch.coiling._panel``."""

import numpy as np
import pandas as pd
import pytest

from ktch.coiling._panel import _check_panel, _Panel, _panel_to_flat

CHANNELS = ["x", "y", "z", "r"]


def _sample_values():
    """Two specimens with different point counts, four channels each."""
    s0 = np.array(
        [
            [0.0, 1.0, 2.0, 0.5],
            [3.0, 4.0, 5.0, 0.6],
            [6.0, 7.0, 8.0, 0.7],
        ]
    )
    s1 = np.array(
        [
            [10.0, 11.0, 12.0, 1.5],
            [13.0, 14.0, 15.0, 1.6],
        ]
    )
    return [s0, s1]


# --------------------------------------------------------------------------- #
# _check_panel: accepted encodings normalize identically
# --------------------------------------------------------------------------- #


def test_check_panel_from_list():
    values = _sample_values()
    panel = _check_panel(values, channel_names=CHANNELS)
    assert isinstance(panel, _Panel)
    assert panel.n_samples == 2
    assert panel.n_channels == 4
    assert panel.channel_names == CHANNELS
    np.testing.assert_array_equal(panel.values[0], values[0])
    np.testing.assert_array_equal(panel.values[1], values[1])


def test_check_panel_flat_roundtrip():
    values = _sample_values()
    flat = _panel_to_flat(values)
    # Channel-blocked, trailing-NaN padded to the common width.
    assert flat.shape == (2, 4 * 3)
    assert np.isnan(flat[1]).sum() == 4  # one missing point * four channels

    panel = _check_panel(flat, channel_names=CHANNELS)
    assert panel.n_samples == 2
    np.testing.assert_allclose(panel.values[0], values[0])
    np.testing.assert_allclose(panel.values[1], values[1])


def test_check_panel_wide_dataframe_matches_flat():
    values = _sample_values()
    flat = _panel_to_flat(values)
    df = pd.DataFrame(flat)
    panel = _check_panel(df, channel_names=CHANNELS)
    np.testing.assert_allclose(panel.values[0], values[0])
    np.testing.assert_allclose(panel.values[1], values[1])


def test_check_panel_long_dataframe():
    values = _sample_values()
    rows = []
    for spec_id, seq in enumerate(values):
        for point_id, row in enumerate(seq):
            rows.append(
                {
                    "specimen": spec_id,
                    "point": point_id,
                    "x": row[0],
                    "y": row[1],
                    "z": row[2],
                    "r": row[3],
                    "s": float(point_id),
                    "c": 0.3 + spec_id,
                    "b": 0.7 + spec_id,
                }
            )
    df = pd.DataFrame(rows).set_index(["specimen", "point"])

    panel = _check_panel(
        df,
        channel_names=CHANNELS,
        domain_coord_names=["s"],
        meta_names=["c", "b"],
    )
    assert panel.n_samples == 2
    np.testing.assert_allclose(panel.values[0], values[0])
    np.testing.assert_allclose(panel.values[1], values[1])

    assert panel.domain_coords is not None
    np.testing.assert_allclose(panel.domain_coords[0].ravel(), [0.0, 1.0, 2.0])
    assert panel.domain_coord_names == ["s"]
    np.testing.assert_allclose(panel.meta, [[0.3, 0.7], [1.3, 1.7]])
    assert panel.meta_names == ["c", "b"]


def test_check_panel_all_encodings_agree():
    values = _sample_values()
    from_list = _check_panel(values, channel_names=CHANNELS)
    from_flat = _check_panel(_panel_to_flat(values), channel_names=CHANNELS)
    for a, b in zip(from_list.values, from_flat.values):
        np.testing.assert_allclose(a, b)


# --------------------------------------------------------------------------- #
# _check_panel: domain coordinates and meta pass-through
# --------------------------------------------------------------------------- #


def test_check_panel_domain_coords_separate_arg():
    values = _sample_values()
    coords = [np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0])]
    panel = _check_panel(values, channel_names=CHANNELS, domain_coords=coords)
    assert panel.domain_coords is not None
    np.testing.assert_allclose(panel.domain_coords[1], [0.0, 1.0])


def test_check_panel_meta_separate_arg():
    values = _sample_values()
    meta = np.array([[0.3, 0.7], [1.3, 1.7]])
    panel = _check_panel(values, channel_names=CHANNELS, meta=meta)
    np.testing.assert_allclose(panel.meta, meta)


# --------------------------------------------------------------------------- #
# _check_panel: validation errors
# --------------------------------------------------------------------------- #


def test_check_panel_wrong_channel_count_raises():
    values = _sample_values()
    with pytest.raises(ValueError, match="n_channels"):
        _check_panel(values, channel_names=["x", "y"])


def test_check_panel_empty_channel_names_raises():
    with pytest.raises(ValueError, match="at least one channel"):
        _check_panel(_sample_values(), channel_names=[])


def test_check_panel_domain_coords_length_mismatch_raises():
    values = _sample_values()
    with pytest.raises(ValueError, match="same length"):
        _check_panel(
            values,
            channel_names=CHANNELS,
            domain_coords=[np.array([0.0, 1.0, 2.0])],
        )


def test_check_panel_domain_coords_rows_mismatch_raises():
    values = _sample_values()
    bad = [np.array([0.0, 1.0]), np.array([0.0, 1.0])]  # first should be 3
    with pytest.raises(ValueError, match="rows"):
        _check_panel(values, channel_names=CHANNELS, domain_coords=bad)


def test_check_panel_meta_rows_mismatch_raises():
    values = _sample_values()
    with pytest.raises(ValueError, match="samples"):
        _check_panel(values, channel_names=CHANNELS, meta=np.array([[0.3, 0.7]]))


def test_check_panel_domain_coord_names_with_list_raises():
    with pytest.raises(ValueError, match="domain_coord_names applies"):
        _check_panel(_sample_values(), channel_names=CHANNELS, domain_coord_names=["s"])


def test_check_panel_3d_array_not_implemented():
    arr = np.zeros((2, 3, 4))
    with pytest.raises(NotImplementedError, match="3D-array"):
        _check_panel(arr, channel_names=CHANNELS)


def test_check_panel_flat_layout_mismatch_raises():
    # A flat row whose valid length is not a multiple of n_channels.
    bad = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    with pytest.raises(ValueError, match="channel-blocked"):
        _check_panel(bad, channel_names=CHANNELS)


def test_check_panel_long_df_missing_column_raises():
    df = pd.DataFrame({"specimen": [0, 0], "point": [0, 1], "x": [1.0, 2.0]}).set_index(
        ["specimen", "point"]
    )
    with pytest.raises(ValueError, match="absent"):
        _check_panel(df, channel_names=CHANNELS)


def test_check_panel_long_df_three_levels_not_implemented():
    df = pd.DataFrame(
        {
            "specimen": [0, 0],
            "trajectory": [0, 0],
            "point": [0, 1],
            "x": [1.0, 2.0],
            "y": [1.0, 2.0],
            "z": [1.0, 2.0],
            "r": [0.5, 0.6],
        }
    ).set_index(["specimen", "trajectory", "point"])
    with pytest.raises(NotImplementedError, match="index levels"):
        _check_panel(df, channel_names=CHANNELS)


def test_check_panel_long_df_with_separate_args_raises():
    df = pd.DataFrame(
        {
            "specimen": [0, 0],
            "point": [0, 1],
            "x": [1.0, 2.0],
            "y": [1.0, 2.0],
            "z": [1.0, 2.0],
            "r": [0.5, 0.6],
        }
    ).set_index(["specimen", "point"])
    with pytest.raises(ValueError, match="select the domain coordinates"):
        _check_panel(
            df,
            channel_names=CHANNELS,
            domain_coords=[np.zeros((2, 1))],
        )
