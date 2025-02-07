from pathlib import Path

import pytest

from ktch.io import read_tps


def test_read_tps_shape():
    path = Path(__file__).parent / "data" / "landmarks_triangle.tps"

    landmarks = read_tps(path)

    assert landmarks.shape == (50, 3, 2)
