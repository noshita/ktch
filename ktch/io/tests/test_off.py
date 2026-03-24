"""Tests for the private OFF parser."""

import numpy as np
import pytest

from ktch.io._off import _read_off


@pytest.fixture()
def tri_off_file(tmp_path):
    """Create a minimal triangular OFF file (tetrahedron)."""
    content = """\
OFF
4 4 0
0.0 0.0 0.0
1.0 0.0 0.0
0.5 1.0 0.0
0.5 0.5 1.0
3 0 1 2
3 0 1 3
3 1 2 3
3 0 2 3
"""
    p = tmp_path / "tetra.off"
    p.write_text(content)
    return p


@pytest.fixture()
def quad_off_file(tmp_path):
    """Create an OFF file with a quad face (non-triangular)."""
    content = """\
OFF
4 1 0
0.0 0.0 0.0
1.0 0.0 0.0
1.0 1.0 0.0
0.0 1.0 0.0
4 0 1 2 3
"""
    p = tmp_path / "quad.off"
    p.write_text(content)
    return p


class TestReadOff:
    """Tests for _read_off."""

    def test_valid_triangular(self, tri_off_file):
        vertices, faces = _read_off(tri_off_file)

        assert vertices.shape == (4, 3)
        assert vertices.dtype == np.float64
        assert faces.shape == (4, 3)
        assert faces.dtype == np.int64

    def test_vertex_values(self, tri_off_file):
        vertices, _ = _read_off(tri_off_file)

        np.testing.assert_array_equal(vertices[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(vertices[1], [1.0, 0.0, 0.0])

    def test_face_indices(self, tri_off_file):
        _, faces = _read_off(tri_off_file)

        np.testing.assert_array_equal(faces[0], [0, 1, 2])
        np.testing.assert_array_equal(faces[1], [0, 1, 3])

    def test_invalid_header(self, tmp_path):
        p = tmp_path / "bad.off"
        p.write_text("COFF\n4 4 0\n")

        with pytest.raises(ValueError, match="Expected 'OFF' header"):
            _read_off(p)

    def test_non_triangular_faces(self, quad_off_file):
        with pytest.raises(ValueError, match="triangular"):
            _read_off(quad_off_file)

    def test_accepts_path_like(self, tri_off_file):
        """Ensure both str and Path objects work."""
        v1, f1 = _read_off(str(tri_off_file))
        v2, f2 = _read_off(tri_off_file)

        np.testing.assert_array_equal(v1, v2)
        np.testing.assert_array_equal(f1, f2)
