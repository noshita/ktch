"""Tests for ktch.datasets.fetch (example data)."""

import os
import tempfile
from unittest.mock import patch

import pytest

from ktch.datasets import fetch


class TestFetchBundled:
    """Tests for bundled example files."""

    def test_returns_existing_path(self):
        path = fetch("landmarks_triangle.tps")
        assert os.path.isfile(path)

    def test_returns_tps_extension(self):
        path = fetch("landmarks_triangle.tps")
        assert path.endswith("landmarks_triangle.tps")

    def test_readable_by_read_tps(self):
        from ktch.io import read_tps

        path = fetch("landmarks_triangle.tps")
        df = read_tps(path, as_frame=True)
        assert len(df) > 0

    def test_data_home_copies_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = fetch("landmarks_triangle.tps", data_home=tmpdir)
            assert path == os.path.join(tmpdir, "landmarks_triangle.tps")
            assert os.path.isfile(path)

    def test_data_home_skips_if_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = fetch("landmarks_triangle.tps", data_home=tmpdir)
            mtime1 = os.path.getmtime(path1)
            path2 = fetch("landmarks_triangle.tps", data_home=tmpdir)
            mtime2 = os.path.getmtime(path2)
            assert path1 == path2
            assert mtime1 == mtime2

    def test_data_home_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "a", "b")
            path = fetch("landmarks_triangle.tps", data_home=nested)
            assert os.path.isfile(path)

    def test_version_ignored_for_bundled(self):
        path = fetch("landmarks_triangle.tps", version="99")
        assert os.path.isfile(path)


class TestFetchRemote:
    """Tests for remote example file error handling."""

    def test_import_error_without_pooch(self):
        with patch("ktch.datasets._examples.pooch", None):
            with pytest.raises(
                ImportError, match="Missing optional dependency 'pooch'"
            ):
                fetch("danshaku_08_allSegments_para.vtp")

    def test_unknown_version_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown version"):
            fetch("danshaku_08_allSegments_para.vtp", version="99")


class TestFetchValidation:
    """Tests for fetch input validation."""

    def test_unknown_name_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown example file"):
            fetch("nonexistent.xyz")

    def test_error_message_lists_available(self):
        with pytest.raises(ValueError, match="landmarks_triangle.tps"):
            fetch("nonexistent.xyz")

    def test_function_signature(self):
        import inspect

        sig = inspect.signature(fetch)
        params = list(sig.parameters.keys())
        assert "name" in params
        assert "data_home" in params
        assert "version" in params
