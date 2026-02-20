"""Tests for _safe_extractall in ktch.datasets._base."""

import io
import zipfile

import pytest

from ktch.datasets._base import _safe_extractall


class TestSafeExtractall:
    """Tests for _safe_extractall()."""

    def test_normal_extraction(self, tmp_path):
        """Flat files should be extracted correctly."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("hello.txt", "hello world")
            zf.writestr("data.csv", "a,b\n1,2\n")
        buf.seek(0)

        with zipfile.ZipFile(buf, "r") as zf:
            _safe_extractall(zf, tmp_path)

        assert (tmp_path / "hello.txt").read_text() == "hello world"
        assert (tmp_path / "data.csv").read_text() == "a,b\n1,2\n"

    def test_nested_directory_extraction(self, tmp_path):
        """Subdirectories should be extracted correctly."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("subdir/nested.txt", "nested content")
        buf.seek(0)

        with zipfile.ZipFile(buf, "r") as zf:
            _safe_extractall(zf, tmp_path)

        assert (tmp_path / "subdir" / "nested.txt").read_text() == "nested content"

    def test_path_traversal_raises(self, tmp_path):
        """Relative path traversal (../) should raise ValueError."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            # Use ZipInfo to bypass Python 3.12+ member name sanitization
            info = zipfile.ZipInfo("../escape.txt")
            zf.writestr(info, "malicious")
        buf.seek(0)

        with zipfile.ZipFile(buf, "r") as zf:
            with pytest.raises(ValueError, match="path traversal"):
                _safe_extractall(zf, tmp_path)

        # Confirm no files were written
        assert list(tmp_path.iterdir()) == []

    def test_absolute_path_traversal_raises(self, tmp_path):
        """Absolute path in zip member should raise ValueError."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            info = zipfile.ZipInfo("/etc/passwd")
            zf.writestr(info, "malicious")
        buf.seek(0)

        with zipfile.ZipFile(buf, "r") as zf:
            with pytest.raises(ValueError, match="path traversal"):
                _safe_extractall(zf, tmp_path)

        # Confirm no files were written
        assert list(tmp_path.iterdir()) == []
