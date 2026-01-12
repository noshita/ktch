"""Tests for the Passiflora leaf image dataset."""

from unittest.mock import patch

import pytest


class TestLoadImagePassifloraLeaves:
    """Tests for load_image_passiflora_leaves function."""

    def test_import_error_without_pooch(self):
        """Test that ImportError is raised when pooch is not installed."""
        with patch("ktch.datasets._base.pooch", None):
            from ktch.datasets._base import _fetch_remote_data

            with pytest.raises(
                ImportError, match="Missing optional dependency 'pooch'"
            ):
                _fetch_remote_data("test.zip", "0.7.0")

    def test_function_signature(self):
        """Test that the function exists and has the expected parameters."""
        import inspect

        from ktch.datasets import load_image_passiflora_leaves

        assert callable(load_image_passiflora_leaves)

        sig = inspect.signature(load_image_passiflora_leaves)
        params = list(sig.parameters.keys())

        assert "return_paths" in params
        assert "as_frame" in params
        assert "version" in params

    @pytest.mark.skip(reason="Requires actual data on remote server")
    def test_load_with_return_paths(self):
        """Test loading with return_paths=True."""
        from ktch.datasets import load_image_passiflora_leaves

        data = load_image_passiflora_leaves(return_paths=True)

        assert hasattr(data, "images")
        assert hasattr(data, "meta")
        assert hasattr(data, "DESCR")
        assert hasattr(data, "data_dir")
        assert hasattr(data, "version")
        assert all(isinstance(p, str) for p in data.images)

    @pytest.mark.skip(reason="Requires actual data on remote server")
    def test_load_as_numpy(self):
        """Test loading images as numpy arrays."""
        import numpy as np

        from ktch.datasets import load_image_passiflora_leaves

        data = load_image_passiflora_leaves(return_paths=False)

        assert hasattr(data, "images")
        assert all(isinstance(img, np.ndarray) for img in data.images)

    @pytest.mark.skip(reason="Requires actual data on remote server")
    def test_metadata_formats(self):
        """Test metadata is returned in correct format based on as_frame."""
        import pandas as pd

        from ktch.datasets import load_image_passiflora_leaves

        # as_frame=True -> DataFrame
        data = load_image_passiflora_leaves(return_paths=True, as_frame=True)
        assert isinstance(data.meta, pd.DataFrame)

        # as_frame=False -> dict
        data = load_image_passiflora_leaves(return_paths=True, as_frame=False)
        assert isinstance(data.meta, dict)


class TestRegistry:
    """Tests for the dataset registry."""

    def test_default_version_in_registry(self):
        """Test that the default version exists and has required datasets."""
        from ktch.datasets._base import _get_default_version
        from ktch.datasets._registry import (
            get_registry,
            get_url,
            method_files_map,
            versioned_registry,
        )

        version = _get_default_version()

        # Default version exists in registry
        assert version in versioned_registry, (
            f"Default version '{version}' not found. "
            f"Available: {list(versioned_registry.keys())}"
        )

        # Required dataset exists
        registry = get_registry(version)
        assert "image_passiflora_leaves.zip" in registry

        # URL is correctly formed
        url = get_url("image_passiflora_leaves.zip", version)
        assert f"releases/v{version}/image_passiflora_leaves.zip" in url

        # Method mapping exists
        assert "image_passiflora_leaves" in method_files_map

    def test_get_registry_invalid_version(self):
        """Test that get_registry raises ValueError for invalid version."""
        from ktch.datasets._registry import get_registry

        with pytest.raises(ValueError, match="not found"):
            get_registry("99.99.99")

    def test_all_registered_versions_valid(self):
        """Test that all registered versions have valid entries."""
        from ktch.datasets._registry import get_registry, get_url, versioned_registry

        for version in versioned_registry:
            registry = get_registry(version)
            assert isinstance(registry, dict)
            assert len(registry) > 0

            for filename, hash_value in registry.items():
                assert len(hash_value) == 64, f"Invalid SHA256 hash for {filename}"
                url = get_url(filename, version)
                assert url.startswith("https://")


class TestVersionDetection:
    """Tests for version detection logic."""

    def test_get_default_version_format(self):
        """Test that _get_default_version returns a valid X.Y.Z format."""
        from ktch.datasets._base import _get_default_version

        version = _get_default_version()
        parts = version.split(".")

        assert len(parts) == 3, f"Expected 'X.Y.Z' format, got '{version}'"
        assert all(part.isdigit() for part in parts), (
            f"Expected numeric version parts, got '{version}'"
        )

    def test_get_default_version_matches_package(self):
        """Test that _get_default_version derives from package version."""
        from importlib.metadata import version as get_package_version

        from ktch.datasets._base import _get_default_version

        pkg_version = get_package_version("ktch")
        default_version = _get_default_version()

        assert pkg_version.startswith(default_version) or default_version in pkg_version
