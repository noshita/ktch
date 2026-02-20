"""Tests for the Passiflora leaf image dataset."""

import os
from unittest.mock import patch

import pytest

_skip_network = pytest.mark.skipif(
    os.environ.get("KTCH_NETWORK_TESTS") != "1",
    reason="Set KTCH_NETWORK_TESTS=1 to run network tests",
)


class TestLoadImagePassifloraLeaves:
    """Tests for load_image_passiflora_leaves function."""

    def test_import_error_without_pooch(self):
        """Test that ImportError is raised when pooch is not installed."""
        with patch("ktch.datasets._base.pooch", None):
            from ktch.datasets._base import _fetch_remote_dataset

            with pytest.raises(
                ImportError, match="Missing optional dependency 'pooch'"
            ):
                _fetch_remote_dataset(
                    "image_passiflora_leaves", "1", "image_passiflora_leaves.zip"
                )

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

    @_skip_network
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

    @_skip_network
    def test_load_as_numpy(self):
        """Test loading images as numpy arrays."""
        import numpy as np

        from ktch.datasets import load_image_passiflora_leaves

        data = load_image_passiflora_leaves(return_paths=False)

        assert hasattr(data, "images")
        assert all(isinstance(img, np.ndarray) for img in data.images)

    @_skip_network
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
        """Test that the default version resolves to a valid registry entry."""
        from ktch.datasets._base import (
            get_dataset_hash,
            get_dataset_url,
            get_default_version,
        )
        from ktch.datasets._registry import (
            dataset_registry,
            default_versions,
        )

        ds_name = "image_passiflora_leaves"
        version = get_default_version(ds_name)

        # Default version exists in dataset_registry
        assert ds_name in dataset_registry
        assert version in dataset_registry[ds_name]

        # Hash is retrievable
        hash_value = get_dataset_hash(
            ds_name, version, "image_passiflora_leaves.zip"
        )
        assert len(hash_value) == 64

        # URL is correctly formed
        url = get_dataset_url(ds_name, version, "image_passiflora_leaves.zip")
        assert f"datasets/{ds_name}/v{version}/" in url

        # All datasets in default_versions exist in dataset_registry
        for name, ver in default_versions.items():
            assert name in dataset_registry
            assert ver in dataset_registry[name]

    def test_get_dataset_hash_invalid_dataset(self):
        """Test that get_dataset_hash raises ValueError for unknown dataset."""
        from ktch.datasets._base import get_dataset_hash

        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset_hash("nonexistent", "1", "file.zip")

    def test_get_dataset_hash_invalid_version(self):
        """Test that get_dataset_hash raises ValueError for invalid version."""
        from ktch.datasets._base import get_dataset_hash

        with pytest.raises(ValueError, match="not found"):
            get_dataset_hash("image_passiflora_leaves", "999", "file.zip")

    def test_all_registered_versions_valid(self):
        """Test that all registered versions have valid entries."""
        from ktch.datasets._base import get_dataset_url
        from ktch.datasets._registry import dataset_registry

        for ds_name, versions in dataset_registry.items():
            for version, files in versions.items():
                assert isinstance(files, dict)
                assert len(files) > 0

                for filename, hash_value in files.items():
                    assert len(hash_value) == 64, (
                        f"Invalid SHA256 hash for {ds_name}/v{version}/{filename}"
                    )
                    url = get_dataset_url(ds_name, version, filename)
                    assert url.startswith("https://")

    def test_get_available_versions(self):
        """Test that get_available_versions returns sorted versions."""
        from ktch.datasets._base import get_available_versions

        versions = get_available_versions("image_passiflora_leaves")
        assert isinstance(versions, list)
        assert len(versions) > 0
        # Should be sorted numerically
        assert versions == sorted(versions, key=int)


class TestResolveDatasetVersion:
    """Tests for _resolve_dataset_version."""

    def test_default_version(self):
        """Test that None resolves to the default version."""
        from ktch.datasets._base import _resolve_dataset_version
        from ktch.datasets._registry import default_versions

        ds_name = "image_passiflora_leaves"
        result = _resolve_dataset_version(ds_name)
        assert result == default_versions[ds_name]

    def test_explicit_version(self):
        """Test that an explicit valid version is returned as-is."""
        from ktch.datasets._base import _resolve_dataset_version
        from ktch.datasets._registry import dataset_registry

        ds_name = "image_passiflora_leaves"
        for version in dataset_registry[ds_name]:
            assert _resolve_dataset_version(ds_name, version) == version

    def test_unknown_dataset(self):
        """Test that ValueError is raised for unknown dataset."""
        from ktch.datasets._base import _resolve_dataset_version

        with pytest.raises(ValueError, match="Unknown dataset"):
            _resolve_dataset_version("nonexistent", "1")

    def test_invalid_version(self):
        """Test that ValueError is raised for non-existent version."""
        from ktch.datasets._base import _resolve_dataset_version

        with pytest.raises(ValueError, match="not found"):
            _resolve_dataset_version("image_passiflora_leaves", "999")

    def test_url_uses_dataset_version(self):
        """Test that the URL uses the dataset-specific version path."""
        from ktch.datasets._base import (
            _resolve_dataset_version,
            get_dataset_url,
        )

        ds_name = "image_passiflora_leaves"
        version = _resolve_dataset_version(ds_name)
        url = get_dataset_url(ds_name, version, "image_passiflora_leaves.zip")
        assert f"datasets/{ds_name}/v{version}/" in url
