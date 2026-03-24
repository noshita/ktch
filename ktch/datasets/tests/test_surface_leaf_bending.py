"""Tests for the synthetic 3D leaf bending surface dataset."""

import os
from unittest.mock import patch

import pytest

_skip_network = pytest.mark.skipif(
    os.environ.get("KTCH_NETWORK_TESTS") != "1",
    reason="Set KTCH_NETWORK_TESTS=1 to run network tests",
)


class TestLoadSurfaceLeafBending:
    """Tests for load_surface_leaf_bending function."""

    def test_import_error_without_pooch(self):
        """Test that ImportError is raised when pooch is not installed."""
        with patch("ktch.datasets._base.pooch", None):
            from ktch.datasets._base import _fetch_remote_dataset

            with pytest.raises(
                ImportError, match="Missing optional dependency 'pooch'"
            ):
                _fetch_remote_dataset(
                    "surface_leaf_bending", "1", "surface_leaf_bending.zip"
                )

    def test_function_signature(self):
        """Test that the function exists and has the expected parameters."""
        import inspect

        from ktch.datasets import load_surface_leaf_bending

        assert callable(load_surface_leaf_bending)

        sig = inspect.signature(load_surface_leaf_bending)
        params = list(sig.parameters.keys())

        assert "return_paths" in params
        assert "as_frame" in params
        assert "version" in params

    @_skip_network
    def test_load_default(self):
        """Test default loading returns correct types and shapes."""
        import numpy as np

        from ktch.datasets import load_surface_leaf_bending

        data = load_surface_leaf_bending()

        assert isinstance(data.vertices, list)
        assert len(data.vertices) == 60
        assert isinstance(data.vertices[0], np.ndarray)
        assert data.vertices[0].ndim == 2
        assert data.vertices[0].shape[1] == 3
        assert data.vertices[0].dtype == np.float64

        assert isinstance(data.faces, list)
        assert len(data.faces) == 60
        assert isinstance(data.faces[0], np.ndarray)
        assert data.faces[0].shape[1] == 3
        assert data.faces[0].dtype == np.int64

        assert isinstance(data.param_vertices, list)
        assert len(data.param_vertices) == 60
        assert data.param_vertices[0].shape[1] == 3

        assert isinstance(data.param_faces, list)
        assert len(data.param_faces) == 60

        assert isinstance(data.meta, dict)

    @_skip_network
    def test_load_with_return_paths(self):
        """Test loading with return_paths=True."""
        from ktch.datasets import load_surface_leaf_bending

        data = load_surface_leaf_bending(return_paths=True)

        assert hasattr(data, "surface_paths")
        assert hasattr(data, "param_paths")
        assert len(data.surface_paths) == 60
        assert len(data.param_paths) == 60
        assert all(isinstance(p, str) for p in data.surface_paths)
        assert all(p.endswith(".off") for p in data.surface_paths)

        # Should not have array attributes
        assert not hasattr(data, "vertices")
        assert not hasattr(data, "faces")

    @_skip_network
    def test_metadata_formats(self):
        """Test metadata is returned in correct format based on as_frame."""
        import pandas as pd

        from ktch.datasets import load_surface_leaf_bending

        data = load_surface_leaf_bending(return_paths=True, as_frame=True)
        assert isinstance(data.meta, pd.DataFrame)

        data = load_surface_leaf_bending(return_paths=True, as_frame=False)
        assert isinstance(data.meta, dict)

    @_skip_network
    def test_bunch_keys_default(self):
        """Test that Bunch contains all expected keys for default loading."""
        from ktch.datasets import load_surface_leaf_bending

        data = load_surface_leaf_bending()

        expected_keys = {
            "vertices",
            "faces",
            "param_vertices",
            "param_faces",
            "meta",
            "DESCR",
            "data_dir",
            "version",
        }
        assert set(data.keys()) == expected_keys

    @_skip_network
    def test_bunch_keys_return_paths(self):
        """Test that Bunch contains all expected keys with return_paths."""
        from ktch.datasets import load_surface_leaf_bending

        data = load_surface_leaf_bending(return_paths=True)

        expected_keys = {
            "surface_paths",
            "param_paths",
            "meta",
            "DESCR",
            "data_dir",
            "version",
        }
        assert set(data.keys()) == expected_keys

    @_skip_network
    def test_descr_not_empty(self):
        """Test that the description is not empty."""
        from ktch.datasets import load_surface_leaf_bending

        data = load_surface_leaf_bending(return_paths=True)

        assert isinstance(data.DESCR, str)
        assert len(data.DESCR) > 0

    @_skip_network
    def test_param_mesh_on_unit_disk(self):
        """Test that parameter mesh vertices lie on the unit disk (z=0)."""
        import numpy as np

        from ktch.datasets import load_surface_leaf_bending

        data = load_surface_leaf_bending()

        for pv in data.param_vertices:
            # z coordinates should all be zero
            np.testing.assert_array_equal(pv[:, 2], 0.0)

            # all points within (or on) the unit disk
            r = np.sqrt(pv[:, 0] ** 2 + pv[:, 1] ** 2)
            assert np.all(r <= 1.0 + 1e-10)


class TestSurfaceLeafBendingRegistry:
    """Tests for the surface_leaf_bending registry entry."""

    def test_default_version_in_registry(self):
        """Test that the default version resolves to a valid registry entry."""
        from ktch.datasets._base import (
            get_dataset_hash,
            get_dataset_url,
            get_default_version,
        )
        from ktch.datasets._registry import dataset_registry

        ds_name = "surface_leaf_bending"
        version = get_default_version(ds_name)

        assert ds_name in dataset_registry
        assert version in dataset_registry[ds_name]

        hash_value = get_dataset_hash(
            ds_name, version, "surface_leaf_bending.zip"
        )
        assert len(hash_value) == 64

        url = get_dataset_url(ds_name, version, "surface_leaf_bending.zip")
        assert f"datasets/{ds_name}/v{version}/" in url
