"""Tests for scripts/update_registry.py."""

import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the script module (not a package, so use importlib)
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _SCRIPTS_DIR / "update_registry.py"

spec = importlib.util.spec_from_file_location("update_registry", _SCRIPT_PATH)
update_registry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(update_registry)

read_registry_toml = update_registry.read_registry_toml
render_registry = update_registry.render_registry
validate_manifest = update_registry.validate_manifest
RegistryError = update_registry.RegistryError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_HASH_A = "a" * 64
SAMPLE_HASH_B = "b" * 64
SAMPLE_HASH_C = "c" * 64


def _exec_rendered(content):
    """Execute rendered _registry.py content and return its namespace."""
    namespace = {}
    exec(content, namespace)
    return namespace


# ---------------------------------------------------------------------------
# Tests for render_registry
# ---------------------------------------------------------------------------


class TestRenderRegistry:
    """Tests for render_registry()."""

    def test_output_is_valid_python(self):
        """Rendered content should be valid Python without syntax errors."""
        registry = {"ds": {"1": {"data.zip": SAMPLE_HASH_A}}}
        defaults = {"ds": "1"}
        content = render_registry(registry, defaults)
        # Should not raise
        compile(content, "_registry.py", "exec")

    def test_roundtrip_single_dataset(self):
        """dataset_registry should survive render -> exec roundtrip."""
        registry = {
            "image_passiflora_leaves": {
                "1": {"image_passiflora_leaves.zip": SAMPLE_HASH_A},
            },
        }
        defaults = {"image_passiflora_leaves": "1"}

        content = render_registry(registry, defaults)
        ns = _exec_rendered(content)

        assert ns["dataset_registry"] == registry
        assert ns["default_versions"] == defaults

    def test_roundtrip_multiple_versions(self):
        """Multiple versions of a dataset should all be preserved."""
        registry = {
            "ds": {
                "1": {"data.zip": SAMPLE_HASH_A},
                "2": {"data.zip": SAMPLE_HASH_B},
            },
        }
        defaults = {"ds": "2"}

        content = render_registry(registry, defaults)
        ns = _exec_rendered(content)

        assert ns["dataset_registry"] == registry
        assert ns["default_versions"] == defaults

    def test_roundtrip_multiple_datasets(self):
        """Multiple datasets should be preserved in roundtrip."""
        registry = {
            "image_passiflora_leaves": {
                "1": {"image_passiflora_leaves.zip": SAMPLE_HASH_A},
            },
            "outline_passiflora_leaves": {
                "1": {"outline_passiflora_leaves.zip": SAMPLE_HASH_B},
            },
        }
        defaults = {
            "image_passiflora_leaves": "1",
            "outline_passiflora_leaves": "1",
        }

        content = render_registry(registry, defaults)
        ns = _exec_rendered(content)

        assert ns["dataset_registry"] == registry
        assert ns["default_versions"] == defaults

    def test_empty_registry(self):
        """Empty registry should produce valid Python with empty dicts."""
        content = render_registry({}, {})
        ns = _exec_rendered(content)

        assert ns["dataset_registry"] == {}
        assert ns["default_versions"] == {}

    def test_versions_sorted_numerically(self):
        """Versions should appear in numerically sorted order."""
        registry = {
            "ds": {
                "10": {"a.zip": SAMPLE_HASH_A},
                "2": {"a.zip": SAMPLE_HASH_B},
                "1": {"a.zip": SAMPLE_HASH_C},
            },
        }
        defaults = {"ds": "10"}

        content = render_registry(registry, defaults)
        ns = _exec_rendered(content)

        # dict preserves insertion order (Python 3.7+); verify numeric sort
        assert list(ns["dataset_registry"]["ds"].keys()) == ["1", "2", "10"]

    def test_datasets_sorted_alphabetically(self):
        """Datasets should appear in alphabetically sorted order."""
        registry = {
            "z_data": {"1": {"z_data.zip": SAMPLE_HASH_A}},
            "a_data": {"1": {"a_data.zip": SAMPLE_HASH_B}},
        }
        defaults = {"a_data": "1", "z_data": "1"}

        content = render_registry(registry, defaults)

        pos_a = content.index('"a_data"')
        pos_z = content.index('"z_data"')
        assert pos_a < pos_z

    def test_filenames_sorted_within_version(self):
        """Filenames within a version should be sorted."""
        registry = {
            "ds": {
                "1": {
                    "z_data.zip": SAMPLE_HASH_A,
                    "a_data.zip": SAMPLE_HASH_B,
                },
            },
        }
        defaults = {"ds": "1"}

        content = render_registry(registry, defaults)

        pos_a = content.index('"a_data.zip"')
        pos_z = content.index('"z_data.zip"')
        assert pos_a < pos_z

    def test_no_functions_in_output(self):
        """Rendered content should be pure data with no function definitions."""
        registry = {"ds": {"1": {"data.zip": SAMPLE_HASH_A}}}
        defaults = {"ds": "1"}

        content = render_registry(registry, defaults)

        assert "def " not in content

    def test_base_url_present(self):
        """Rendered content should include BASE_URL."""
        registry = {"ds": {"1": {"data.zip": SAMPLE_HASH_A}}}
        defaults = {"ds": "1"}

        content = render_registry(registry, defaults)
        ns = _exec_rendered(content)

        assert "BASE_URL" in ns
        assert ns["BASE_URL"].startswith("https://")

    def test_auto_generated_comment(self):
        """Rendered content should include auto-generation comment."""
        content = render_registry({}, {})
        assert "auto-generated" in content
        assert "registry.toml" in content


# ---------------------------------------------------------------------------
# Tests for validate_manifest
# ---------------------------------------------------------------------------


class TestValidateManifest:
    """Tests for validate_manifest()."""

    def test_valid_manifest(self):
        """Valid manifest should not raise."""
        manifest = {"data.zip": SAMPLE_HASH_A}
        # Should not raise
        validate_manifest(manifest)

    def test_invalid_hash_too_short(self):
        """Hash shorter than 64 chars should raise RegistryError."""
        manifest = {"data.zip": "abc123"}
        with pytest.raises(RegistryError):
            validate_manifest(manifest)

    def test_invalid_hash_uppercase(self):
        """Uppercase hex should be rejected (SHA256 hashes are lowercase)."""
        manifest = {"data.zip": "A" * 64}
        with pytest.raises(RegistryError):
            validate_manifest(manifest)

    def test_invalid_hash_non_hex(self):
        """Non-hex characters should be rejected."""
        manifest = {"data.zip": "g" * 64}
        with pytest.raises(RegistryError):
            validate_manifest(manifest)

    def test_non_string_hash_raises(self):
        """Non-string hash value should raise RegistryError."""
        manifest = {"data.zip": None}
        with pytest.raises(RegistryError, match="expected string hash"):
            validate_manifest(manifest)

    def test_integer_hash_raises(self):
        """Integer hash value should raise RegistryError."""
        manifest = {"data.zip": 12345}
        with pytest.raises(RegistryError, match="expected string hash"):
            validate_manifest(manifest)

    def test_empty_manifest(self):
        """Empty manifest should pass validation."""
        validate_manifest({})


# ---------------------------------------------------------------------------
# Tests for read_registry_toml
# ---------------------------------------------------------------------------


def _write_toml(tmp_path, content, monkeypatch):
    """Write a TOML file and patch the module to read from it."""
    toml_path = tmp_path / "registry.toml"
    toml_path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(update_registry, "REGISTRY_TOML_PATH", toml_path)


class TestReadRegistryToml:
    """Tests for read_registry_toml() validation logic."""

    def test_valid_config(self, tmp_path, monkeypatch):
        """Valid TOML should parse without errors."""
        _write_toml(
            tmp_path,
            '[my_dataset]\ndefault = "1"\nversions = ["1", "2"]\n',
            monkeypatch,
        )
        datasets, defaults = read_registry_toml()
        assert datasets == {"my_dataset": ["1", "2"]}
        assert defaults == {"my_dataset": "1"}

    def test_invalid_dataset_name_raises(self, tmp_path, monkeypatch):
        """Dataset name with uppercase or special chars should raise."""
        _write_toml(
            tmp_path,
            '[InvalidName]\nversions = ["1"]\n',
            monkeypatch,
        )
        with pytest.raises(RegistryError, match="invalid dataset name"):
            read_registry_toml()

    def test_invalid_version_string_raises(self, tmp_path, monkeypatch):
        """Non-integer version string should raise."""
        _write_toml(
            tmp_path,
            '[my_dataset]\nversions = ["v1"]\n',
            monkeypatch,
        )
        with pytest.raises(RegistryError, match="invalid version"):
            read_registry_toml()

    def test_default_not_in_versions_raises(self, tmp_path, monkeypatch):
        """Default version not in versions list should raise."""
        _write_toml(
            tmp_path,
            '[my_dataset]\ndefault = "3"\nversions = ["1", "2"]\n',
            monkeypatch,
        )
        with pytest.raises(RegistryError, match="not in versions list"):
            read_registry_toml()

    def test_empty_versions_skipped(self, tmp_path, monkeypatch, capsys):
        """Dataset with no versions should be skipped with a warning."""
        _write_toml(
            tmp_path,
            '[my_dataset]\nversions = []\n',
            monkeypatch,
        )
        datasets, defaults = read_registry_toml()
        assert datasets == {}
        assert defaults == {}
        assert "no versions listed" in capsys.readouterr().err

    def test_no_default_uses_none(self, tmp_path, monkeypatch):
        """Dataset without default key should have no entry in defaults."""
        _write_toml(
            tmp_path,
            '[my_dataset]\nversions = ["1"]\n',
            monkeypatch,
        )
        datasets, defaults = read_registry_toml()
        assert datasets == {"my_dataset": ["1"]}
        assert defaults == {}


# ---------------------------------------------------------------------------
# Registry sync check (registry.toml vs _registry.py)
# ---------------------------------------------------------------------------


class TestRegistrySync:
    """Verify that registry.toml and _registry.py are structurally in sync."""

    def test_datasets_and_versions_match(self):
        """Dataset names and version sets in TOML must match _registry.py."""
        from ktch.datasets._registry import (
            dataset_registry,
            default_versions,
        )

        toml_datasets, toml_defaults = read_registry_toml()

        # Dataset names must match
        assert set(toml_datasets.keys()) == set(dataset_registry.keys()), (
            "Dataset names in registry.toml and _registry.py differ. "
            "Run: uv run python scripts/update_registry.py"
        )

        # Versions per dataset must match
        for ds_name, toml_versions in toml_datasets.items():
            registry_versions = set(dataset_registry[ds_name].keys())
            assert set(toml_versions) == registry_versions, (
                f"Versions for '{ds_name}' differ between registry.toml "
                f"{sorted(toml_versions)} and _registry.py "
                f"{sorted(registry_versions)}. "
                "Run: uv run python scripts/update_registry.py"
            )

        # Default versions must match
        assert toml_defaults == default_versions, (
            "Default versions differ between registry.toml and _registry.py. "
            "Run: uv run python scripts/update_registry.py"
        )
