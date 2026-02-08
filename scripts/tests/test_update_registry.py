"""Tests for scripts/update_registry.py.

Covers render_registry(), build_method_files_map(), and validate_manifest().
"""

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

render_registry = update_registry.render_registry
build_method_files_map = update_registry.build_method_files_map
validate_manifest = update_registry.validate_manifest


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
        registry = {"0.7.0": {"data.zip": SAMPLE_HASH_A}}
        mfm = {"data": ["data.zip"]}
        content = render_registry(registry, mfm)
        # Should not raise
        compile(content, "_registry.py", "exec")

    def test_roundtrip_single_version(self):
        """versioned_registry should survive render -> exec roundtrip."""
        registry = {"0.7.0": {"image_passiflora_leaves.zip": SAMPLE_HASH_A}}
        mfm = {"image_passiflora_leaves": ["image_passiflora_leaves.zip"]}

        content = render_registry(registry, mfm)
        ns = _exec_rendered(content)

        assert ns["versioned_registry"] == registry
        assert ns["method_files_map"] == mfm

    def test_roundtrip_multiple_versions(self):
        """Multiple versions should all be preserved in roundtrip."""
        registry = {
            "0.7.0": {"data_a.zip": SAMPLE_HASH_A},
            "0.8.0": {"data_a.zip": SAMPLE_HASH_B},
        }
        mfm = {"data_a": ["data_a.zip"]}

        content = render_registry(registry, mfm)
        ns = _exec_rendered(content)

        assert ns["versioned_registry"] == registry

    def test_roundtrip_multiple_datasets(self):
        """Multiple datasets in a single version should be preserved."""
        registry = {
            "0.7.0": {
                "image_passiflora_leaves.zip": SAMPLE_HASH_A,
                "outline_passiflora_leaves.zip": SAMPLE_HASH_B,
            },
        }
        mfm = {
            "image_passiflora_leaves": ["image_passiflora_leaves.zip"],
            "outline_passiflora_leaves": ["outline_passiflora_leaves.zip"],
        }

        content = render_registry(registry, mfm)
        ns = _exec_rendered(content)

        assert ns["versioned_registry"] == registry
        assert ns["method_files_map"] == mfm

    def test_empty_registry(self):
        """Empty registry should produce valid Python with empty dicts."""
        content = render_registry({}, {})
        ns = _exec_rendered(content)

        assert ns["versioned_registry"] == {}
        assert ns["method_files_map"] == {}

    def test_versions_sorted(self):
        """Versions should appear in sorted order in the rendered output."""
        registry = {
            "0.10.0": {"a.zip": SAMPLE_HASH_A},
            "0.7.0": {"a.zip": SAMPLE_HASH_B},
            "0.8.0": {"a.zip": SAMPLE_HASH_C},
        }
        mfm = {"a": ["a.zip"]}

        content = render_registry(registry, mfm)

        # Find positions of version strings in output
        pos_070 = content.index('"0.7.0"')
        pos_080 = content.index('"0.8.0"')
        pos_0100 = content.index('"0.10.0"')
        assert pos_0100 < pos_070 < pos_080  # lexicographic: "0.10.0" < "0.7.0" < "0.8.0"

    def test_filenames_sorted_within_version(self):
        """Filenames within a version should be sorted."""
        registry = {
            "0.7.0": {
                "z_data.zip": SAMPLE_HASH_A,
                "a_data.zip": SAMPLE_HASH_B,
            },
        }
        mfm = {"a_data": ["a_data.zip"], "z_data": ["z_data.zip"]}

        content = render_registry(registry, mfm)

        pos_a = content.index('"a_data.zip"')
        pos_z = content.index('"z_data.zip"')
        assert pos_a < pos_z

    def test_functions_present(self):
        """Rendered content should include get_registry and get_url functions."""
        registry = {"0.7.0": {"data.zip": SAMPLE_HASH_A}}
        mfm = {"data": ["data.zip"]}

        content = render_registry(registry, mfm)
        ns = _exec_rendered(content)

        assert callable(ns["get_registry"])
        assert callable(ns["get_url"])

    def test_get_registry_works_after_render(self):
        """get_registry() should work correctly in rendered output."""
        registry = {"0.7.0": {"data.zip": SAMPLE_HASH_A}}
        mfm = {"data": ["data.zip"]}

        content = render_registry(registry, mfm)
        ns = _exec_rendered(content)

        result = ns["get_registry"]("0.7.0")
        assert result == {"data.zip": SAMPLE_HASH_A}

        with pytest.raises(ValueError, match="not found"):
            ns["get_registry"]("99.99.99")

    def test_get_url_works_after_render(self):
        """get_url() should produce correct URLs in rendered output."""
        registry = {"0.7.0": {"data.zip": SAMPLE_HASH_A}}
        mfm = {"data": ["data.zip"]}

        content = render_registry(registry, mfm)
        ns = _exec_rendered(content)

        url = ns["get_url"]("data.zip", "0.7.0")
        assert url == (
            "https://pub-c1d6dba6c94843f88f0fd096d19c0831.r2.dev"
            "/releases/v0.7.0/data.zip"
        )


# ---------------------------------------------------------------------------
# Tests for build_method_files_map
# ---------------------------------------------------------------------------


class TestBuildMethodFilesMap:
    """Tests for build_method_files_map()."""

    def test_single_dataset(self):
        """Single dataset across one version."""
        registry = {"0.7.0": {"image_passiflora_leaves.zip": SAMPLE_HASH_A}}
        result = build_method_files_map(registry)
        assert result == {"image_passiflora_leaves": ["image_passiflora_leaves.zip"]}

    def test_deduplication_across_versions(self):
        """Same filename in multiple versions should not duplicate."""
        registry = {
            "0.7.0": {"data.zip": SAMPLE_HASH_A},
            "0.8.0": {"data.zip": SAMPLE_HASH_B},
        }
        result = build_method_files_map(registry)
        assert result == {"data": ["data.zip"]}

    def test_multiple_datasets(self):
        """Multiple different datasets should yield multiple entries."""
        registry = {
            "0.7.0": {
                "image_passiflora_leaves.zip": SAMPLE_HASH_A,
                "outline_passiflora_leaves.zip": SAMPLE_HASH_B,
            },
        }
        result = build_method_files_map(registry)
        assert result == {
            "image_passiflora_leaves": ["image_passiflora_leaves.zip"],
            "outline_passiflora_leaves": ["outline_passiflora_leaves.zip"],
        }

    def test_sorted_by_method_name(self):
        """Output should be sorted by method name."""
        registry = {
            "0.7.0": {
                "z_data.zip": SAMPLE_HASH_A,
                "a_data.zip": SAMPLE_HASH_B,
            },
        }
        result = build_method_files_map(registry)
        assert list(result.keys()) == ["a_data", "z_data"]

    def test_empty_registry(self):
        """Empty registry should return empty map."""
        assert build_method_files_map({}) == {}

    def test_new_dataset_added_in_later_version(self):
        """A dataset appearing only in a later version should be included."""
        registry = {
            "0.7.0": {"data_a.zip": SAMPLE_HASH_A},
            "0.8.0": {
                "data_a.zip": SAMPLE_HASH_B,
                "data_b.zip": SAMPLE_HASH_C,
            },
        }
        result = build_method_files_map(registry)
        assert result == {
            "data_a": ["data_a.zip"],
            "data_b": ["data_b.zip"],
        }


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
        """Hash shorter than 64 chars should cause SystemExit."""
        manifest = {"data.zip": "abc123"}
        with pytest.raises(SystemExit):
            validate_manifest(manifest)

    def test_invalid_hash_uppercase(self):
        """Uppercase hex should be rejected (SHA256 hashes are lowercase)."""
        manifest = {"data.zip": "A" * 64}
        with pytest.raises(SystemExit):
            validate_manifest(manifest)

    def test_invalid_hash_non_hex(self):
        """Non-hex characters should be rejected."""
        manifest = {"data.zip": "g" * 64}
        with pytest.raises(SystemExit):
            validate_manifest(manifest)

    def test_empty_manifest(self):
        """Empty manifest should pass validation."""
        validate_manifest({})
