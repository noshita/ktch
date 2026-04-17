#!/usr/bin/env python
"""Update ktch dataset and example data registry from R2 manifest.json files.

Reads registry.toml for the list of datasets/examples and versions, fetches
the manifest.json for each version from R2, and regenerates
ktch/datasets/_registry.py accordingly.

Usage:
    uv run python scripts/update_registry.py
    uv run python scripts/update_registry.py --dry-run

Examples
--------
    uv run python scripts/update_registry.py
    uv run python scripts/update_registry.py --dry-run
"""

import argparse
import difflib
import json
import os
import re
import sys
import tomllib
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

R2_BASE_URL = "https://pub-c1d6dba6c94843f88f0fd096d19c0831.r2.dev"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_TOML_PATH = PROJECT_ROOT / "ktch" / "datasets" / "registry.toml"
REGISTRY_PY_PATH = PROJECT_ROOT / "ktch" / "datasets" / "_registry.py"

_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")
_VERSION_RE = re.compile(r"^(0|[1-9]\d*)$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_TOP_LEVEL_KEYS = {"bundled_examples", "datasets", "examples"}


class RegistryError(Exception):
    """Raised when registry operations fail (invalid data, network errors, etc.)."""


###########################################################
# TOML reading
###########################################################


def _validate_name(name, kind="dataset"):
    """Validate a dataset or example name."""
    if not _NAME_RE.match(name):
        raise RegistryError(
            f"invalid {kind} name {name!r}: "
            "must match [a-zA-Z][a-zA-Z0-9_]*"
        )


def _parse_versioned_entry(name, entry, kind="dataset"):
    """Parse a versioned entry (dataset or example) from TOML config.

    Returns (versions_list, default_version) or raises RegistryError.
    """
    _validate_name(name, kind)

    versions = entry.get("versions", [])
    default = entry.get("default")

    for v in versions:
        if not _VERSION_RE.match(v):
            raise RegistryError(
                f"invalid version {v!r} for {kind} {name}: "
                "must be a non-negative integer string"
            )

    if not versions:
        print(
            f"Warning: {kind} {name} has no versions listed, skipping.",
            file=sys.stderr,
        )
        return None, None

    if default is not None:
        if default not in versions:
            raise RegistryError(
                f"default version '{default}' for {kind} {name} "
                f"is not in versions list {versions}"
            )

    return versions, default


def _validate_top_level_keys(config):
    """Validate that registry.toml uses the expected top-level schema."""
    unknown_keys = sorted(set(config) - _ALLOWED_TOP_LEVEL_KEYS)
    if unknown_keys:
        unknown = ", ".join(repr(key) for key in unknown_keys)
        allowed = ", ".join(sorted(_ALLOWED_TOP_LEVEL_KEYS))
        raise RegistryError(
            f"unknown top-level key(s) in registry.toml: {unknown}. "
            f"Allowed keys are: {allowed}. "
            "Use [datasets.<name>] for datasets and [examples.<name>] "
            "for remote examples."
        )


def read_registry_toml():
    """Read registry.toml and return parsed configuration.

    Returns
    -------
    dict with keys:
        datasets: {name: [version, ...]}
        dataset_defaults: {name: version}
        bundled_examples: list of str
        examples: {name: {"filename": str, "versions": [str, ...]}}
        example_defaults: {filename: version}
    """
    with open(REGISTRY_TOML_PATH, "rb") as f:
        config = tomllib.load(f)

    _validate_top_level_keys(config)

    # --- Datasets ---
    datasets = {}
    dataset_defaults = {}
    ds_section = config.get("datasets", {})
    for ds_name, ds_config in ds_section.items():
        versions, default = _parse_versioned_entry(ds_name, ds_config, "dataset")
        if versions is None:
            continue
        datasets[ds_name] = versions
        if default is not None:
            dataset_defaults[ds_name] = default

    # --- Bundled examples ---
    bundled = config.get("bundled_examples", [])
    bundled_examples = sorted(bundled)

    # --- Remote examples ---
    examples = {}
    example_defaults = {}
    ex_section = config.get("examples", {})
    for ex_name, ex_config in ex_section.items():
        filename = ex_config.get("filename")
        if not filename:
            raise RegistryError(
                f"example {ex_name!r} is missing 'filename' field"
            )
        versions, default = _parse_versioned_entry(ex_name, ex_config, "example")
        if versions is None:
            continue
        examples[ex_name] = {"filename": filename, "versions": versions}
        if default is not None:
            example_defaults[filename] = default

    return {
        "datasets": datasets,
        "dataset_defaults": dataset_defaults,
        "bundled_examples": bundled_examples,
        "examples": examples,
        "example_defaults": example_defaults,
    }


###########################################################
# Manifest fetching & validation
###########################################################


def fetch_manifest(prefix, name, version):
    """Fetch manifest.json from R2 for the given entry.

    Parameters
    ----------
    prefix : str
        R2 prefix ("datasets" or "examples").
    name : str
        Entry name (e.g., "image_passiflora_leaves" or
        "danshaku_08_allSegments_para").
    version : str
        Version string (e.g., "2").

    Returns
    -------
    dict
        Mapping of filename to SHA256 hash.
    """
    url = f"{R2_BASE_URL}/{prefix}/{name}/v{version}/manifest.json"
    req = Request(url, headers={"User-Agent": "ktch-registry-updater/1.0"})
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except HTTPError as exc:
        if exc.code == 404:
            raise RegistryError(
                f"manifest.json not found at {url}"
            ) from exc
        else:
            raise RegistryError(
                f"HTTP {exc.code} fetching {url}"
            ) from exc
    except URLError as exc:
        raise RegistryError(
            f"network error fetching {url}: {exc.reason}"
        ) from exc


def validate_manifest(manifest):
    """Validate that all manifest entries have valid SHA256 hashes.

    Parameters
    ----------
    manifest : dict
        Mapping of filename to hash string.

    Raises
    ------
    RegistryError
        If any entry is invalid.
    """
    for filename, hash_value in manifest.items():
        if not isinstance(hash_value, str):
            raise RegistryError(
                f"invalid manifest entry for '{filename}': "
                f"expected string hash, got {type(hash_value).__name__}"
            )
        if not _SHA256_RE.match(hash_value):
            raise RegistryError(
                f"invalid SHA256 hash for '{filename}': '{hash_value}'"
            )


###########################################################
# Rendering _registry.py
###########################################################


def _render_dict(lines, data, indent=4, sort_key=None):
    """Render a nested dict as Python source lines."""
    prefix = " " * indent
    keys = sorted(data.keys(), key=sort_key)
    for key in keys:
        value = data[key]
        if isinstance(value, dict):
            lines.append(f"{prefix}{json.dumps(key)}: {{")
            _render_dict(lines, value, indent + 4)
            lines.append(f"{prefix}}},")
        else:
            lines.append(f"{prefix}{json.dumps(key)}: {json.dumps(value)},")


def _render_dataset_registry(lines, registry):
    """Render dataset_registry with numeric version sorting."""
    for ds_name in sorted(registry.keys()):
        versions = registry[ds_name]
        lines.append(f"    {json.dumps(ds_name)}: {{")
        for version in sorted(versions.keys(), key=int):
            entries = versions[version]
            lines.append(f"        {json.dumps(version)}: {{")
            for filename in sorted(entries.keys()):
                hash_value = entries[filename]
                lines.append(
                    f"            {json.dumps(filename)}: {json.dumps(hash_value)},"
                )
            lines.append("        },")
        lines.append("    },")


def _render_example_registry(lines, registry):
    """Render example_registry with numeric version sorting."""
    for filename in sorted(registry.keys()):
        versions = registry[filename]
        lines.append(f"    {json.dumps(filename)}: {{")
        for version in sorted(versions.keys(), key=int):
            hash_value = versions[version]
            lines.append(
                f"        {json.dumps(version)}: {json.dumps(hash_value)},"
            )
        lines.append("    },")


def render_registry(
    dataset_registry, dataset_defaults,
    bundled_examples, example_registry, example_defaults,
):
    """Render the full content of _registry.py.

    Parameters
    ----------
    dataset_registry : dict
        ``{dataset_name: {version: {filename: hash}}}``
    dataset_defaults : dict
        ``{dataset_name: version}``
    bundled_examples : list of str
        Filenames of bundled example files.
    example_registry : dict
        ``{filename: {version: hash}}``
    example_defaults : dict
        ``{filename: version}``

    Returns
    -------
    str
        The rendered file content.
    """
    lines = []
    lines.append('"""Dataset and example data registry for ktch.datasets module."""')
    lines.append("")
    lines.append("# This file is auto-generated by:")
    lines.append("#   uv run python scripts/update_registry.py")
    lines.append("# Do not edit manually. Edit registry.toml instead.")
    lines.append("")
    lines.append("# Base URL for remote data")
    lines.append(f"BASE_URL = {json.dumps(R2_BASE_URL)}")
    lines.append("")

    # --- Datasets ---
    lines.append("# " + "-" * 75)
    lines.append("# Datasets")
    lines.append("# " + "-" * 75)
    lines.append("")
    lines.append(
        "# Per-dataset registry: {dataset_name: {version: {filename: sha256_hash}}}"
    )
    lines.append("dataset_registry = {")
    _render_dataset_registry(lines, dataset_registry)
    lines.append("}")
    lines.append("")
    lines.append("# Default dataset versions for the current ktch release.")
    lines.append(
        "# Updated at ktch release time to pin the recommended dataset version."
    )
    lines.append("default_versions = {")
    _render_dict(lines, dataset_defaults)
    lines.append("}")
    lines.append("")

    # --- Example data ---
    lines.append("# " + "-" * 75)
    lines.append("# Example data (sample files for tutorials)")
    lines.append("# " + "-" * 75)
    lines.append("")
    lines.append("# Bundled examples shipped with the package (< 100 KB).")
    lines.append("# Set of filenames within ktch/datasets/data/.")
    lines.append("bundled_examples = {")
    for name in sorted(bundled_examples):
        lines.append(f"    {json.dumps(name)},")
    lines.append("}")
    lines.append("")
    lines.append("# Remote examples: {filename: {version: sha256_hash}}")
    lines.append(
        "# URL pattern: {BASE_URL}/examples/{stem}/v{version}/{filename}"
    )
    lines.append("example_registry = {")
    _render_example_registry(lines, example_registry)
    lines.append("}")
    lines.append("")
    lines.append("# Default example versions for the current ktch release.")
    lines.append("example_default_versions = {")
    _render_dict(lines, example_defaults)
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


###########################################################
# Diff display
###########################################################


def _show_diff(old_content, new_content):
    """Show a unified diff between old and new content."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines, fromfile="_registry.py (old)", tofile="_registry.py (new)"
    )
    diff_str = "".join(diff)
    if diff_str:
        print("\nDiff:")
        print(diff_str)


###########################################################
# Main
###########################################################


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Update ktch dataset and example data registry from R2 "
            "manifest.json files."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without writing to file",
    )
    args = parser.parse_args()

    try:
        # Read TOML config
        print(f"Reading {REGISTRY_TOML_PATH}...")
        config = read_registry_toml()
        datasets = config["datasets"]
        dataset_defaults = config["dataset_defaults"]
        bundled_examples = config["bundled_examples"]
        examples = config["examples"]
        example_defaults = config["example_defaults"]
        print(f"  Found {len(datasets)} dataset(s), {len(examples)} remote example(s), "
              f"{len(bundled_examples)} bundled example(s)")

        # Fetch and validate manifests for datasets
        dataset_registry = {}
        for ds_name in sorted(datasets.keys()):
            versions = datasets[ds_name]
            dataset_registry[ds_name] = {}
            for version in versions:
                print(f"Fetching manifest for dataset {ds_name} v{version}...")
                manifest = fetch_manifest("datasets", ds_name, version)
                validate_manifest(manifest)
                dataset_registry[ds_name][version] = manifest
                print(
                    f"  Found {len(manifest)} file(s): "
                    f"{', '.join(sorted(manifest.keys()))}"
                )

        # Fetch and validate manifests for remote examples
        example_registry = {}
        for ex_name in sorted(examples.keys()):
            ex_info = examples[ex_name]
            filename = ex_info["filename"]
            versions = ex_info["versions"]
            example_registry[filename] = {}
            for version in versions:
                print(f"Fetching manifest for example {ex_name} v{version}...")
                manifest = fetch_manifest("examples", ex_name, version)
                validate_manifest(manifest)
                if filename not in manifest:
                    raise RegistryError(
                        f"expected file {filename!r} not found in manifest "
                        f"for example {ex_name} v{version}. "
                        f"Found: {sorted(manifest.keys())}"
                    )
                example_registry[filename][version] = manifest[filename]
                print(f"  Hash: {manifest[filename][:16]}...")

        # Render new content
        new_content = render_registry(
            dataset_registry, dataset_defaults,
            bundled_examples, example_registry, example_defaults,
        )
    except RegistryError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Show diff
    if REGISTRY_PY_PATH.exists():
        old_content = REGISTRY_PY_PATH.read_text(encoding="utf-8")
    else:
        old_content = ""

    if old_content == new_content:
        print("No changes needed.")
        return

    _show_diff(old_content, new_content)

    if args.dry_run:
        print("\n--dry-run: no files modified.")
        return

    # Write atomically: write to temp file, then rename
    tmp_path = REGISTRY_PY_PATH.with_suffix(".py.tmp")
    try:
        tmp_path.write_text(new_content, encoding="utf-8")
        os.replace(tmp_path, REGISTRY_PY_PATH)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    print(f"\nUpdated {REGISTRY_PY_PATH}")


if __name__ == "__main__":
    main()
