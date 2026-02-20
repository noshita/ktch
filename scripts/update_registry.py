#!/usr/bin/env python
"""Update ktch dataset registry from R2 manifest.json files.

Reads registry.toml for the list of datasets and versions, fetches the
manifest.json for each version from R2, and regenerates
ktch/datasets/_registry.py accordingly.

Usage:
    uv run python scripts/update_registry.py
    uv run python scripts/update_registry.py --dry-run

Examples:
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

_DATASET_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_VERSION_RE = re.compile(r"^(0|[1-9]\d*)$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RegistryError(Exception):
    """Raised when registry operations fail (invalid data, network errors, etc.)."""
###########################################################
# TOML reading
###########################################################


def read_registry_toml():
    """Read registry.toml and return (datasets, default_versions).

    Returns
    -------
    tuple of (dict, dict)
        datasets: {dataset_name: [version, ...]}
        default_versions: {dataset_name: version}
    """
    with open(REGISTRY_TOML_PATH, "rb") as f:
        config = tomllib.load(f)

    datasets = {}
    default_versions = {}

    for ds_name, ds_config in config.items():
        if not _DATASET_NAME_RE.match(ds_name):
            raise RegistryError(
                f"invalid dataset name {ds_name!r}: "
                "must match [a-z][a-z0-9_]*"
            )

        versions = ds_config.get("versions", [])
        default = ds_config.get("default")

        for v in versions:
            if not _VERSION_RE.match(v):
                raise RegistryError(
                    f"invalid version {v!r} for {ds_name}: "
                    "must be a non-negative integer string"
                )

        if not versions:
            print(
                f"Warning: {ds_name} has no versions listed, skipping.",
                file=sys.stderr,
            )
            continue

        datasets[ds_name] = versions

        if default is not None:
            if default not in versions:
                raise RegistryError(
                    f"default version '{default}' for {ds_name} "
                    f"is not in versions list {versions}"
                )
            default_versions[ds_name] = default

    return datasets, default_versions


###########################################################
# Manifest fetching & validation
###########################################################


def fetch_manifest(dataset_name, version):
    """Fetch manifest.json from R2 for the given dataset and version.

    Parameters
    ----------
    dataset_name : str
        Dataset name (e.g., "image_passiflora_leaves").
    version : str
        Version string (e.g., "2").

    Returns
    -------
    dict
        Mapping of filename to SHA256 hash.
    """
    url = f"{R2_BASE_URL}/datasets/{dataset_name}/v{version}/manifest.json"
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


def render_registry(dataset_registry, default_versions):
    """Render the full content of _registry.py (pure data, no functions).

    Parameters
    ----------
    dataset_registry : dict
        ``{dataset_name: {version: {filename: hash, ...}, ...}, ...}``
    default_versions : dict
        ``{dataset_name: version, ...}``

    Returns
    -------
    str
        The rendered file content.
    """
    lines = []
    lines.append('"""Dataset registry for ktch.datasets module."""')
    lines.append("")
    lines.append("# This file is auto-generated by:")
    lines.append("#   uv run python scripts/update_registry.py")
    lines.append("# Do not edit manually. Edit registry.toml instead.")
    lines.append("")
    lines.append("# Base URL for remote datasets")
    lines.append(f"BASE_URL = {json.dumps(R2_BASE_URL)}")
    lines.append("")
    lines.append(
        "# Per-dataset registry: {dataset_name: {version: {filename: sha256_hash}}}"
    )
    lines.append("dataset_registry = {")
    for ds_name in sorted(dataset_registry.keys()):
        versions = dataset_registry[ds_name]
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
    lines.append("}")
    lines.append("")
    lines.append("# Default dataset versions for the current ktch release.")
    lines.append(
        "# Updated at ktch release time to pin the recommended dataset version."
    )
    lines.append("default_versions = {")
    for ds_name in sorted(default_versions.keys()):
        version = default_versions[ds_name]
        lines.append(f"    {json.dumps(ds_name)}: {json.dumps(version)},")
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
        description="Update ktch dataset registry from R2 manifest.json files."
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
        datasets, default_versions = read_registry_toml()
        print(f"  Found {len(datasets)} dataset(s)")

        # Fetch and validate manifests for all datasets/versions
        dataset_registry = {}
        for ds_name in sorted(datasets.keys()):
            versions = datasets[ds_name]
            dataset_registry[ds_name] = {}
            for version in versions:
                print(f"Fetching manifest for {ds_name} v{version}...")
                manifest = fetch_manifest(ds_name, version)
                validate_manifest(manifest)
                dataset_registry[ds_name][version] = manifest
                print(
                    f"  Found {len(manifest)} file(s): "
                    f"{', '.join(sorted(manifest.keys()))}"
                )

        # Render new content
        new_content = render_registry(dataset_registry, default_versions)
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
