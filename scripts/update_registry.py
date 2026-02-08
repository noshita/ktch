#!/usr/bin/env python
"""Update ktch dataset registry from R2 manifest.json.

Fetches the manifest.json for a given version from R2 and updates
ktch/datasets/_registry.py accordingly.

Usage:
    uv run python scripts/update_registry.py <version>
    uv run python scripts/update_registry.py --dry-run <version>

Examples:
    uv run python scripts/update_registry.py 0.8.0
    uv run python scripts/update_registry.py --dry-run 0.8.0
"""

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

R2_BASE_URL = "https://pub-c1d6dba6c94843f88f0fd096d19c0831.r2.dev"
REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent / "ktch" / "datasets" / "_registry.py"
)


###########################################################
# Manifest fetching & validation
###########################################################


def fetch_manifest(version):
    """Fetch manifest.json from R2 for the given version.

    Parameters
    ----------
    version : str
        Version string (e.g., "0.8.0").

    Returns
    -------
    dict
        Mapping of filename to SHA256 hash.
    """
    url = f"{R2_BASE_URL}/releases/v{version}/manifest.json"
    req = Request(url, headers={"User-Agent": "ktch-registry-updater/1.0"})
    try:
        with urlopen(req) as resp:
            return json.loads(resp.read())
    except HTTPError as exc:
        if exc.code == 404:
            print(f"Error: manifest.json not found at {url}", file=sys.stderr)
        else:
            print(f"Error: HTTP {exc.code} fetching {url}", file=sys.stderr)
        sys.exit(1)
    except URLError as exc:
        print(f"Error: network error fetching {url}: {exc.reason}", file=sys.stderr)
        sys.exit(1)


def validate_manifest(manifest):
    """Validate that all manifest entries have valid SHA256 hashes.

    Parameters
    ----------
    manifest : dict
        Mapping of filename to hash string.

    Raises
    ------
    SystemExit
        If any entry is invalid.
    """
    hex_pattern = re.compile(r"^[0-9a-f]{64}$")
    for filename, hash_value in manifest.items():
        if not hex_pattern.match(hash_value):
            print(
                f"Error: invalid SHA256 hash for '{filename}': '{hash_value}'",
                file=sys.stderr,
            )
            sys.exit(1)


###########################################################
# Registry reading & updating
###########################################################


def read_current_registry():
    """Read the current _registry.py and extract versioned_registry.

    Returns
    -------
    dict
        The current versioned_registry dict.
    """
    namespace = {}
    exec(REGISTRY_PATH.read_text(), namespace)
    return dict(namespace["versioned_registry"])


def build_method_files_map(versioned_registry):
    """Build method_files_map from all filenames across all versions.

    Derives method names by stripping the file extension from each filename.
    For example, ``"image_passiflora_leaves.zip"`` yields method name
    ``"image_passiflora_leaves"`` with file list ``["image_passiflora_leaves.zip"]``.

    Parameters
    ----------
    versioned_registry : dict
        ``{version: {filename: hash, ...}, ...}``

    Returns
    -------
    dict
        ``{method_name: [filename, ...], ...}`` sorted by method name.
    """
    method_files = {}
    for version_entries in versioned_registry.values():
        for filename in version_entries:
            method_name = Path(filename).stem
            if method_name not in method_files:
                method_files[method_name] = set()
            method_files[method_name].add(filename)
    return {k: sorted(v) for k, v in sorted(method_files.items())}


###########################################################
# Rendering _registry.py
###########################################################


def render_registry(versioned_registry, method_files_map):
    """Render the full content of _registry.py.

    Parameters
    ----------
    versioned_registry : dict
        ``{version: {filename: hash, ...}, ...}``
    method_files_map : dict
        ``{method_name: [filename, ...], ...}``

    Returns
    -------
    str
        The rendered file content.
    """
    lines = []
    lines.append('"""Dataset registry for ktch.datasets module."""')
    lines.append("")
    lines.append("# Registry is auto-updated via:")
    lines.append("#   uv run python scripts/update_registry.py <version>")
    lines.append("# See ref/ktch-registry-update-via-manifest.md for details.")
    lines.append("")
    lines.append("# Base URL for remote datasets")
    lines.append('BASE_URL = "https://pub-c1d6dba6c94843f88f0fd096d19c0831.r2.dev"')
    lines.append("")
    lines.append("# Version-specific registry: {version: {filename: sha256_hash}}")
    lines.append("versioned_registry = {")
    for version in sorted(versioned_registry.keys()):
        entries = versioned_registry[version]
        lines.append(f'    "{version}": {{')
        for filename in sorted(entries.keys()):
            hash_value = entries[filename]
            lines.append(f'        "{filename}": "{hash_value}",')
        lines.append("    },")
    lines.append("}")
    lines.append("")
    lines.append("# dataset method mapping with their associated filenames")
    lines.append('# <method_name> : ["filename1", "filename2", ...]')
    lines.append("method_files_map = {")
    for method_name, filenames in sorted(method_files_map.items()):
        files_str = ", ".join(f'"{f}"' for f in filenames)
        lines.append(f'    "{method_name}": [{files_str}],')
    lines.append("}")
    lines.append("")
    lines.append("")
    lines.append(_FUNCTIONS_SOURCE)

    return "\n".join(lines)


_FUNCTIONS_SOURCE = '''\
def get_registry(version):
    """Get the registry for a specific version.

    Parameters
    ----------
    version : str
        The version string (e.g., "0.7.0").

    Returns
    -------
    dict
        Registry mapping filename to SHA256 hash.

    Raises
    ------
    ValueError
        If the version is not found in the registry.
    """
    if version not in versioned_registry:
        available = ", ".join(sorted(versioned_registry.keys()))
        raise ValueError(
            f"Dataset version \'{version}\' not found. Available versions: {available}"
        )
    return versioned_registry[version]


def get_url(filename, version):
    """Get the download URL for a specific file and version.

    Parameters
    ----------
    filename : str
        The filename to download.
    version : str
        The version string (e.g., "0.7.0").

    Returns
    -------
    str
        The full download URL.
    """
    return f"{BASE_URL}/releases/v{version}/{filename}"
'''


###########################################################
# Main
###########################################################


def main():
    parser = argparse.ArgumentParser(
        description="Update ktch dataset registry from R2 manifest.json."
    )
    parser.add_argument("version", help="Version to update (e.g., 0.8.0)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without writing to file",
    )
    args = parser.parse_args()

    version = args.version

    # Fetch and validate manifest
    print(f"Fetching manifest.json for v{version}...")
    manifest = fetch_manifest(version)
    validate_manifest(manifest)
    print(f"  Found {len(manifest)} dataset(s): {', '.join(sorted(manifest.keys()))}")

    # Read current registry and update
    current_registry = read_current_registry()

    if version in current_registry:
        print(f"  Warning: version '{version}' already exists, will be overwritten.")

    current_registry[version] = manifest

    # Build method_files_map from all versions
    method_files_map = build_method_files_map(current_registry)

    # Render new content
    new_content = render_registry(current_registry, method_files_map)

    # Show diff
    old_content = REGISTRY_PATH.read_text()
    if old_content == new_content:
        print("No changes needed.")
        return

    _show_diff(old_content, new_content)

    if args.dry_run:
        print("\n--dry-run: no files modified.")
        return

    # Write
    REGISTRY_PATH.write_text(new_content)
    print(f"\nUpdated {REGISTRY_PATH}")


def _show_diff(old_content, new_content):
    """Show a unified diff between old and new content."""
    import difflib

    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines, fromfile="_registry.py (old)", tofile="_registry.py (new)"
    )
    diff_str = "".join(diff)
    if diff_str:
        print("\nDiff:")
        print(diff_str)


if __name__ == "__main__":
    main()
