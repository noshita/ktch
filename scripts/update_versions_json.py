#!/usr/bin/env python
"""Update doc/_static/versions.json for a new release.

Sets the specified version as the stable version in the version switcher
configuration used by pydata-sphinx-theme.

Usage:
    uv run python scripts/update_versions_json.py <version>
    uv run python scripts/update_versions_json.py --dry-run <version>

Examples:
    uv run python scripts/update_versions_json.py 0.8.0
    uv run python scripts/update_versions_json.py --dry-run 0.8.0
"""

import argparse
import json
import sys
from pathlib import Path

VERSIONS_PATH = (
    Path(__file__).resolve().parent.parent / "doc" / "_static" / "versions.json"
)


def build_versions(version):
    """Build versions.json content for the given release version.

    Parameters
    ----------
    version : str
        Version string (e.g., "0.8.0").

    Returns
    -------
    list[dict]
        Version switcher entries.
    """
    return [
        {
            "name": f"{version} (stable)",
            "version": version,
            "url": "https://doc.ktch.dev/stable/",
            "preferred": True,
        },
        {
            "name": "dev",
            "version": "dev",
            "url": "https://doc.ktch.dev/dev/",
        },
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Update doc/_static/versions.json for a new release."
    )
    parser.add_argument("version", help="Version to set as stable (e.g., 0.8.0)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without writing to file",
    )
    args = parser.parse_args()

    version = args.version

    # Validate version format
    parts = version.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        print(f"Error: invalid version format '{version}' (expected X.Y.Z)", file=sys.stderr)
        sys.exit(1)

    # Build new content
    new_data = build_versions(version)
    new_content = json.dumps(new_data, indent=2, ensure_ascii=False) + "\n"

    # Read current content
    old_content = VERSIONS_PATH.read_text() if VERSIONS_PATH.exists() else ""

    if old_content.rstrip("\n") == new_content.rstrip("\n"):
        print(f"versions.json already up to date for {version}.")
        return

    # Show diff
    _show_diff(old_content, new_content)

    if args.dry_run:
        print("\n--dry-run: no files modified.")
        return

    # Write
    VERSIONS_PATH.write_text(new_content)
    print(f"\nUpdated {VERSIONS_PATH}")


def _show_diff(old_content, new_content):
    """Show a unified diff between old and new content."""
    import difflib

    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile="versions.json (old)",
        tofile="versions.json (new)",
    )
    diff_str = "".join(diff)
    if diff_str:
        print("Diff:")
        print(diff_str)


if __name__ == "__main__":
    main()
