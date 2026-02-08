"""Dataset registry for ktch.datasets module."""

# Registry is auto-updated via:
#   uv run python scripts/update_registry.py <version>
# See ref/ktch-registry-update-via-manifest.md for details.

# Base URL for remote datasets
BASE_URL = "https://pub-c1d6dba6c94843f88f0fd096d19c0831.r2.dev"

# Version-specific registry: {version: {filename: sha256_hash}}
versioned_registry = {
    "0.7.0": {
        "image_passiflora_leaves.zip": "d23b22252d0baf3b60ef1986e6518f0550e2e1aa81f677b5991938ab65e2e45b",
    },
}

# dataset method mapping with their associated filenames
# <method_name> : ["filename1", "filename2", ...]
method_files_map = {
    "image_passiflora_leaves": ["image_passiflora_leaves.zip"],
}


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
            f"Dataset version '{version}' not found. Available versions: {available}"
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
