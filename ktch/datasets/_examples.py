"""Example data fetching for tutorials and demonstrations."""

# Copyright 2026 Koji Noshita
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from importlib import resources
from pathlib import Path

from ._registry import (
    BASE_URL,
    bundled_examples,
    example_default_versions,
    example_registry,
)

try:
    import pooch
except ImportError:
    pooch = None

DATA_MODULE = "ktch.datasets.data"


def fetch(
    name: str, *, data_home: str | Path | None = None, version: str | None = None
) -> str:
    """Fetch an example data file and return its local path.

    Example data are individual sample files used in tutorials and
    demonstrations.  Unlike :func:`load_landmark_mosquito_wings` and other
    ``load_*`` functions that return loaded data, this function returns a
    file path so that tutorials can demonstrate I/O operations.

    Available example files:

    ==========================================  ========  ============
    Name                                        Type      Size
    ==========================================  ========  ============
    ``landmarks_triangle.tps``                  bundled   6.5 KB
    ``danshaku_08_allSegments_SPHARM.coef``     bundled   22 KB
    ``danshaku_08_allSegments_para.vtp``        remote    --
    ``danshaku_08_allSegments_surf.vtp``        remote    --
    ==========================================  ========  ============

    Parameters
    ----------
    name : str
        Filename of the example data (e.g., ``"landmarks_triangle.tps"``).
    data_home : str, Path, or None
        Directory to store the file.  For remote files, this is the
        download cache directory; defaults to
        ``pooch.os_cache("ktch-examples")``.  For bundled files, if
        specified the file is copied to this directory; if None the
        in-package path is returned directly.
    version : str or None
        Version string for remote files (e.g., ``"1"``).  Ignored for
        bundled files.  If None, uses the default version for the current
        ktch release.

    Returns
    -------
    str
        Absolute path to the local file.

    Raises
    ------
    ValueError
        If *name* is not a known example file.
    ImportError
        If the file is remote and pooch is not installed.

    Examples
    --------
    >>> from ktch.datasets import fetch
    >>> path = fetch("landmarks_triangle.tps")
    >>> path.endswith("landmarks_triangle.tps")
    True
    """
    if name in bundled_examples:
        return _fetch_bundled_example(name, data_home=data_home)

    if name in example_registry:
        return _fetch_remote_example(name, version=version, data_home=data_home)

    all_names = sorted(bundled_examples | set(example_registry))
    raise ValueError(f"Unknown example file {name!r}. Available: {all_names}")


def _fetch_bundled_example(name, *, data_home=None):
    """Return path to a bundled example, optionally copying to *data_home*."""
    ref = resources.files(DATA_MODULE) / name
    src_path = str(ref)

    if data_home is None:
        return src_path

    dest_dir = Path(data_home)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / name

    if not dest_path.exists():
        import shutil

        shutil.copy2(src_path, dest_path)

    return str(dest_path)


def _fetch_remote_example(name, *, version=None, data_home=None):
    """Download a remote example file and return its local path."""
    if pooch is None:
        raise ImportError(
            "Missing optional dependency 'pooch' for downloading remote example "
            "files. Please install it using: pip install ktch[data]"
        )

    if version is None:
        try:
            version = example_default_versions[name]
        except KeyError:
            raise ValueError(
                f"Unknown example file {name!r} in default versions"
            ) from None

    versions = example_registry[name]
    if version not in versions:
        raise ValueError(
            f"Unknown version {version!r} for example {name!r}. "
            f"Available: {sorted(versions)}"
        )
    known_hash = versions[version]

    stem = Path(name).stem
    url = f"{BASE_URL}/examples/{stem}/v{version}/{name}"

    if data_home is None:
        cache_path = pooch.os_cache("ktch-examples") / stem / f"v{version}"
    else:
        cache_path = Path(data_home) / stem / f"v{version}"

    return pooch.retrieve(
        url=url,
        known_hash=f"sha256:{known_hash}",
        path=cache_path,
        fname=name,
    )
