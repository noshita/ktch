"""Base utilities for plot module."""

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

import importlib

_VALID_DEPS = frozenset({"matplotlib", "seaborn", "plotly"})


def _check_import(name: str) -> bool:
    """Return True if *name* is importable, False otherwise."""
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


def require_dependencies(*dep_names: str) -> None:
    """Check if required plotting dependencies are available.

    Parameters
    ----------
    *dep_names : str
        Names of required dependencies. Valid values are:
        'matplotlib', 'seaborn', 'plotly'

    Raises
    ------
    ValueError
        If any dependency name is not recognized.
    ImportError
        If any required dependency is not installed.

    Examples
    --------
    >>> from ktch.plot._base import require_dependencies
    >>> require_dependencies('matplotlib')
    >>> require_dependencies('matplotlib', 'seaborn')
    """
    unknown = [n for n in dep_names if n not in _VALID_DEPS]
    if unknown:
        raise ValueError(
            f"Unknown dependency name(s): {unknown}. Valid names: {sorted(_VALID_DEPS)}"
        )

    missing = [name for name in dep_names if not _check_import(name)]

    if missing:
        deps_str = ", ".join(missing)
        raise ImportError(
            f"The following dependencies are required: {deps_str}\n"
            f"Install them with:\n"
            f"  pip: pip install ktch[plot]\n"
            f"  conda: conda install ktch-plot"
        )


__all__ = [
    "require_dependencies",
]
