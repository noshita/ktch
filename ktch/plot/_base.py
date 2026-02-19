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
from types import ModuleType

_VALID_DEPS = frozenset({"matplotlib", "seaborn", "plotly"})

# Lazy import cache.
# Key absent: not yet attempted.
# Value is module object: import succeeded.
# Value is None: import failed.
_DEPS = {}

_IMPORT_TARGETS = {
    "matplotlib": "matplotlib.pyplot",
    "seaborn": "seaborn",
}

# Maps module-level attribute names to _DEPS keys for lazy access.
_ATTR_TO_DEP = {
    "plt": "matplotlib",
    "sns": "seaborn",
    "go": ("plotly", "plotly.graph_objects"),
    "px": ("plotly", "plotly.express"),
}


def _try_import(dep_name: str) -> ModuleType | None:
    """Try to import a dependency, caching the result.

    Returns the imported module, or None if unavailable.
    """
    if dep_name in _DEPS:
        return _DEPS[dep_name]

    if dep_name == "plotly":
        # Import both subpackages together; cache each separately.
        try:
            go = importlib.import_module("plotly.graph_objects")
            px = importlib.import_module("plotly.express")
        except ImportError:
            go = None
            px = None
        _DEPS["plotly"] = go
        _DEPS["plotly.graph_objects"] = go
        _DEPS["plotly.express"] = px
        return go

    target = _IMPORT_TARGETS.get(dep_name)
    if target is None:
        return None
    try:
        mod = importlib.import_module(target)
    except ImportError:
        mod = None
    _DEPS[dep_name] = mod
    return mod


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
            f"Unknown dependency name(s): {unknown}. "
            f"Valid names: {sorted(_VALID_DEPS)}"
        )

    missing = [name for name in dep_names if _try_import(name) is None]

    if missing:
        deps_str = ", ".join(missing)
        raise ImportError(
            f"The following dependencies are required: {deps_str}\n"
            f"Install them with:\n"
            f"  pip: pip install ktch[plot]\n"
            f"  conda: conda install ktch-plot"
        )


def __getattr__(name):
    """Lazy attribute access for plt, sns, go, px."""
    if name not in _ATTR_TO_DEP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    mapping = _ATTR_TO_DEP[name]
    if isinstance(mapping, tuple):
        # (dep_name, subpackage) — e.g. ("plotly", "plotly.graph_objects")
        dep_name, subpackage = mapping
        _try_import(dep_name)  # ensures all plotly subpackages are cached
        return _DEPS.get(subpackage)
    else:
        return _try_import(mapping)


__all__ = [
    "require_dependencies",
]
