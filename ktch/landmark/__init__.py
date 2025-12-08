"""
The :mod:`ktch.landmark` module implements landmark-based morphometrics.
"""

# Copyright 2020 Koji Noshita
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

import warnings

from ._Procrustes_analysis import GeneralizedProcrustesAnalysis, centroid_size

_MOVED_FUNCTIONS = {
    "tps_grid_2d_plot": "ktch.plot",
}

__all__ = ["GeneralizedProcrustesAnalysis", "centroid_size", "tps_grid_2d_plot"]


def __getattr__(name: str):
    if name in _MOVED_FUNCTIONS:
        new_module = _MOVED_FUNCTIONS[name]
        warnings.warn(
            f"'{name}' has been moved to '{new_module}'. "
            f"Importing from 'ktch.landmark' is deprecated "
            f"and will be removed in v0.9.",
            DeprecationWarning,
            stacklevel=2,
        )
        import importlib

        module = importlib.import_module(new_module)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Include moved functions in dir() and autocompletion."""
    return list(__all__) + list(_MOVED_FUNCTIONS.keys())
