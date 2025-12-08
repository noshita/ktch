"""
The :mod:`ktch.plot` module implements plotting functions for morphometrics.
"""
# Copyright 2025 Koji Noshita
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

from ._kriging import tps_grid_2d_plot
from ._pca import explained_variance_ratio_plot

_RENAMED_FUNCTIONS = {
    "plot_explained_variance_ratio": "explained_variance_ratio_plot",
}

__all__ = [
    "explained_variance_ratio_plot",
    "tps_grid_2d_plot",
]


def __getattr__(name: str):
    if name in _RENAMED_FUNCTIONS:
        new_name = _RENAMED_FUNCTIONS[name]
        warnings.warn(
            f"'{name}' has been renamed to '{new_name}'. "
            f"The {name} is deprecated and will be removed in version v0.9.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new_name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Include old function names in dir() and autocompletion."""
    return list(__all__) + list(_RENAMED_FUNCTIONS.keys())
