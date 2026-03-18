"""Protocol and mixin for morphometric data containers."""

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

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class MorphoData(Protocol):
    """Interface for morphometric data containers.

    Any class that has ``specimen_name``, ``to_numpy()``, and
    ``to_dataframe()`` structurally satisfies this protocol without
    explicit inheritance.

    Examples
    --------
    >>> isinstance(NefData(...), MorphoData)
    True
    """

    @property
    def specimen_name(self) -> str: ...

    def to_numpy(self) -> np.ndarray: ...

    def to_dataframe(self) -> pd.DataFrame: ...


class MorphoDataMixin:
    """Shared implementation for morphometric data containers.

    Provides ``__array__`` (NumPy protocol) and ``__repr__`` based on
    the ``to_numpy()`` and ``specimen_name`` that concrete classes
    must supply.

    Subclasses can define ``_repr_detail()`` returning a string of
    extra fields to include in the repr (e.g. ``"n_harmonics=20"``).
    """

    def __array__(self, dtype=None, copy=None):
        """Support ``np.asarray(container)``.

        Raises
        ------
        TypeError
            If ``to_numpy()`` does not return an ndarray (e.g. when
            ``TPSData`` contains curves and returns a tuple).
        """
        result = self.to_numpy()
        if not isinstance(result, np.ndarray):
            raise TypeError(
                f"{type(self).__name__}.to_numpy() returned "
                f"{type(result).__name__}, not ndarray. "
                f"Call to_numpy() directly to handle the result."
            )
        return np.array(result, dtype=dtype, copy=copy)

    def __repr__(self):
        detail = self._repr_detail() if hasattr(self, "_repr_detail") else ""
        if detail:
            return (
                f"{type(self).__name__}(specimen_name={self.specimen_name!r}, {detail})"
            )
        return f"{type(self).__name__}(specimen_name={self.specimen_name!r})"
