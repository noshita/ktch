"""SPHARM-PDM file (_para.vtk, _surf.vtk, and .coef) I/O functions"""

# Copyright 2023 Koji Noshita
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

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from ._converters import _cvt_spharm_coef_spharmpdm_to_list
from ._protocols import MorphoDataMixin


@dataclass(repr=False)
class SpharmPdmData(MorphoDataMixin):
    """SPHARM-PDM coefficient data for a single specimen.

    Stores spherical harmonic coefficients in the nested list format
    where ``coeffs[l]`` has shape ``(2*l+1, 3)`` for degree ``l``.

    Parameters
    ----------
    specimen_name : str
        Specimen name.
    coeffs : list of np.ndarray
        Nested list of complex128 coefficient arrays.
        ``coeffs[l]`` has shape ``(2*l+1, 3)`` for degree ``l``.
    """

    specimen_name: str
    coeffs: list[np.ndarray]

    @property
    def l_max(self) -> int:
        """Maximum spherical harmonic degree."""
        return len(self.coeffs) - 1

    @property
    def n_degrees(self) -> int:
        """Number of spherical harmonic degrees (l_max + 1)."""
        return len(self.coeffs)

    def _repr_detail(self):
        return f"l_max={self.l_max}"

    def to_numpy(self) -> np.ndarray:
        """Return flat coefficient array.

        Returns
        -------
        coeffs : np.ndarray of shape ((l_max+1)^2, 3), complex128
            Concatenated coefficient matrix.
        """
        return np.vstack(self.coeffs)

    def to_dataframe(self) -> pd.DataFrame:
        """Return coefficients as a DataFrame.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with columns ``x``, ``y``, ``z``.
            Index is a MultiIndex ``(specimen_id, (l, m))``.
        """
        rows = []
        index_tuples = []
        for l_val, coef_l in enumerate(self.coeffs):
            for idx, m_val in enumerate(range(-l_val, l_val + 1)):
                rows.append(coef_l[idx])
                index_tuples.append((self.specimen_name, (l_val, m_val)))

        return pd.DataFrame(
            rows,
            columns=["x", "y", "z"],
            index=pd.MultiIndex.from_tuples(
                index_tuples,
                names=["specimen_id", "l_m"],
            ),
        )


def read_spharmpdm_coef(path: str | Path) -> SpharmPdmData:
    """Read .coef file of SPHARM-PDM.

    The .coef file is an output of the `ParaToSPHARMMesh` step
    of `SPHARM-PDM <https://www.nitrc.org/projects/spharm-pdm>`_,
    and contains SPHARM coefficients.

    The file format consists of:
    - First value: total number of coefficients (3*(lmax+1)^2)
    - Remaining values: real and imaginary parts of coefficients
      arranged in SPHARM-PDM's specific order

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the .coef file.

    Returns
    -------
    data : SpharmPdmData
        SPHARM-PDM coefficient data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is invalid or coefficients are malformed.
    """
    path = Path(path)

    # Validate file existence and readability
    if not path.exists():
        raise FileNotFoundError(f"SPHARM-PDM coefficient file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    try:
        with open(path, "r") as f:
            coef_txt = f.read()
    except IOError as e:
        raise IOError(f"Error reading SPHARM-PDM file: {e}")
    if not coef_txt.strip():
        raise ValueError("SPHARM-PDM file is empty")

    try:
        cleaned_txt = coef_txt.translate(str.maketrans("", "", "{}\n"))
        coef = [float(val) for val in cleaned_txt.split(",") if val.strip()]
    except ValueError as e:
        raise ValueError(f"Invalid coefficient format in SPHARM-PDM file: {e}")

    if len(coef) < 1:
        raise ValueError("SPHARM-PDM file must contain at least one value")

    # Validate coefficient count
    # First value is (lmax+1)^2, total coefficients should be 3 * (lmax+1)^2
    n_coef_per_coord = int(coef[0])
    expected_total = 3 * n_coef_per_coord
    actual_count = len(coef) - 1
    if actual_count != expected_total:
        raise ValueError(
            f"Coefficient count mismatch: expected {expected_total} (3 * {n_coef_per_coord}), found {actual_count}"
        )
    # Convert to numpy array and validate dimensions
    try:
        coef_array = np.array(coef[1:]).reshape((n_coef_per_coord, 3))
    except ValueError as e:
        raise ValueError(f"Failed to convert coefficients to numpy array: {e}")

    if coef_array.size == 0:
        raise ValueError("No coefficients found after the count value")

    lmax_plus_one = np.sqrt(n_coef_per_coord)
    if not lmax_plus_one.is_integer():
        raise ValueError(f"Invalid coefficient count: {n_coef_per_coord} != (lmax+1)^2")

    coef = _cvt_spharm_coef_spharmpdm_to_list(coef_array)

    specimen_name = Path(path).stem
    return SpharmPdmData(specimen_name=specimen_name, coeffs=coef)
