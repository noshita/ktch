"""Normalized EFD file I/O functions."""

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

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ._protocols import MorphoDataMixin


@dataclass(repr=False)
class NefData(MorphoDataMixin):
    """Normalized elliptic Fourier descriptor data.

    Stores the normalized EFD coefficients produced by
    SHAPE's Chc2Nef program.
    Each harmonic *n* has four coefficients (a_n, b_n, c_n, d_n).

    Parameters
    ----------
    specimen_name : str
        Specimen name (e.g. ``"Sample1_1"``).
    coeffs : np.ndarray
        Coefficient matrix with shape ``(n_harmonics, 4)``.
        Columns are ``[a, b, c, d]`` for each harmonic (1 .. n).
    const_flags : tuple of int
        Four flags indicating which first-harmonic coefficients are
        constant after normalization.  ``(1, 1, 1, 0)`` for
        first-ellipse normalization (a1=1, b1=0, c1=0, d1 varies).
    """

    specimen_name: str
    coeffs: np.ndarray
    const_flags: tuple[int, int, int, int] = (1, 1, 1, 0)

    @property
    def sample_name(self) -> str:
        """Deprecated. Use ``specimen_name`` instead."""
        warnings.warn(
            "sample_name is deprecated, use specimen_name",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.specimen_name

    def __post_init__(self):
        if not isinstance(self.coeffs, np.ndarray):
            self.coeffs = np.asarray(self.coeffs, dtype=float)
        if self.coeffs.ndim != 2 or self.coeffs.shape[1] != 4:
            raise ValueError(
                f"coeffs must have shape (n_harmonics, 4), got {self.coeffs.shape}"
            )

    @property
    def n_harmonics(self):
        """Number of harmonics."""
        return self.coeffs.shape[0]

    def _repr_detail(self):
        return f"n_harmonics={self.n_harmonics}"

    @classmethod
    def from_efa_coeffs(cls, coeffs, specimen_name=""):
        """Create NefData from a 2D EFA flat coefficient vector.

        Parameters
        ----------
        coeffs : np.ndarray of shape (n_features,)
            Flat EFA coefficient vector (with or without orientation/scale).
        specimen_name : str
            Specimen name for the record.

        Returns
        -------
        nef_data : NefData
        """
        from ._converters import efa_coeffs_to_nef

        result = efa_coeffs_to_nef(
            np.atleast_2d(coeffs), specimen_names=[specimen_name]
        )
        return result[0]

    def to_numpy(self):
        """Return the coefficient matrix.

        Returns
        -------
        coeffs : np.ndarray
            Coefficient matrix with shape ``(n_harmonics, 4)``.
            Columns are ``[a, b, c, d]``.
        """
        return self.coeffs

    def to_dataframe(self):
        """Return the coefficients as a DataFrame.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with columns ``a``, ``b``, ``c``, ``d``.
            Index is a MultiIndex ``(specimen_id, harmonic)``.
        """
        harmonics = np.arange(1, self.n_harmonics + 1)
        df = pd.DataFrame(
            self.coeffs,
            columns=["a", "b", "c", "d"],
            index=pd.MultiIndex.from_arrays(
                [
                    [self.specimen_name] * self.n_harmonics,
                    harmonics,
                ],
                names=["specimen_id", "harmonic"],
            ),
        )
        return df


def read_nef(file_path, as_frame=False):
    """Read normalized EFD (.nef) file.

    The .nef format stores normalized elliptic Fourier coefficients
    produced by SHAPE's Chc2Nef program::

        #CONST [a1_const] [b1_const] [c1_const] [d1_const]
        #HARMO [n_harmonics]
        [Sample name]
          [a1] [b1] [c1] [d1]
          [a2] [b2] [c2] [d2]
          ...

    Parameters
    ----------
    file_path : str or Path
        Path to the ``.nef`` file.
    as_frame : bool, default=False
        If True, return a :class:`~pandas.DataFrame`.

    Returns
    -------
    result : NefData or list of NefData or pd.DataFrame
        If a single record and ``as_frame=False``, returns a
        :class:`NefData`.  If multiple records and ``as_frame=False``,
        returns a list of :class:`NefData`.  If ``as_frame=True``,
        returns a concatenated DataFrame.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    if path.suffix.lower() != ".nef":
        raise ValueError(f"{path} is not a .nef file.")

    with open(path, "r") as f:
        lines = [line.rstrip("\n") for line in f]

    const_flags = (1, 1, 1, 0)
    n_harmonics = None
    data_start = 0

    # Parse optional headers
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#CONST"):
            parts = stripped.split()
            const_flags = tuple(int(x) for x in parts[1:5])
            data_start = i + 1
        elif stripped.startswith("#HARMO"):
            parts = stripped.split()
            n_harmonics = int(parts[1])
            data_start = i + 1
        elif stripped.startswith("#"):
            data_start = i + 1
        else:
            break

    nef_data_list = []
    i = data_start

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        sample_name = line
        i += 1

        # Read coefficient lines
        coeff_rows = []
        while i < len(lines):
            row_line = lines[i].strip()
            if not row_line:
                i += 1
                continue
            parts = row_line.split()
            try:
                row = [float(x) for x in parts]
            except ValueError:
                break
            if len(row) != 4:
                break
            coeff_rows.append(row)
            i += 1

        if not coeff_rows:
            raise ValueError(
                f"No coefficients found for record {sample_name!r} in {path}"
            )

        coeffs = np.array(coeff_rows, dtype=float)

        if n_harmonics is not None and coeffs.shape[0] != n_harmonics:
            raise ValueError(
                f"Expected {n_harmonics} harmonics for record "
                f"{sample_name!r}, got {coeffs.shape[0]} in {path}"
            )

        nef_data_list.append(
            NefData(
                specimen_name=sample_name,
                coeffs=coeffs,
                const_flags=const_flags,
            )
        )

    if len(nef_data_list) == 1:
        if as_frame:
            return nef_data_list[0].to_dataframe()
        return nef_data_list[0]
    else:
        if as_frame:
            return pd.concat([d.to_dataframe() for d in nef_data_list])
        return nef_data_list


def write_nef(
    file_path,
    coeffs,
    sample_names=None,
    const_flags=None,
):
    """Write normalized EFD coefficients to a .nef file.

    Parameters
    ----------
    file_path : str or Path
        Path to the output ``.nef`` file.
    coeffs : np.ndarray or list of np.ndarray
        Coefficient matrices with shape ``(n_harmonics, 4)`` each.
        A single 2-D array is treated as one record.
    sample_names : str or list of str, optional
        Sample names.  Defaults to ``"Sample"``.
    const_flags : tuple of int, optional
        Four flags for the ``#CONST`` header.
        Defaults to ``(1, 1, 1, 0)``.
    """
    path = Path(file_path)

    if isinstance(coeffs, np.ndarray):
        if coeffs.ndim == 2:
            coeffs = [coeffs]
        elif coeffs.ndim == 3:
            coeffs = [coeffs[i] for i in range(coeffs.shape[0])]
        else:
            raise ValueError("coeffs must be a 2-D or 3-D array.")
    elif not isinstance(coeffs, list):
        raise ValueError("coeffs must be a numpy array or a list of numpy arrays.")

    n_samples = len(coeffs)

    if sample_names is None:
        sample_names = ["Sample"] * n_samples
    elif isinstance(sample_names, str):
        sample_names = [sample_names] * n_samples

    if const_flags is None:
        const_flags = (1, 1, 1, 0)

    if len(sample_names) != n_samples:
        raise ValueError(
            f"Length of sample_names ({len(sample_names)}) does not match "
            f"number of coefficient matrices ({n_samples})."
        )

    n_harmonics = coeffs[0].shape[0] if len(coeffs) > 0 else 0

    with open(path, "w") as f:
        f.write(
            f"#CONST {const_flags[0]} {const_flags[1]} "
            f"{const_flags[2]} {const_flags[3]}\n"
        )
        f.write(f"#HARMO {n_harmonics}\n")

        for i in range(n_samples):
            c = np.asarray(coeffs[i], dtype=float)
            if c.ndim != 2 or c.shape[1] != 4:
                raise ValueError(
                    f"coeffs[{i}] must have shape (n_harmonics, 4), got {c.shape}"
                )
            f.write(f"{sample_names[i]}\n")
            for row in c:
                f.write(
                    f"  {row[0]: .7e}  {row[1]: .7e}  {row[2]: .7e}  {row[3]: .7e}\n"
                )
