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

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt


def read_spharmpdm_coef(path: Union[str, Path]) -> List[npt.NDArray[np.complex128]]:
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
    coef : List[npt.NDArray[np.complex128]]
        List of numpy arrays containing SPHARM coefficients for each coordinate.

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
    except Exception as e:
        raise ValueError(f"Failed to convert coefficients to numpy array: {e}")

    if coef_array.size == 0:
        raise ValueError("No coefficients found after the count value")

    lmax_plus_one = np.sqrt(n_coef_per_coord)
    if not lmax_plus_one.is_integer():
        raise ValueError(f"Invalid coefficient count: {n_coef_per_coord} != (lmax+1)^2")

    coef = cvt_spharm_coef_spharmpdm_to_list(coef_array)

    return coef


def cvt_spharm_coef_spharmpdm_to_list(
    coef_spharmpdm: npt.NDArray[np.float64],
) -> List[npt.NDArray[np.complex128]]:
    """Convert SPHARM-PDM format coefficients to list format.

    SPHARM-PDM stores coefficients in a specific order:
    - For m=0: only real part is stored
    - For m>0: real and imaginary parts are stored separately
    - Complex conjugate symmetry is used for m<0

    Parameters
    ----------
    coef_spharmpdm : np.ndarray of shape ((lmax+1)^2,3)
        Flattened array of SPHARM coefficients in SPHARM-PDM format.
        Contains coefficients for x, y, z coordinates.

    Returns
    -------
    coef_list : list of np.ndarray
        List where coef_list[l] contains coefficients for degree l.
        Each element is an array of shape (2*l+1, 3) with complex values.
        The order is m = -l, -l+1, ..., l-1, l.

    Raises
    ------
    ValueError
        If the input array has invalid shape or dimensions.
    """

    lmax = int(np.sqrt(coef_spharmpdm.shape[0]) - 1)
    if coef_spharmpdm.shape != ((lmax + 1) ** 2, 3):
        raise ValueError(
            f"Invalid coefficient array shape: expected {((lmax + 1) ** 2, 3)}, got {coef_spharmpdm.shape}"
        )

    # Convert to list format
    coef_list = []
    for l in range(lmax + 1):
        coef_l = np.zeros((2 * l + 1, 3), dtype=np.complex128)

        for idx, m in enumerate(range(-l, l + 1)):
            if m == 0:
                # m=0: only real part
                coef_l[idx] = coef_spharmpdm[l**2]
            elif m > 0:
                # m>0: combine real and imaginary parts
                real_idx = l**2 + 2 * m - 1
                imag_idx = l**2 + 2 * m
                coef_l[idx] = (
                    coef_spharmpdm[real_idx] - coef_spharmpdm[imag_idx] * 1j
                ) / 2
            else:
                # m<0: use complex conjugate symmetry
                abs_m = abs(m)
                real_idx = l**2 + 2 * abs_m - 1
                imag_idx = l**2 + 2 * abs_m
                coef_l[idx] = (
                    ((-1) ** m)
                    * (coef_spharmpdm[real_idx] + coef_spharmpdm[imag_idx] * 1j)
                    / 2
                )

        coef_list.append(coef_l)
    return coef_list


def cvt_spharm_coef_list_to_spharmpdm(
    coef_list: List[npt.NDArray[np.complex128]],
) -> npt.NDArray[np.float64]:
    """Convert list format coefficients to SPHARM-PDM format.

    Converts complex spherical harmonic coefficients from standard
    list format to SPHARM-PDM's specific storage format.

    Parameters
    ----------
    coef_list : list of np.ndarray
        List where coef_list[l] contains coefficients for degree l.
        Each element should have shape (2*l+1,) with complex values.
        The order is m = -l, -l+1, ..., l-1, l.

    Returns
    -------
    coef_spharmpdm : np.ndarray of shape ((lmax+1)^2, 3)
        Flattened array of coefficients in SPHARM-PDM format.
        Real and imaginary parts are stored separately.

    Raises
    ------
    ValueError
        If the input list has invalid structure or dimensions.

    Notes
    -----
    The conversion uses the complex conjugate symmetry property:
    Y_l^{-m} = (-1)^m * conj(Y_l^m)
    """
    if not isinstance(coef_list, list):
        raise ValueError("coef_list must be a list")

    if len(coef_list) == 0:
        raise ValueError("coef_list cannot be empty")

    lmax = len(coef_list) - 1

    # Validate structure of coefficient list
    for l, coef_l in enumerate(coef_list):
        expected_len = 2 * l + 1
        if len(coef_l) != expected_len:
            raise ValueError(
                f"coef_list[{l}] has length {len(coef_l)}, expected {expected_len}"
            )

    # Convert to SPHARM-PDM format
    coef_spharmpdm = np.zeros(((lmax + 1) ** 2, 3))
    for l in range(lmax + 1):
        l_squared = l**2

        # m = 0
        coef_spharmpdm[l_squared] = coef_list[l][l].real

        # m > 0
        for m in range(1, l + 1):
            # Get positive and negative m coefficients
            coef_pos_m = coef_list[l][m + l]
            coef_neg_m = coef_list[l][-m + l]
            sign = (-1) ** m

            # Real part: sum of positive and negative m
            coef_spharmpdm[l_squared + 2 * m - 1] = (
                coef_pos_m + sign * coef_neg_m
            ).real

            # Imaginary part: difference of positive and negative m
            coef_spharmpdm[l_squared + 2 * m] = (
                (coef_pos_m - sign * coef_neg_m) * 1j
            ).real

    return coef_spharmpdm
