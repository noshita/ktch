"""SPHARM-PDM file I/O functions"""

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

from ..outline import SPHARMCoefficients


def read_coef_SPHARM_PDM(path: Union[str, Path]) -> Tuple[SPHARMCoefficients, SPHARMCoefficients, SPHARMCoefficients]:
    """Read .coef file of SPHARM-PDM.
    
    The .coef file is an output of the `ParaToSPHARMMesh` step
    of `SPHARM-PDM <https://www.nitrc.org/projects/spharm-pdm>`_,
    and contains SPHARM coefficients for a 3D surface.
    
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
    coef_x : SPHARMCoefficients
        SPHARM coefficients for x-coordinate.
    coef_y : SPHARMCoefficients
        SPHARM coefficients for y-coordinate.
    coef_z : SPHARMCoefficients
        SPHARM coefficients for z-coordinate.
        
    Examples
    --------
    >>> coef_x, coef_y, coef_z = read_coef_SPHARM_PDM('surface.coef')
    >>> print(coef_x.n_degree)  # Maximum degree l
    15
    """
    with open(path, "r") as f:
        coef_txt = f.read()
    coef = [
        float(val)
        for val in coef_txt.replace("\n", "")
        .replace("{", "")
        .replace("}", "")
        .split(",")
    ]

    coef_lists = _cvt_spharm_coef_SPHARMPDM_to_list(np.array(coef[1:]))

    coefficients = [
        (
            [
                [coef_lists[l][m_, i] for m_ in range(2 * l + 1)]
                for l in range(len(coef_lists))
            ]
        )
        for i in range(3)
    ]

    coef_x = SPHARMCoefficients()
    coef_y = SPHARMCoefficients()
    coef_z = SPHARMCoefficients()
    coef_x.from_list(coefficients[0])
    coef_y.from_list(coefficients[1])
    coef_z.from_list(coefficients[2])

    return coef_x, coef_y, coef_z


def _cvt_spharm_coef_SPHARMPDM_to_list(coef_SlicerSALT: npt.NDArray[np.float64]) -> List[npt.NDArray[np.complex128]]:
    """Convert SPHARM-PDM format coefficients to list format.
    
    SPHARM-PDM stores coefficients in a specific order:
    - For m=0: only real part is stored
    - For m>0: real and imaginary parts are stored separately
    - Complex conjugate symmetry is used for m<0
    
    Parameters
    ----------
    coef_SlicerSALT : np.ndarray of shape ((lmax+1)^2 * 3,)
        Flattened array of SPHARM coefficients in SPHARM-PDM format.
        Contains coefficients for x, y, z coordinates.
        
    Returns
    -------
    coef_list : list of np.ndarray
        List where coef_list[l] contains coefficients for degree l.
        Each element is an array of shape (2*l+1, 3) with complex values.
        The order is m = -l, -l+1, ..., l-1, l.
    """
    coef_ = coef_SlicerSALT.reshape((-1, 3))
    lmax = int(np.sqrt(coef_.shape)[0] - 1)
    coef_list = [
        np.array(
            [
                coef_[l**2]
                if m == 0
                else (coef_[l**2 + 2 * m - 1] - coef_[l**2 + 2 * m] * 1j) / 2
                if m > 0
                else ((-1) ** m)
                * (
                    coef_[l**2 + 2 * np.abs(m) - 1]
                    + coef_[l**2 + 2 * np.abs(m)] * 1j
                )
                / 2
                for m in range(-l, l + 1, 1)
            ]
        )
        for l in range(0, lmax + 1, 1)
    ]
    return coef_list


def _cvt_spharm_coef_list_to_SPHARM_PDM(coef_list: List[npt.NDArray[np.complex128]]) -> npt.NDArray[np.float64]:
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
    coef_SlicerSALT : np.ndarray of shape ((lmax+1)^2 * 3,)
        Flattened array of coefficients in SPHARM-PDM format.
        Real and imaginary parts are stored separately.
        
    Notes
    -----
    The conversion uses the complex conjugate symmetry property:
    Y_l^{-m} = (-1)^m * conj(Y_l^m)
    """
    lmax = len(coef_list) - 1
    coef_SlicerSALT = np.zeros(((lmax + 1) ** 2, 3))
    for l in range(0, lmax + 1, 1):
        for m in range(0, l + 1, 1):
            if m == 0:
                coef_SlicerSALT[l**2] = coef_list[l][m + l].real
            else:
                coef_SlicerSALT[l**2 + 2 * m - 1] = (
                    coef_list[l][m + l] + ((-1) ** m) * coef_list[l][-m + l]
                ).real
                coef_SlicerSALT[l**2 + 2 * m] = (
                    (coef_list[l][m + l] - ((-1) ** m) * coef_list[l][-m + l]) * 1j
                ).real

    return coef_SlicerSALT.reshape(-1)
