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
        
    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is invalid or coefficients are malformed.
        
    Examples
    --------
    >>> coef_x, coef_y, coef_z = read_coef_SPHARM_PDM('surface.coef')
    >>> print(coef_x.n_degree)  # Maximum degree l
    15
    """
    # Convert to Path object for better handling
    path = Path(path)
    
    # Check if file exists
    if not path.exists():
        raise FileNotFoundError(f"SPHARM-PDM coefficient file not found: {path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    try:
        with open(path, "r") as f:
            coef_txt = f.read()
    except IOError as e:
        raise IOError(f"Error reading SPHARM-PDM file: {e}")
    # Validate file is not empty
    if not coef_txt.strip():
        raise ValueError("SPHARM-PDM file is empty")
    
    # Parse coefficients more efficiently
    try:
        # Remove curly braces and newlines in one pass
        cleaned_txt = coef_txt.translate(str.maketrans('', '', '{}\n'))
        # Split and convert to float, filtering empty values
        coef = [float(val) for val in cleaned_txt.split(',') if val.strip()]
    except ValueError as e:
        raise ValueError(f"Invalid coefficient format in SPHARM-PDM file: {e}")
    
    # Validate we have at least one coefficient (the count)
    if len(coef) < 1:
        raise ValueError("SPHARM-PDM file must contain at least one value")
    
    # Validate coefficient count
    # First value is (lmax+1)^2, total coefficients should be 3 times that
    n_coef_per_coord = int(coef[0])
    expected_total = 3 * n_coef_per_coord
    actual_count = len(coef) - 1
    if actual_count != expected_total:
        raise ValueError(
            f"Coefficient count mismatch: expected {expected_total} (3 * {n_coef_per_coord}), found {actual_count}"
        )

    # Convert to numpy array and validate dimensions
    try:
        coef_array = np.array(coef[1:])
    except Exception as e:
        raise ValueError(f"Failed to convert coefficients to numpy array: {e}")
    
    if coef_array.size == 0:
        raise ValueError("No coefficients found after the count value")
    
    # Validate that the number of coefficients is consistent with a valid lmax
    # For SPHARM-PDM: total coefficients = 3 * (lmax+1)^2
    n_coef_per_coord = coef_array.size // 3
    if coef_array.size % 3 != 0:
        raise ValueError(
            f"Total coefficient count ({coef_array.size}) must be divisible by 3"
        )
    
    # Check if n_coef_per_coord is a perfect square
    lmax_plus_one = np.sqrt(n_coef_per_coord)
    if not lmax_plus_one.is_integer():
        raise ValueError(
            f"Invalid coefficient count: {n_coef_per_coord} is not a perfect square"
        )
    
    coef_lists = _cvt_spharm_coef_SPHARMPDM_to_list(coef_array)

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


def write_coef_SPHARM_PDM(
    path: Union[str, Path],
    coef_x: SPHARMCoefficients,
    coef_y: SPHARMCoefficients,
    coef_z: SPHARMCoefficients
) -> None:
    """Write SPHARM coefficients to .coef file in SPHARM-PDM format.
    
    This function provides the inverse operation of read_coef_SPHARM_PDM,
    allowing round-trip conversion of SPHARM coefficients.
    
    Parameters
    ----------
    path : str or pathlib.Path
        Path where the .coef file will be written.
    coef_x : SPHARMCoefficients
        SPHARM coefficients for x-coordinate.
    coef_y : SPHARMCoefficients
        SPHARM coefficients for y-coordinate.
    coef_z : SPHARMCoefficients
        SPHARM coefficients for z-coordinate.
        
    Raises
    ------
    ValueError
        If the coefficients have inconsistent degrees or invalid structure.
    IOError
        If the file cannot be written.
        
    Examples
    --------
    >>> # Read coefficients
    >>> coef_x, coef_y, coef_z = read_coef_SPHARM_PDM('input.coef')
    >>> # Process coefficients...
    >>> # Write back to file
    >>> write_coef_SPHARM_PDM('output.coef', coef_x, coef_y, coef_z)
    """
    # Convert to Path object
    path = Path(path)
    
    # Validate input coefficients
    if not all(isinstance(c, SPHARMCoefficients) for c in [coef_x, coef_y, coef_z]):
        raise ValueError("All coefficients must be SPHARMCoefficients objects")
    
    # Check that all coefficients have the same degree
    if not (coef_x.n_degree == coef_y.n_degree == coef_z.n_degree):
        raise ValueError(
            f"All coefficients must have the same degree. Got: "
            f"x={coef_x.n_degree}, y={coef_y.n_degree}, z={coef_z.n_degree}"
        )
    
    if coef_x.n_degree is None:
        raise ValueError("Coefficients have not been initialized")
    
    # Convert coefficients to lists
    coef_x_list = coef_x.as_list()
    coef_y_list = coef_y.as_list()
    coef_z_list = coef_z.as_list()
    
    # Combine into a single array for each degree
    combined_coefs = []
    for l in range(len(coef_x_list)):
        coef_l = np.zeros((2 * l + 1, 3), dtype=np.complex128)
        coef_l[:, 0] = coef_x_list[l]
        coef_l[:, 1] = coef_y_list[l]
        coef_l[:, 2] = coef_z_list[l]
        combined_coefs.append(coef_l)
    
    # Convert to SPHARM-PDM format
    spharm_pdm_array = _cvt_spharm_coef_list_to_SPHARM_PDM(combined_coefs)
    
    # Format output
    n_coef_per_coord = (coef_x.n_degree + 1) ** 2
    
    # Build output string
    output_lines = [f"{{ {n_coef_per_coord},"]
    
    # Format coefficients in groups of 3 (x, y, z)
    for i in range(0, len(spharm_pdm_array), 3):
        x, y, z = spharm_pdm_array[i:i+3]
        output_lines.append(f"{{{x:.6f}, {y:.6f}, {z:.6f}}},")
    
    # Remove trailing comma from last line
    output_lines[-1] = output_lines[-1][:-1]
    
    # Close the outer brace
    output_lines.append("}")
    
    # Write to file
    try:
        with open(path, 'w') as f:
            f.write('\n'.join(output_lines))
    except IOError as e:
        raise IOError(f"Failed to write SPHARM-PDM file: {e}")


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
        
    Raises
    ------
    ValueError
        If the input array has invalid shape or dimensions.
    """
    if coef_SlicerSALT.ndim != 1:
        raise ValueError(
            f"Expected 1D array, got {coef_SlicerSALT.ndim}D array"
        )
    
    if coef_SlicerSALT.size % 3 != 0:
        raise ValueError(
            f"Array size ({coef_SlicerSALT.size}) must be divisible by 3"
        )
    
    coef_ = coef_SlicerSALT.reshape((-1, 3))
    lmax = int(np.sqrt(coef_.shape[0]) - 1)
    
    # Validate that we have the correct number of coefficients
    expected_size = (lmax + 1) ** 2
    if coef_.shape[0] != expected_size:
        raise ValueError(
            f"Invalid coefficient array size: expected {expected_size}, got {coef_.shape[0]}"
        )
    # Pre-allocate list for better performance
    coef_list = []
    
    for l in range(lmax + 1):
        # Pre-allocate array for this degree
        coef_l = np.zeros(2 * l + 1, dtype=np.complex128)
        
        for idx, m in enumerate(range(-l, l + 1)):
            if m == 0:
                # m=0: only real part
                coef_l[idx] = coef_[l**2]
            elif m > 0:
                # m>0: combine real and imaginary parts
                real_idx = l**2 + 2 * m - 1
                imag_idx = l**2 + 2 * m
                coef_l[idx] = (coef_[real_idx] - coef_[imag_idx] * 1j) / 2
            else:
                # m<0: use complex conjugate symmetry
                abs_m = abs(m)
                real_idx = l**2 + 2 * abs_m - 1
                imag_idx = l**2 + 2 * abs_m
                coef_l[idx] = ((-1) ** m) * (coef_[real_idx] + coef_[imag_idx] * 1j) / 2
        
        coef_list.append(coef_l)
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
    # Pre-allocate output array
    coef_SlicerSALT = np.zeros(((lmax + 1) ** 2, 3))
    
    for l in range(lmax + 1):
        l_squared = l ** 2
        
        # m = 0 case
        coef_SlicerSALT[l_squared] = coef_list[l][l].real
        
        # m > 0 cases
        for m in range(1, l + 1):
            # Get positive and negative m coefficients
            coef_pos_m = coef_list[l][m + l]
            coef_neg_m = coef_list[l][-m + l]
            sign = (-1) ** m
            
            # Real part: sum of positive and negative m
            coef_SlicerSALT[l_squared + 2 * m - 1] = (
                coef_pos_m + sign * coef_neg_m
            ).real
            
            # Imaginary part: difference of positive and negative m
            coef_SlicerSALT[l_squared + 2 * m] = (
                (coef_pos_m - sign * coef_neg_m) * 1j
            ).real

    return coef_SlicerSALT.reshape(-1)
