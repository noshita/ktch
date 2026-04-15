"""Conversion functions between file-format and processing-ready representations."""

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

import numpy as np
import pandas as pd

#
# NEF <-> EFA coefficient
#


def nef_to_efa_coeffs(nef_data, dc_offset=None):
    """Convert NefData to EFA-compatible flat coefficient vectors.

    Parameters
    ----------
    nef_data : NefData or list of NefData
        Normalized EFD data read by :func:`read_nef`.
    dc_offset : array-like of shape (4,) or (n_samples, 4), optional
        DC offset ``[a_0, b_0, c_0, d_0]`` to prepend.
        Defaults to zeros.

    Returns
    -------
    coeffs : np.ndarray of shape (n_samples, 4 * (n_harmonics + 1))
        Flat coefficient vectors compatible with
        :meth:`~ktch.harmonic.EllipticFourierAnalysis.inverse_transform`.
        Layout: ``[a_0..a_n, b_0..b_n, c_0..c_n, d_0..d_n]``.
    """
    if not isinstance(nef_data, list):
        nef_data = [nef_data]

    n_samples = len(nef_data)
    n_harmonics = nef_data[0].coeffs.shape[0]

    if dc_offset is None:
        dc_offset = np.zeros((n_samples, 4))
    else:
        dc_offset = np.atleast_2d(dc_offset)
        if dc_offset.shape[0] == 1 and n_samples > 1:
            dc_offset = np.tile(dc_offset, (n_samples, 1))

    result = np.empty((n_samples, 4 * (n_harmonics + 1)))
    for i, nef in enumerate(nef_data):
        # nef.coeffs shape: (n_harmonics, 4), columns [a, b, c, d]
        # EFA flat layout: [a_0..a_n, b_0..b_n, c_0..c_n, d_0..d_n]
        for ax in range(4):
            result[i, ax * (n_harmonics + 1)] = dc_offset[i, ax]
            result[i, ax * (n_harmonics + 1) + 1 : (ax + 1) * (n_harmonics + 1)] = (
                nef.coeffs[:, ax]
            )

    return result


def efa_coeffs_to_nef(coeffs, specimen_names=None, n_dim=2):
    """Convert EFA flat coefficient vectors to NefData objects.

    Parameters
    ----------
    coeffs : np.ndarray of shape (n_samples, n_features) or (n_features,)
        Flat EFA coefficient vectors from
        :meth:`~ktch.harmonic.EllipticFourierAnalysis.transform`.
        Trailing orientation/scale columns are stripped automatically.
    specimen_names : list of str, optional
        Specimen names. Defaults to ``"Specimen_0"``, ``"Specimen_1"``, etc.
    n_dim : int, default=2
        Number of dimensions. Only ``n_dim=2`` is supported.

    Returns
    -------
    nef_list : list of NefData
        One :class:`NefData` per sample, ready for :func:`write_nef`.

    Raises
    ------
    ValueError
        If ``n_dim`` is not 2.
    """
    if n_dim != 2:
        raise ValueError(f"Only 2D EFA coefficients are supported, got n_dim={n_dim}.")

    # Lazy import to avoid circular dependency
    from ._nef import NefData

    coeffs = np.atleast_2d(coeffs)
    n_samples = coeffs.shape[1]

    n_axes = 4  # a, b, c, d for 2D
    # Determine n_harmonics: strip orientation/scale if present
    # 2D with orientation/scale: 4*(n+1) + 2
    # 2D without: 4*(n+1)
    n_features = coeffs.shape[1]
    extra_2d = 2  # psi, scale

    if n_features % n_axes == 0:
        n_harmonics_plus_1 = n_features // n_axes
    elif (n_features - extra_2d) % n_axes == 0:
        n_harmonics_plus_1 = (n_features - extra_2d) // n_axes
        coeffs = coeffs[:, : n_axes * n_harmonics_plus_1]
    else:
        raise ValueError(
            f"Cannot parse EFA coefficient vector of length {n_features} "
            f"for n_dim=2. Expected 4*(n+1) or 4*(n+1)+2."
        )

    n_harmonics = n_harmonics_plus_1 - 1

    if specimen_names is None:
        specimen_names = [f"Specimen_{i}" for i in range(coeffs.shape[0])]

    nef_list = []
    for i in range(coeffs.shape[0]):
        # Reshape flat [a_0..a_n, b_0..b_n, c_0..c_n, d_0..d_n]
        # to (4, n+1), drop DC (column 0), transpose to (n, 4)
        coef_matrix = coeffs[i].reshape(n_axes, n_harmonics_plus_1)
        # Drop DC offset (index 0 along axis 1)
        coef_matrix = coef_matrix[:, 1:].T  # (n_harmonics, 4)

        nef_list.append(
            NefData(
                specimen_name=specimen_names[i],
                coeffs=coef_matrix,
            )
        )

    return nef_list


#
# SPHARM-PDM <-> SHA coefficient
#
# The following helpers convert between complex and real SH coefficient
# representations.  They duplicate the logic in
# ``ktch.harmonic._spherical_harmonic_analysis`` so that the io module
# does not depend on the harmonic module.
#


def _complex_to_real_sph_coef(coef_complex):
    """Convert complex SH coefficients to real SH coefficients.

    Includes the ``(-1)^m`` Condon-Shortley phase factor.
    """
    coef_real = np.empty_like(coef_complex, dtype=np.float64)
    n_coef = coef_complex.shape[0]
    l_max = int(np.sqrt(n_coef)) - 1

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            idx = l**2 + l + m
            if m == 0:
                coef_real[idx] = np.real(coef_complex[idx])
            elif m > 0:
                coef_real[idx] = np.sqrt(2) * (-1) ** m * np.real(coef_complex[idx])
            else:
                idx_pos = l**2 + l + (-m)
                coef_real[idx] = -np.sqrt(2) * (-1) ** abs(m) * np.imag(
                    coef_complex[idx_pos]
                )

    return coef_real


def _real_to_complex_sph_coef(coef_real):
    """Convert real SH coefficients to complex SH coefficients.

    Inverse of ``_complex_to_real_sph_coef``.
    """
    coef_complex = np.empty_like(coef_real, dtype=np.complex128)
    n_coef = coef_real.shape[0]
    l_max = int(np.sqrt(n_coef)) - 1

    for l in range(l_max + 1):
        idx_0 = l**2 + l
        coef_complex[idx_0] = coef_real[idx_0] + 0j

        for m in range(1, l + 1):
            idx_pos = l**2 + l + m
            idx_neg = l**2 + l - m
            c_pos = (
                (-1) ** m
                * (coef_real[idx_pos] - 1j * coef_real[idx_neg])
                / np.sqrt(2)
            )
            coef_complex[idx_pos] = c_pos
            coef_complex[idx_neg] = (-1) ** m * np.conj(c_pos)

    return coef_complex


def spharmpdm_to_sha_coeffs(spharmpdm_data):
    """Convert SpharmPdmData to SHA-compatible flat coefficient vectors.

    The output contains real-valued coefficients with respect to the
    standard orthonormal real spherical harmonic basis used by
    :class:`~ktch.harmonic.SphericalHarmonicAnalysis`.

    Parameters
    ----------
    spharmpdm_data : SpharmPdmData or list of SpharmPdmData
        SPHARM-PDM data read by :func:`read_spharmpdm_coef`.

    Returns
    -------
    coeffs : np.ndarray of shape (n_samples, 3 * (l_max + 1)**2), float64
        Flat coefficient vectors compatible with
        :meth:`~ktch.harmonic.SphericalHarmonicAnalysis.inverse_transform`.
        Layout: ``[cx_0_0, cx_1_-1, ..., cy_0_0, ..., cz_0_0, ...]``
        (axis-major, then by degree and order).
    """
    if not isinstance(spharmpdm_data, list):
        spharmpdm_data = [spharmpdm_data]

    n_samples = len(spharmpdm_data)
    l_max = spharmpdm_data[0].l_max
    n_coeffs_per_axis = (l_max + 1) ** 2

    result = np.empty((n_samples, 3 * n_coeffs_per_axis))
    for i, data in enumerate(spharmpdm_data):
        # SpharmPdmData.coeffs: complex list (SPHARM-PDM convention)
        # Convert to real SH coefficients
        stacked = np.vstack(data.coeffs)  # ((l_max+1)^2, 3), complex
        real_coef = _complex_to_real_sph_coef(stacked)  # float64
        result[i] = real_coef.T.ravel()

    return result


def sha_coeffs_to_spharmpdm(coeffs, specimen_names=None):
    """Convert SHA flat coefficient vectors to SpharmPdmData objects.

    The input should contain real-valued coefficients from
    :class:`~ktch.harmonic.SphericalHarmonicAnalysis`.
    The output ``SpharmPdmData.coeffs`` uses complex coefficients
    matching the SPHARM-PDM convention.

    Parameters
    ----------
    coeffs : np.ndarray of shape (n_samples, 3 * (l_max + 1)**2)
        Flat SHA coefficient vectors from
        :meth:`~ktch.harmonic.SphericalHarmonicAnalysis.transform`.
    specimen_names : list of str, optional
        Specimen names. Defaults to ``"Specimen_0"``, ``"Specimen_1"``, etc.

    Returns
    -------
    spharmpdm_list : list of SpharmPdmData
        One :class:`SpharmPdmData` per sample.

    Raises
    ------
    ValueError
        If the coefficient vector length is not divisible by 3 or the
        per-axis count is not a perfect square.
    """
    # Lazy import to avoid circular dependency
    from ._spharm_pdm import SpharmPdmData

    coeffs = np.atleast_2d(coeffs)
    n_samples, n_features = coeffs.shape

    if n_features % 3 != 0:
        raise ValueError(
            f"Coefficient vector length {n_features} is not divisible by 3."
        )

    n_coeffs_per_axis = n_features // 3
    l_max_plus_one = np.sqrt(n_coeffs_per_axis)
    if not l_max_plus_one.is_integer():
        raise ValueError(
            f"Per-axis coefficient count {n_coeffs_per_axis} is not a perfect "
            f"square ((l_max+1)^2)."
        )
    l_max = int(l_max_plus_one) - 1

    if specimen_names is None:
        specimen_names = [f"Specimen_{i}" for i in range(n_samples)]

    spharmpdm_list = []
    for i in range(n_samples):
        # Reshape axis-major flat to ((l_max+1)^2, 3), real
        stacked_real = coeffs[i].reshape(3, n_coeffs_per_axis).T
        # Convert real SH → complex SH for SpharmPdmData
        stacked_complex = _real_to_complex_sph_coef(stacked_real)

        # Split into degree-indexed list
        coef_list = []
        for l in range(l_max + 1):
            start = l**2
            end = (l + 1) ** 2
            coef_list.append(stacked_complex[start:end])

        spharmpdm_list.append(
            SpharmPdmData(specimen_name=specimen_names[i], coeffs=coef_list)
        )

    return spharmpdm_list


#
# SPHARM-PDM format packing (private)
#


def _cvt_spharm_coef_spharmpdm_to_list(
    coef_spharmpdm: np.ndarray,
) -> list[np.ndarray]:
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


def _cvt_spharm_coef_list_to_spharmpdm(
    coef_list: list[np.ndarray],
) -> np.ndarray:
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


#
# Coordinate DataFrame conversion utilities
#
def convert_coords_df_to_list(df_coords: pd.DataFrame) -> list[np.ndarray]:
    """Convert a coordinate DataFrame to a list of per-specimen arrays.

    Bridges the DataFrame output of :func:`read_tps(as_frame=True)` or
    :func:`read_chc(as_frame=True)` to the list format expected by
    :class:`~ktch.harmonic.EllipticFourierAnalysis`.

    Parameters
    ----------
    df_coords : pd.DataFrame
        DataFrame with ``MultiIndex (specimen_id, coord_id)`` and
        columns ``(x, y [, z])``.

    Returns
    -------
    coords_list : list of np.ndarray
        Each element has shape ``(n_coords_i, n_dim)``.
    """
    dim = df_coords.shape[1]
    coords_list = [
        df_coords.loc[specimen_id].to_numpy().reshape(-1, dim)
        for specimen_id in df_coords.index.get_level_values(0).unique()
    ]
    return coords_list


def convert_coords_df_to_df_sklearn_transform(
    df_coords: pd.DataFrame,
) -> pd.DataFrame:
    """Convert a coordinate DataFrame to sklearn-compatible wide format.

    Parameters
    ----------
    df_coords : pd.DataFrame
        DataFrame with ``MultiIndex (specimen_id, coord_id)`` and
        columns ``(x, y [, z])``.

    Returns
    -------
    df_wide : pd.DataFrame
        DataFrame with index ``specimen_id`` and columns
        ``(coord_id, axis)``, compatible with sklearn transformers.
    """
    df_coords_new = df_coords.unstack().swaplevel(axis=1).sort_index(axis=1)
    return df_coords_new
