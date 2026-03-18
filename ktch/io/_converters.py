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
