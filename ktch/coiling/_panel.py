"""Input normalization for the coiling estimators.

A coiling specimen is a variable-length sequence of measured points,
optionally with domain coordinates and per-specimen scalars.
"""

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

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass(frozen=True)
class _Panel:
    """Normalized per-specimen panel of variable-length sample sequences.

    Holds one sequence per specimen: a 2D array whose rows are ordered sample
    points and whose columns are the dependent per-point channels.

    Attributes
    ----------
    values : list of ndarray
        One ``(n_points_i, n_channels)`` array per specimen.
    channel_names : list of str
        Names of the per-point channels; ``len(channel_names) == n_channels``.
    domain_coords : list of ndarray or None
        Per-point domain coordinate values. One array per specimen, each with
        ``n_points_i`` rows. ``None`` when not supplied.
    domain_coord_names : list of str or None
        Names of the domain-coordinate columns. ``None`` for list/array input.
    meta : ndarray of shape (n_samples, n_meta) or None
        Per-specimen scalars. ``None`` when not supplied.
    meta_names : list of str or None
        Names of the per-specimen scalar columns.
    """

    values: list[npt.NDArray[np.float64]]
    channel_names: list[str]
    domain_coords: list[npt.NDArray[np.float64]] | None = None
    domain_coord_names: list[str] | None = None
    meta: npt.NDArray[np.float64] | None = None
    meta_names: list[str] | None = None

    @property
    def n_samples(self) -> int:
        """Number of specimens in the panel."""
        return len(self.values)

    @property
    def n_channels(self) -> int:
        """Number of dependent per-point channels."""
        return len(self.channel_names)


def _unflatten_row(
    row: npt.NDArray[np.float64], n_channels: int
) -> npt.NDArray[np.float64]:
    """Recover a ``(n_points, n_channels)`` sequence from a flat padded row.

    The flat layout is channel blocked with trailing NaN padding: the valid
    values are ``[c0_0..c0_{n-1}, c1_0..c1_{n-1}, ...]`` followed by NaNs.

    Parameters
    ----------
    row : ndarray
        One flat, NaN-padded sample row.
    n_channels : int
        Number of channels (columns of the recovered sequence).

    Returns
    -------
    ndarray of shape (n_points, n_channels)
    """
    valid = row[~np.isnan(row)]
    if valid.size % n_channels != 0:
        raise ValueError(
            f"Flat sample length after dropping NaNs ({valid.size}) is not a "
            f"multiple of n_channels ({n_channels}); the row does not match "
            "the channel-blocked layout."
        )
    return valid.reshape(n_channels, -1).T


def _panel_to_flat(
    values: list[npt.ArrayLike],
) -> npt.NDArray[np.float64]:
    """Pack variable-length sequences into a NaN-padded flat 2D array.

    Each sequence is flattened channel-blocked (``v.T.ravel()``) and
    right-padded with NaN to a common width ``n_channels * max_points``.

    Parameters
    ----------
    values : list of array-like
        One ``(n_points_i, n_channels)`` sequence per specimen.

    Returns
    -------
    ndarray of shape (n_samples, n_channels * max_points)
    """
    arrs = [np.asarray(v, dtype=float) for v in values]
    if not arrs:
        return np.empty((0, 0), dtype=float)
    n_channels = arrs[0].shape[1]
    max_points = max(v.shape[0] for v in arrs)
    out = np.full((len(arrs), n_channels * max_points), np.nan, dtype=float)
    for i, v in enumerate(arrs):
        flat = v.T.ravel()
        out[i, : flat.size] = flat
    return out


def _check_panel(
    X,
    *,
    channel_names: list[str],
    domain_coords=None,
    domain_coord_names: list[str] | None = None,
    meta=None,
    meta_names: list[str] | None = None,
) -> _Panel:
    """Normalize a panel input to a :class:`_Panel`.

    Accepts these encodings of a per-specimen panel and normalizes them to a
    common representation:

    - a ragged ``list`` of ``(n_points_i, n_channels)`` arrays;
    - a NaN-padded flat 2D ``ndarray`` of shape
      ``(n_samples, n_channels * max_points)`` (channel blocked), or a
      single-index ``DataFrame`` carrying such rows;
    - a long ``MultiIndex`` ``DataFrame`` indexed by ``(specimen, point)`` with
      the channels (and optionally the domain coordinates and per-specimen
      scalars) as named columns.

    Parameters
    ----------
    X : list of array-like, ndarray, or DataFrame
        The panel in one of the accepted encodings.
    channel_names : list of str
        Names and count of the dependent per-point channels.
    domain_coords : list of array-like or None
        Per-point domain coordinate values for the list/flat encodings.
    domain_coord_names : list of str or None
        Columns holding the domain coordinates for the long-DataFrame encoding.
    meta : array-like of shape (n_samples, n_meta) or None
        Per-specimen scalars for the list/flat encodings.
    meta_names : list of str or None
        Columns holding the per-specimen scalars for the long-DataFrame
        encoding.

    Returns
    -------
    _Panel

    Raises
    ------
    ValueError
        For an unrecognized encoding, a wrong channel count, a length mismatch
        between the domain coordinates (or meta) and the panel, or a column
        named in ``channel_names`` / ``domain_coord_names`` / ``meta_names``
        that is absent from a long DataFrame.
    NotImplementedError
        For the equal-length 3D-array encoding and for DataFrames with more
        than two index levels (multiple trajectories), both deferred.
    """
    channel_names = list(channel_names)
    n_channels = len(channel_names)
    if n_channels < 1:
        raise ValueError("channel_names must name at least one channel.")

    if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.MultiIndex):
        values, domain_coords, meta = _panel_from_long_dataframe(
            X,
            channel_names=channel_names,
            domain_coord_names=domain_coord_names,
            meta_names=meta_names,
            domain_coords=domain_coords,
            meta=meta,
        )
    else:
        if domain_coord_names is not None:
            raise ValueError(
                "domain_coord_names applies to a long MultiIndex DataFrame "
                "only; for list/array input pass domain_coords directly."
            )
        if meta_names is not None:
            raise ValueError(
                "meta_names applies to a long MultiIndex DataFrame only; for "
                "list/array input pass meta directly."
            )
        values = _panel_values_from_list_or_flat(X, n_channels=n_channels)

    _validate_channels(values, n_channels=n_channels)
    domain_coords = _validate_domain_coords(values, domain_coords)
    meta = _validate_meta(values, meta)

    return _Panel(
        values=values,
        channel_names=channel_names,
        domain_coords=domain_coords,
        domain_coord_names=(
            list(domain_coord_names) if domain_coord_names is not None else None
        ),
        meta=meta,
        meta_names=list(meta_names) if meta_names is not None else None,
    )


def _panel_values_from_list_or_flat(
    X, *, n_channels: int
) -> list[npt.NDArray[np.float64]]:
    """Extract per-specimen sequences from list / flat-2D / wide-DataFrame X."""
    if isinstance(X, pd.DataFrame):
        arr = X.to_numpy(dtype=float)
        return [_unflatten_row(row, n_channels) for row in arr]

    if isinstance(X, np.ndarray):
        if X.ndim == 3:
            raise NotImplementedError(
                "The equal-length 3D-array encoding "
                "(n_samples, n_points, n_channels) is not supported yet; pass "
                "a list of arrays, a NaN-padded flat 2D array, or a long "
                "DataFrame."
            )
        if X.ndim != 2:
            raise ValueError(
                f"An ndarray panel must be 2D (NaN-padded flat rows); got {X.ndim}D."
            )
        return [_unflatten_row(row, n_channels) for row in X.astype(float)]

    if isinstance(X, (list, tuple)):
        return [np.asarray(x, dtype=float) for x in X]

    raise ValueError(
        "Unrecognized panel encoding: expected a list of arrays, a 2D ndarray, "
        f"or a DataFrame; got {type(X).__name__}."
    )


def _panel_from_long_dataframe(
    X: pd.DataFrame,
    *,
    channel_names: list[str],
    domain_coord_names: list[str] | None,
    meta_names: list[str] | None,
    domain_coords,
    meta,
):
    """Extract a panel from a long MultiIndex DataFrame ``(specimen, point)``."""
    if domain_coords is not None or meta is not None:
        raise ValueError(
            "For a long MultiIndex DataFrame, select the domain coordinates "
            "and per-specimen scalars via domain_coord_names / meta_names, not "
            "the domain_coords / meta arguments."
        )
    if X.index.nlevels != 2:
        raise NotImplementedError(
            "Long DataFrames with more than two index levels are not supported "
            "yet; use a 2-level (specimen, point) index."
        )

    needed = list(channel_names)
    needed += list(domain_coord_names) if domain_coord_names else []
    needed += list(meta_names) if meta_names else []
    missing = [c for c in needed if c not in X.columns]
    if missing:
        raise ValueError(f"Columns absent from the DataFrame: {missing}.")

    groups = [g for _, g in X.groupby(level=0, sort=False)]
    values = [g[channel_names].to_numpy(dtype=float) for g in groups]

    domain_coords = (
        [g[domain_coord_names].to_numpy(dtype=float) for g in groups]
        if domain_coord_names
        else None
    )
    if meta_names:
        meta = np.array([g[meta_names].iloc[0].to_numpy(dtype=float) for g in groups])
    else:
        meta = None

    return values, domain_coords, meta


def _validate_channels(values, *, n_channels: int) -> None:
    """Check every sequence is 2D with the expected channel count."""
    for i, v in enumerate(values):
        v = np.asarray(v)
        if v.ndim != 2:
            raise ValueError(
                f"Sample {i} must be a 2D (n_points, n_channels) array; got {v.ndim}D."
            )
        if v.shape[1] != n_channels:
            raise ValueError(
                f"Sample {i} must have n_channels={n_channels} columns; got "
                f"{v.shape[1]}."
            )


def _validate_domain_coords(values, domain_coords):
    """Check the domain coordinates align with the panel; normalize to a list."""
    if domain_coords is None:
        return None
    if len(domain_coords) != len(values):
        raise ValueError(
            f"domain_coords ({len(domain_coords)}) must have the same length "
            f"as the panel ({len(values)})."
        )
    normalized = []
    for i, (p, v) in enumerate(zip(domain_coords, values)):
        p = np.asarray(p, dtype=float)
        if p.shape[0] != np.asarray(v).shape[0]:
            raise ValueError(
                f"domain_coords[{i}] has {p.shape[0]} rows but sample {i} has "
                f"{np.asarray(v).shape[0]} points."
            )
        normalized.append(p)
    return normalized


def _validate_meta(values, meta):
    """Check the per-specimen scalars align with the panel; normalize."""
    if meta is None:
        return None
    meta = np.asarray(meta, dtype=float)
    if meta.ndim != 2:
        raise ValueError(
            f"meta must be a 2D (n_samples, n_meta) array; got {meta.ndim}D."
        )
    if meta.shape[0] != len(values):
        raise ValueError(
            f"meta has {meta.shape[0]} rows but the panel has {len(values)} samples."
        )
    return meta


def _check_surface_panel(
    X, *, coord_names: tuple[str, str, str] = ("x", "y", "z")
) -> list[npt.NDArray[np.float64]]:
    """Normalize a panel of tube surfaces to a list of ``(n_s, n_phi, 3)`` arrays.

    A surface is the structured coordinate output of a coiling model's
    ``inverse_transform``: rows index the growth direction and columns index the
    aperture angle. Only the coordinate values are used; the grid parameter
    values (``s``/``theta`` and ``phi``) are not carried and are recovered by the
    estimator.

    Accepts these encodings and normalizes them to a common representation:

    - a ``list``/``tuple`` of ``(n_s, n_phi, 3)`` arrays (ragged ``n_s``/``n_phi``
      allowed);
    - a 4D ``ndarray`` of shape ``(n_samples, n_s, n_phi, 3)``;
    - a single 3D ``ndarray`` of shape ``(n_s, n_phi, 3)`` (a one-specimen panel);
    - a long 3-level ``MultiIndex`` ``DataFrame`` indexed by
      ``(specimen, trajectory, phi)`` with the coordinates as named columns
      (as produced by ``inverse_transform(as_frame=True)``).

    Parameters
    ----------
    X : list of array-like, ndarray, or DataFrame
        The surface panel in one of the accepted encodings.
    coord_names : tuple of str, default = ("x", "y", "z")
        Coordinate columns for the long-DataFrame encoding.

    Returns
    -------
    list of ndarray
        One ``(n_s, n_phi, 3)`` array per specimen.

    Raises
    ------
    ValueError
        For an unrecognized encoding, a wrong trailing dimension, a surface with
        fewer than three rows or columns, non-finite (NaN/inf) values, or a
        missing coordinate column.
    """
    if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.MultiIndex):
        surfaces = _surface_panel_from_long_dataframe(X, coord_names=coord_names)
    elif isinstance(X, np.ndarray):
        if X.ndim == 4:
            surfaces = [x.astype(float) for x in X]
        elif X.ndim == 3:
            surfaces = [X.astype(float)]
        else:
            raise ValueError(
                "A surface ndarray panel must be 4D (n_samples, n_s, n_phi, 3) "
                f"or 3D (n_s, n_phi, 3); got {X.ndim}D."
            )
    elif isinstance(X, (list, tuple)):
        surfaces = [np.asarray(x, dtype=float) for x in X]
    else:
        raise ValueError(
            "Unrecognized surface panel encoding: expected a list of "
            "(n_s, n_phi, 3) arrays, a 3D/4D ndarray, or a 3-level MultiIndex "
            f"DataFrame; got {type(X).__name__}."
        )

    for i, surf in enumerate(surfaces):
        if surf.ndim != 3 or surf.shape[2] != 3:
            raise ValueError(
                f"Surface {i} must be a 3D (n_s, n_phi, 3) array; got shape "
                f"{surf.shape}."
            )
        if surf.shape[0] < 3 or surf.shape[1] < 3:
            raise ValueError(
                f"Surface {i} must have at least 3 rows and 3 columns; got "
                f"shape {surf.shape}."
            )
        if not np.isfinite(surf).all():
            raise ValueError(
                f"Surface {i} contains non-finite values (NaN/inf). The surface "
                "estimator requires complete surfaces; partial apertures "
                "(missing points) are not supported yet."
            )
    return surfaces


def _surface_panel_from_long_dataframe(
    X: pd.DataFrame, *, coord_names: tuple[str, str, str]
) -> list[npt.NDArray[np.float64]]:
    """Rebuild ``(n_s, n_phi, 3)`` surfaces from a long ``(specimen, s, phi)`` frame."""
    if X.index.nlevels != 3:
        raise ValueError(
            "A long surface DataFrame must have a 3-level "
            "(specimen, trajectory, phi) index."
        )
    missing = [c for c in coord_names if c not in X.columns]
    if missing:
        raise ValueError(f"Coordinate columns absent from the DataFrame: {missing}.")
    surfaces = []
    for _, g in X.groupby(level=0, sort=False):
        g = g.sort_index()
        n_s = g.index.get_level_values(1).nunique()
        n_phi = g.index.get_level_values(2).nunique()
        arr = g.loc[:, list(coord_names)].to_numpy(dtype=float)
        surfaces.append(arr.reshape(n_s, n_phi, 3))
    return surfaces
