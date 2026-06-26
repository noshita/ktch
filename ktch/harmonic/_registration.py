"""Shared registration utilities for harmonic methods.

Registration removes nuisance similarity transforms of the codomain (group A:
translation, rotation, scale) and the parameter-domain symmetry (group B).
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

import numpy as np
import numpy.typing as npt

# Tolerance for detecting a near-zero shape size.
_SIZE_TOL = 1e-12

# Tolerance below which a principal-axis skewness is treated as zero (the
# shape is symmetric along that axis, so its sign is intrinsically ambiguous).
_SKEW_TOL = 1e-9

# Registration methods reserved (contract only) but not implemented yet.
_RESERVED_REGISTRATIONS = {"landmark", "rotational_match"}
_VALID_REGISTRATIONS = {None, "first_order", "moment"} | _RESERVED_REGISTRATIONS


def validate_registration(
    method,
    scale_method,
    scale_methods_by_registration,
    *,
    n_dim,
    return_transform,
    allow_first_order,
    align_parameter=True,
):
    """Validate registration settings for SPHARM/DHA-style estimators.

    Registration applies to 2D/3D shape data only; for non-shape codomains
    (``n_dim`` not in ``(2, 3)``) it must be ``None``.

    Parameters
    ----------
    method : {None, "first_order", "moment", ...}
        Requested registration method.
    scale_method : str or None
        Requested size measure.
    scale_methods_by_registration : dict
        Maps each implemented method name to its set of valid ``scale_method``
        values.
    n_dim : int
        Codomain dimension.
    return_transform : bool
        Whether the estimated transform is requested (not yet implemented).
    allow_first_order : bool
        Whether ``"first_order"`` is implemented for this estimator.
    align_parameter : bool, default=True
        Whether the parameter-domain group (B) is aligned. ``first_order``
        always aligns it; ``False`` is not yet implemented and raises.

    Raises
    ------
    ValueError
        For unknown methods, registration on non-2D/3D data, incompatible
        ``scale_method``, or ``return_transform`` with ``method=None``.
    NotImplementedError
        For reserved methods, unimplemented ``first_order``,
        ``return_transform``, or ``align_parameter=False``.
    """
    if method not in _VALID_REGISTRATIONS:
        raise ValueError(
            f"registration must be one of "
            f"{sorted(str(m) for m in _VALID_REGISTRATIONS)}, got '{method}'"
        )
    if method in _RESERVED_REGISTRATIONS:
        raise NotImplementedError(
            f"registration='{method}' is reserved and not implemented yet."
        )
    if method is None:
        if return_transform:
            raise ValueError("return_transform requires registration != None.")
        return
    if method == "first_order" and not allow_first_order:
        raise NotImplementedError(
            "registration='first_order' is not yet implemented for this "
            "estimator; use registration='moment' or registration=None."
        )
    if method == "first_order" and not align_parameter:
        raise NotImplementedError(
            "align_parameter=False is not yet implemented; 'first_order' "
            "always aligns the parameter domain (group B). Use "
            "align_parameter=True."
        )
    if n_dim not in (2, 3):
        raise ValueError(
            f"registration='{method}' applies to 2D/3D shape data only; got "
            f"n_dim={n_dim}. Use registration=None for non-shape data "
            "(normalization of n-D fields belongs to a separate interface)."
        )
    allowed = scale_methods_by_registration[method]
    if scale_method not in allowed:
        raise ValueError(
            f"scale_method='{scale_method}' is not valid for "
            f"registration='{method}'; valid options: "
            f"{sorted(str(m) for m in allowed)}."
        )
    if return_transform:
        raise NotImplementedError(
            "return_transform is not yet implemented for this estimator."
        )


def moment_frame(
    vectors: npt.NDArray[np.float64],
    reflect: bool = False,
) -> tuple[npt.NDArray[np.float64], float, npt.NDArray[np.float64]]:
    """Principal-axis frame from the second moment of coefficient vectors.

    For an orthonormal basis, ``M = sum_k a_k a_k^T`` equals the shape second
    moment (covariance up to the measure). Its eigenvectors are the principal
    axes (group A, codomain rotation only).

    Parameters
    ----------
    vectors : ndarray of shape (n_modes, n_dim)
        Coefficient vectors of all non-constant modes (one row per mode).
    reflect : bool, default=False
        If ``False``, force a proper rotation (``det = +1``) by flipping the
        least-significant axis when needed.

    Returns
    -------
    Q : ndarray of shape (n_dim, n_dim)
        Rotation whose columns are the principal axes ordered by descending
        eigenvalue. The per-axis sign is fixed by the skewness of the mode
        projections (rotation-invariant; positive skew), falling back to the
        largest-magnitude component for (near-)symmetric axes. Express
        coordinates in the principal frame with ``Q.T @ v``.
    size : float
        ``sqrt(trace(M))``, the RMS (centroid) size.
    eigenvalues : ndarray of shape (n_dim,)
        Variances along the principal axes (descending).
    """
    vectors = np.asarray(vectors, dtype=float)
    n_dim = vectors.shape[1]

    M = vectors.T @ vectors
    w, V = np.linalg.eigh(M)  # ascending eigenvalues, orthonormal columns

    order = np.argsort(w)[::-1]
    w = w[order]
    Q = V[:, order]

    # Sign convention: positive skewness of the projections along each axis
    # (rotation-invariant). For symmetric axes (near-zero skew) fall back to a
    # deterministic largest-magnitude rule.
    proj = vectors @ Q  # (n_modes, n_dim)
    skew = np.sum(proj**3, axis=0)
    for j in range(n_dim):
        if abs(skew[j]) > _SKEW_TOL:
            if skew[j] < 0:
                Q[:, j] = -Q[:, j]
        else:
            col = Q[:, j]
            k = int(np.argmax(np.abs(col)))
            if col[k] < 0:
                Q[:, j] = -col

    # Proper rotation unless reflection is explicitly allowed.
    if not reflect and n_dim > 0 and np.linalg.det(Q) < 0:
        Q[:, -1] = -Q[:, -1]

    size = float(np.sqrt(np.sum(np.clip(w, 0.0, None))))
    return Q, size, w


def moment_register(
    coef_flat: npt.NDArray[np.float64],
    n_dim: int,
    *,
    scale: bool = True,
    reflect: bool = False,
) -> npt.NDArray[np.float64]:
    """Apply moment-based codomain registration to a flat coefficient vector.

    Removes translation (constant mode -> 0), rotates the codomain to the
    principal-axis frame, and optionally divides by the centroid size. Does
    not touch the parameter domain (group B is not resolved by ``moment``).

    Parameters
    ----------
    coef_flat : ndarray of shape (n_dim * n_modes,)
        Axis-major flat coefficients; constant mode at column 0 of the
        ``reshape(n_dim, n_modes)`` view.
    n_dim : int
        Codomain dimension.
    scale : bool, default=True
        Divide by ``centroid_size`` (shape space) when ``True``; keep size
        (form space) when ``False``.
    reflect : bool, default=False
        Allow improper rotations (remove chirality) when ``True``.

    Returns
    -------
    ndarray of shape (n_dim * n_modes,)
        Registered flat coefficients in the same layout as the input.

    Raises
    ------
    ValueError
        If ``scale`` is requested but the shape size is near zero.
    """
    coef_flat = np.asarray(coef_flat, dtype=float)
    n_modes = coef_flat.shape[0] // n_dim
    mat = coef_flat.reshape(n_dim, n_modes).copy()

    vectors = mat[:, 1:].T  # exclude the constant (translation) mode
    Q, size, _ = moment_frame(vectors, reflect=reflect)

    mat[:, 0] = 0.0  # remove translation
    rotated = Q.T @ mat

    if scale:
        if size < _SIZE_TOL:
            raise ValueError(
                "Degenerate shape: near-zero centroid size; cannot scale. "
                "Use scale=False."
            )
        rotated = rotated / size

    return rotated.ravel()
