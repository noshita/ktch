"""Kernel functions for landmark morphometrics"""

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
import scipy as sp


def tps_kernel(r, n_dim=2):
    """Compute TPS kernel value(s).

    Parameters
    ----------
    r : float or ndarray
        Distance(s). Must be non-negative.
    n_dim : int, default=2
        Spatial dimension (2 or 3).

    Returns
    -------
    float or ndarray
        Kernel value(s).
        - 2D: U(r) = r^2 * log(|r|), with U(0) = 0
        - 3D: U(r) = -|r|
    """
    r = np.asarray(r)
    if n_dim == 2:
        result = np.zeros_like(r, dtype=float)
        mask = r > 0
        result[mask] = r[mask] ** 2 * np.log(r[mask])
        return result
    elif n_dim == 3:
        return -r
    else:
        raise ValueError(f"n_dim must be 2 or 3, got {n_dim}")


def tps_kernel_matrix(X):
    """Compute TPS kernel matrix K.

    Parameters
    ----------
    X : ndarray, shape (n_points, n_dim)
        Point coordinates.

    Returns
    -------
    K : ndarray, shape (n_points, n_points)
        Kernel matrix where K[i,j] = U(||X[i] - X[j]||).
    """
    X = np.asarray(X)
    n_dim = X.shape[1]

    r = sp.spatial.distance.cdist(X, X, "euclidean")

    K = tps_kernel(r, n_dim=n_dim)

    return K


def tps_system_matrix(X):
    """Compute full TPS system matrix L.

    The system matrix has the form:
        L = [K  Q]
            [Q' 0]

    where K is the kernel matrix and Q = [1, X] is the affine basis.

    Parameters
    ----------
    X : ndarray, shape (n_points, n_dim)
        Point coordinates.

    Returns
    -------
    L : ndarray, shape (n_points + n_dim + 1, n_points + n_dim + 1)
        TPS system matrix.
    """
    X = np.asarray(X)
    n = len(X)
    n_dim = X.shape[1]

    K = tps_kernel_matrix(X)

    # Affine basis: [1, x1, x2, ...] for each point
    Q = np.hstack([np.ones((n, 1)), X])  # n x (n_dim + 1)

    # TPS system matrix L
    L = np.zeros((n + n_dim + 1, n + n_dim + 1))
    L[:n, :n] = K
    L[:n, n:] = Q
    L[n:, :n] = Q.T

    return L


def tps_coefficients(source, target):
    """Compute TPS coefficients for warping source to target.

    Solves the TPS interpolation problem to find coefficients that
    warp the source configuration to the target configuration.

    Parameters
    ----------
    source : ndarray, shape (n_points, n_dim)
        Source (reference) configuration.
    target : ndarray, shape (n_points, n_dim)
        Target configuration.

    Returns
    -------
    W : ndarray, shape (n_points, n_dim)
        Non-affine (warping) coefficients.
    c : ndarray, shape (n_dim,)
        Translation coefficients.
    A : ndarray, shape (n_dim, n_dim)
        Affine transformation matrix.

    Notes
    -----
    The TPS transformation is:
        f(x) = c + A @ x + sum_i(W[i] * U(||x - source[i]||))
    """
    source = np.asarray(source)
    target = np.asarray(target)
    n_dim = source.shape[1]
    n_landmarks = len(source)

    if source.shape != target.shape:
        raise ValueError("source and target must be arrays of the same shape.")

    L = tps_system_matrix(source)
    rhs = np.vstack([target, np.zeros((n_dim + 1, n_dim))])

    try:
        sol = np.linalg.solve(L, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(L, rhs, rcond=None)[0]

    W = sol[:n_landmarks]
    c = sol[n_landmarks]
    A = sol[n_landmarks + 1 :]

    return W, c, A


def tps_warp(points, source, W, c, A):
    """Warp points using TPS transformation.

    Parameters
    ----------
    points : ndarray, shape (n_points, n_dim)
        Points to warp.
    source : ndarray, shape (n_landmarks, n_dim)
        Source landmarks (control points).
    W : ndarray, shape (n_landmarks, n_dim)
        Non-affine coefficients from tps_coefficients().
    c : ndarray, shape (n_dim,)
        Translation from tps_coefficients().
    A : ndarray, shape (n_dim, n_dim)
        Affine matrix from tps_coefficients().

    Returns
    -------
    warped : ndarray, shape (n_points, n_dim)
        Warped points.
    """
    points = np.asarray(points)
    source = np.asarray(source)
    n_dim = points.shape[1]

    # Compute distances from each point to each source landmark
    r = sp.spatial.distance.cdist(points, source, "euclidean")

    # Apply kernel
    U = tps_kernel(r, n_dim=n_dim)

    # Warp: f(x) = c + A @ x + U @ W
    warped = c + points @ A.T + U @ W

    return warped


def tps_bending_energy_matrix(X):
    """Compute the TPS bending energy matrix.

    The bending energy matrix is the upper-left k x k submatrix of L^{-1},
    where L is the full TPS system matrix.
    This matrix is used in the closed-form semi-landmark sliding solution
    (Bookstein 1997, Eq. 8).

    Parameters
    ----------
    X : ndarray, shape (k, n_dim)
        Reference landmark configuration.

    Returns
    -------
    Lk_inv : ndarray, shape (k, k)
        Bending energy matrix (upper-left block of L^{-1}).
    """
    X = np.asarray(X)
    k = len(X)
    L = tps_system_matrix(X)
    L_inv = np.linalg.pinv(L)
    Lk_inv = L_inv[:k, :k]
    return Lk_inv


def tps_bending_energy(source, target):
    """Compute TPS bending energy between configurations.

    The bending energy is defined as:
        B_e = trace(W^T @ K @ W)

    where W are the non-affine TPS coefficients and K is the kernel matrix.

    Parameters
    ----------
    source : ndarray, shape (n_points, n_dim)
        Source configuration.
    target : ndarray, shape (n_points, n_dim)
        Target configuration.

    Returns
    -------
    energy : float
        Bending energy. Always non-negative.
        Zero indicates a pure affine transformation.
    """
    source = np.asarray(source)
    target = np.asarray(target)

    W, c, A = tps_coefficients(source, target)

    K = tps_kernel_matrix(source)
    bending_energy = np.trace(W.T @ K @ W)

    return bending_energy
