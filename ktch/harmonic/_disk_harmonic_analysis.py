"""Disk Harmonic Analysis"""

# Copyright 2025 Koji Noshita
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

import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
from scipy.special import jnp_zeros, jv
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
)
from sklearn.utils.parallel import Parallel, delayed


class DiskHarmonicAnalysis(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
):
    r"""Disk Harmonic Analysis

    Parameters
    ----------
    n_harmonics : int, default=10
        Maximum radial degree (:math:`n_\mathrm{max}`).
    n_dim : int, default=3
        Dimension of the coordinate space.
        Must be 2 (for planar mappings) or 3 (for surface mappings).
    n_jobs : int, default=None
        The number of jobs to run in parallel. None means 1 unless in a
        joblib.parallel_backend context. -1 means using all processors.
    verbose : int, default=0
        The verbosity level.

    Notes
    -----
    [Wolf_1979]_, [Boyd_etal_2011]_, [Shaqfa_etal_2025]_

    The surface is expanded as:

    .. math::
        \mathbf{p}(r, \theta) = \sum_{n=0}^{N} \sum_{m=-n}^{n}
            a_n^m\, \tilde{D}_n^m(r, \theta)

    where :math:`\tilde{D}_n^m` are real-valued disk harmonic basis
    functions constructed from Bessel functions of the first kind
    :math:`J_m` and their derivative zeros :math:`\lambda_{n,m}`:

    .. math::
        \tilde{D}_n^0(r, \theta) &= N_{n,0}\, J_0(\lambda_{n,0}\, r)
        \\
        \tilde{D}_n^m(r, \theta) &= \sqrt{2}\, N_{n,m}\, J_m(\lambda_{n,m}\, r)\, \cos(m\,\theta)
        \quad (m > 0)
        \\
        \tilde{D}_n^m(r, \theta) &= \sqrt{2}\, N_{n,|m|}\, J_{|m|}(\lambda_{n,|m|}\, r)\, \sin(|m|\,\theta)
        \quad (m < 0)

    with normalization constants of Fourier–Bessel basis functions:

    .. math::
        N_{n,m} = \frac{1}{\sqrt{\pi\,(1 - m^2/\lambda_{n,m}^2)\, J_m(\lambda_{n,m})^2}}

    References
    ----------
    .. [Wolf_1979] Wolf, K.B., 1979. Normal Mode Expansion and Bessel Series 221–251.
    .. [Boyd_etal_2011] Boyd, J.P., Yu, F., 2011. Comparing seven spectral methods for interpolation and for solving the Poisson equation in a disk: Zernike polynomials, Logan–Shepp ridge polynomials, Chebyshev–Fourier Series, cylindrical Robert functions, Bessel–Fourier expansions, square-to-disk conformal mapping and radial basis functions. J. Comput. Phys. 230, 1408–1438. 
    .. [Shaqfa_etal_2025] Shaqfa, M., Choi, G.P.T., Anciaux, G., Beyer, K., 2025. Disk harmonics for analysing curved and flat self-affine rough surfaces and the topological reconstruction of open surfaces. J. Comput. Phys. 522, 113578. 
    
    """

    def __init__(
        self,
        n_harmonics=10,
        n_dim=3,
        n_jobs=None,
        verbose=0,
    ):
        self.n_harmonics = n_harmonics
        self.n_dim = n_dim
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y=None):
        """Fit the model (no-op for stateless transformer).

        Parameters
        ----------
        X : ignored
        y : ignored

        Returns
        -------
        self
        """
        return self

    def __sklearn_is_fitted__(self):
        """Return True since this is a stateless transformer."""
        return True

    def fit_transform(self, X, y=None, r_theta=None):
        """Fit and transform in a single step.

        Overridden to support metadata routing of ``r_theta``.

        Parameters
        ----------
        X : list of array-like of shape (n_coords_i, n_dim)
            Coordinate values of n_samples.
        y : ignored
        r_theta : list of array-like of shape (n_coords_i, 2)
            Disk parameterization of n_samples.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
        """
        return self.fit(X, y).transform(X, r_theta=r_theta)

    def _transform_single(self, X, r_theta):
        """Compute DHA coefficients for a single sample.

        Parameters
        ----------
        X : array-like of shape (n_coords, n_dim)
            Vertex coordinates.
        r_theta : array-like of shape (n_coords, 2)
            Polar coordinates ``(r, theta)`` on the unit disk.

        Returns
        -------
        ndarray of shape (n_dim * (n_harmonics+1)**2,)
            Flat real-valued coefficient vector.
        """
        n_max = self.n_harmonics
        r = r_theta[:, 0]
        theta = r_theta[:, 1]

        n_coords = len(r)
        n_coeffs = (n_max + 1) ** 2
        if n_coords < n_coeffs:
            warnings.warn(
                f"Underdetermined system: n_coords ({n_coords}) < "
                f"(n_harmonics+1)**2 ({n_coeffs}). "
                f"lstsq will return a least-norm solution, not a "
                f"least-squares fit. "
                f"Consider reducing n_harmonics or providing more "
                f"sample points.",
                UserWarning,
                stacklevel=2,
            )

        B = _disk_harm_basis_matrix(n_max, r, theta)
        sol = sp.linalg.lstsq(B, X)

        return sol[0].T.ravel()

    def transform(self, X, r_theta=None):
        """Compute disk harmonic coefficients.

        Parameters
        ----------
        X : list of array-like
            Coordinate values of n_samples.
            The i-th element has shape ``(n_coords_i, n_dim)``
            representing vertex coordinates.
        r_theta : list of array-like of shape (n_coords_i, 2)
            Disk parameterization of n_samples.
            The i-th element holds ``(r, theta)`` polar coordinates.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            Disk harmonic coefficients.
        """
        if self.n_dim not in (2, 3):
            raise ValueError(f"n_dim must be 2 or 3, got {self.n_dim}")

        if r_theta is None:
            raise ValueError(
                "r_theta is required for DiskHarmonicAnalysis.transform(). "
                "Provide disk parameterization for each sample."
            )

        n_dim = self.n_dim
        if isinstance(X, pd.DataFrame):
            X_ = [
                row.dropna().to_numpy().reshape(n_dim, -1).T for _, row in X.iterrows()
            ]
        else:
            X_ = X

        if len(r_theta) != len(X_):
            raise ValueError(
                f"r_theta ({len(r_theta)}) must have the same length as X ({len(X_)})"
            )

        X_transformed = np.stack(
            Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(self._transform_single)(X_[i], r_theta[i])
                for i in range(len(X_))
            )
        )

        return X_transformed

    def get_feature_names_out(
        self, input_features: None | npt.ArrayLike = None
    ) -> np.ndarray:
        """Get output feature names.

        Parameters
        ----------
        input_features : ignored

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        return np.asarray(self._build_feature_names(), dtype=str)

    _AXIS_NAMES = {2: ("cx", "cy"), 3: ("cx", "cy", "cz")}

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        return self.n_dim * (self.n_harmonics + 1) ** 2

    def _build_feature_names(self) -> list[str]:
        n_max = self.n_harmonics
        axes = self._AXIS_NAMES[self.n_dim]
        names = []
        for axis in axes:
            for n in range(n_max + 1):
                for m in range(-n, n + 1):
                    names.append(f"{axis}_{n}_{m}")
        return names

    def _inverse_transform_single(
        self,
        X_transformed,
        r_range,
        theta_range,
        n_max,
    ):
        """Reconstruct a single surface from DHA coefficients.

        Parameters
        ----------
        X_transformed : ndarray
            Flat coefficient vector for one sample.
        r_range : array-like of shape (n_r,)
            Radial coordinates for the reconstruction grid.
        theta_range : array-like of shape (n_theta,)
            Angular coordinates for the reconstruction grid.
        n_max : int
            Maximum degree of harmonics to use.

        Returns
        -------
        ndarray of shape (n_theta, n_r, n_dim)
            Reconstructed surface coordinates.
        """
        n_dim = self.n_dim
        n_full = (self.n_harmonics + 1) ** 2
        n_coeffs = (n_max + 1) ** 2
        coef_matrix = X_transformed.reshape(n_dim, n_full)[:, :n_coeffs].T
        coef_list = _cvt_dha_coef_to_list(coef_matrix)
        coords = disk_harm(n_max, coef_list, r_range, theta_range)
        return np.stack(coords, axis=-1)

    def inverse_transform(
        self,
        X_transformed,
        r_range=None,
        theta_range=None,
        n_max=None,
    ):
        """Reconstruct surfaces from disk harmonic coefficients.

        Parameters
        ----------
        X_transformed : array-like of shape (n_samples, n_features)
            Disk harmonic coefficients.
        r_range : array-like of shape (n_r,), optional
            Radial coordinates.  Defaults to
            ``np.linspace(0, 1, 100)``.
        theta_range : array-like of shape (n_theta,), optional
            Angular coordinates.  Defaults to
            ``np.linspace(0, 2*pi, 180)``.
        n_max : int, optional
            Maximum degree of harmonics to use.  Defaults to
            ``self.n_harmonics``.

        Returns
        -------
        X_coords : ndarray of shape (n_samples, n_theta, n_r, n_dim)
            Reconstructed surface coordinates.
        """
        if r_range is None:
            r_range = np.linspace(0, 1, 100)
        if theta_range is None:
            theta_range = np.linspace(0, 2 * np.pi, 180)
        if n_max is None:
            n_max = self.n_harmonics

        X_coords = np.stack(
            Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(self._inverse_transform_single)(
                    X_transformed[i], r_range, theta_range, n_max
                )
                for i in range(len(X_transformed))
            )
        )

        return X_coords


###########################################################
#
#   Public utility functions
#
###########################################################


def xy2polar(
    xy: npt.NDArray[np.float64], *, centered: bool = True
) -> npt.NDArray[np.float64]:
    """Convert Cartesian coordinates to polar coordinates on the unit disk.

    Parameters
    ----------
    xy : ndarray of shape (N, 2)
        Cartesian coordinates.  If ``centered=True`` (default),
        assumed in ``[-1, 1] x [-1, 1]``.  If ``centered=False``,
        assumed in ``[0, 1] x [0, 1]``.
    centered : bool, default=True
        Whether the input is already centered at the origin.

    Returns
    -------
    ndarray of shape (N, 2)
        Polar coordinates ``[r, theta]`` where ``r`` is in ``[0, 1]``
        and ``theta`` is in ``[0, 2*pi)``.
    """
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must have shape (N, 2), got {xy.shape}")

    xy_c = xy if centered else 2.0 * (xy - 0.5)
    r = np.linalg.norm(xy_c, axis=1)
    theta = np.arctan2(xy_c[:, 1], xy_c[:, 0]) + np.pi

    return np.column_stack([r, theta])


def disk_harm(
    n_max: int,
    coef: list[npt.ArrayLike],
    r_range: npt.NDArray[np.float64] | None = None,
    theta_range: npt.NDArray[np.float64] | None = None,
) -> tuple[npt.NDArray[np.float64], ...]:
    """Reconstruct coordinates from disk harmonic coefficients.

    The number of output arrays is determined by the trailing dimension
    of the coefficient arrays (2 for planar, 3 for surface mappings).

    Parameters
    ----------
    n_max : int
        Maximum degree of disk harmonics to use.  Can be less than
        the degree used for estimation (truncated reconstruction).
    coef : list of array-like
        Disk harmonic coefficients.  ``coef[n]`` has shape
        ``(2*n+1, n_dim)`` and ``coef[n][n+m]`` holds the
        coefficients for degree ``n`` and order ``m``.
    r_range : array-like of shape (n_r,), optional
        Radial coordinates for the reconstruction grid.
        Defaults to ``np.linspace(0, 1, 100)``.
    theta_range : array-like of shape (n_theta,), optional
        Angular coordinates for the reconstruction grid.
        Defaults to ``np.linspace(0, 2*pi, 180)``.

    Returns
    -------
    tuple of ndarray of shape (n_theta, n_r)
        Reconstructed coordinates.  Length equals ``n_dim``
        (e.g., ``(x, y)`` for 2D or ``(x, y, z)`` for 3D).
    """
    if r_range is None:
        r_range = np.linspace(0, 1, 100)
    if theta_range is None:
        theta_range = np.linspace(0, 2 * np.pi, 180)

    r_grid, theta_grid = np.meshgrid(r_range, theta_range)
    B = _disk_harm_basis_matrix(n_max, r_grid.ravel(), theta_grid.ravel())

    coef_matrix = np.vstack(coef[: n_max + 1])
    coords = B @ coef_matrix

    n_theta, n_r = len(theta_range), len(r_range)
    return tuple(
        coords[:, d].reshape(n_theta, n_r) for d in range(coef_matrix.shape[1])
    )


###########################################################
#
#   Private helpers
#
###########################################################


def _calc_eigenvalues(n_max: int) -> np.ndarray:
    """Compute eigenvalue table for disk harmonics.

    The eigenvalues are the zeros of the derivative of the Bessel
    function of the first kind (Neumann boundary condition on the
    unit disk).

    Parameters
    ----------
    n_max : int
        Maximum radial degree.

    Returns
    -------
    ndarray of shape (n_max+1, n_max+1)
        Lower-triangular eigenvalue table where entry ``[n, m]``
        contains ``lambda_{n,m}``.  Entries with ``m > n`` are zero.
    """
    if n_max == 0:
        return np.array([[0.0]])

    # m=0 column: prepend 0 for the n=0 constant mode
    col_0 = np.concatenate([np.zeros(1), jnp_zeros(0, n_max)]).reshape(-1, 1)

    # m=1..n_max columns: zero-pad the first m entries (invalid pairs)
    cols = np.array(
        [
            np.concatenate([np.zeros(m), jnp_zeros(m, n_max + 1)])[:-m]
            for m in range(1, n_max + 1)
        ]
    ).T

    return np.hstack([col_0, cols])


def _disk_harm_basis_matrix(
    n_max: int,
    r: npt.NDArray[np.float64],
    theta: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""Build real-valued disk harmonic design matrix.

    Columns correspond to ``(n, m)`` pairs in the ordering
    ``(0,0), (1,-1), (1,0), (1,1), (2,-2), ...``.

    The real-valued basis functions are:

    * ``n=0, m=0``: :math:`1/\sqrt{\pi}`
    * ``n>0, m=0``: :math:`N_{n,0}\, J_0(\lambda_{n,0}\, r)`
    * ``m>0``: :math:`\sqrt{2}\, N_{n,m}\, J_m(\lambda_{n,m}\, r)\, \cos(m\,\theta)`
    * ``m<0``: :math:`\sqrt{2}\, N_{n,|m|}\, J_{|m|}(\lambda_{n,|m|}\, r)\, \sin(|m|\,\theta)`

    Parameters
    ----------
    n_max : int
        Maximum degree.
    r : ndarray of shape (N,)
        Radial coordinates in [0, 1].
    theta : ndarray of shape (N,)
        Angular coordinates.

    Returns
    -------
    ndarray of shape (N, (n_max+1)**2)
        Real-valued design matrix.
    """
    l_nm_table = _calc_eigenvalues(n_max)

    # Build flat arrays of (n, m) indices for all basis functions
    n_arr = np.array([n for n in range(n_max + 1) for m in range(-n, n + 1)])
    m_arr = np.array([m for n in range(n_max + 1) for m in range(-n, n + 1)])
    m_abs_arr = np.abs(m_arr)
    n_coeffs = len(n_arr)

    # Eigenvalues and normalization for each (n, m) pair — shape (K,)
    lam = l_nm_table[n_arr, m_abs_arr]
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(
            n_arr == 0,
            1.0 / np.sqrt(np.pi),
            1.0
            / np.sqrt(np.pi * (1 - m_abs_arr**2 / lam**2) * jv(m_abs_arr, lam) ** 2),
        )

    # Radial part: J_{|m|}(lambda * r_i) — shape (K, N)
    radial = np.where(
        (n_arr == 0)[:, None],
        1.0,
        jv(m_abs_arr[:, None], lam[:, None] * r[None, :]),
    )

    # Angular part — shape (K, N)
    angular = np.where(
        (m_arr == 0)[:, None],
        1.0,
        np.where(
            (m_arr > 0)[:, None],
            np.sqrt(2) * np.cos(m_arr[:, None] * theta[None, :]),
            np.sqrt(2) * np.sin(m_abs_arr[:, None] * theta[None, :]),
        ),
    )

    # Combine: (K, N) -> transpose to (N, K)
    return (norm[:, None] * radial * angular).T


def _cvt_dha_coef_to_list(
    coef: npt.NDArray[np.float64],
) -> list[npt.NDArray[np.float64]]:
    """Convert flat DHA coefficient array to a nested list by degree.

    Parameters
    ----------
    coef : ndarray of shape ((n_max+1)**2,) or ((n_max+1)**2, D)
        Flat coefficient array.

    Returns
    -------
    list of ndarray
        ``coef_list[n]`` has shape ``(2*n+1,)`` or ``(2*n+1, D)``.

    Raises
    ------
    ValueError
        If the number of coefficients is not a perfect square.
    """
    n_coef = coef.shape[0]
    n_max_plus_one = np.sqrt(n_coef)
    if not n_max_plus_one.is_integer():
        raise ValueError(
            f"Invalid coefficient count: {n_coef} is not a perfect square "
            f"((n_max+1)^2)."
        )
    n_max = int(n_max_plus_one) - 1

    coef_list = []
    for n in range(n_max + 1):
        start = n**2
        end = (n + 1) ** 2
        coef_list.append(coef[start:end])

    return coef_list


###########################################################
#
#   Complex <-> Real coefficient conversion
#
###########################################################


def _complex_to_real_coef(
    coef_complex: npt.NDArray[np.complexfloating],
) -> npt.NDArray[np.float64]:
    r"""Convert complex disk harmonic coefficients to real coefficients.

    Given complex coefficients from a complex-basis expansion,
    return the equivalent real-basis coefficients.

    The mapping for degree ``n``, order ``m`` is:

    * ``m = 0``:  ``a_{n,0} = Re(c_{n,0})``
    * ``m > 0``:  ``a_{n,m} = \sqrt{2}\, Re(c_{n,m})``
    * ``m < 0``:  ``a_{n,m} = -\sqrt{2}\, Im(c_{n,|m|})``

    Parameters
    ----------
    coef_complex : ndarray of shape ((n_max+1)**2,) or ((n_max+1)**2, D)
        Complex coefficients in flat ordering.

    Returns
    -------
    ndarray of same shape, float64
        Real-valued coefficients.
    """
    coef_real = np.empty_like(coef_complex, dtype=np.float64)
    n_coef = coef_complex.shape[0]
    n_max = int(np.sqrt(n_coef)) - 1

    for n in range(n_max + 1):
        for m in range(-n, n + 1):
            idx = n**2 + n + m
            if m == 0:
                coef_real[idx] = np.real(coef_complex[idx])
            elif m > 0:
                coef_real[idx] = np.sqrt(2) * np.real(coef_complex[idx])
            else:
                idx_pos = n**2 + n + (-m)
                coef_real[idx] = -np.sqrt(2) * np.imag(coef_complex[idx_pos])

    return coef_real


def _real_to_complex_coef(
    coef_real: npt.NDArray[np.float64],
) -> npt.NDArray[np.complexfloating]:
    r"""Convert real disk harmonic coefficients to complex coefficients.

    Inverse of :func:`_complex_to_real_coef`.  The output satisfies
    conjugate symmetry: ``c_{n,-m} = (-1)^m \overline{c_{n,m}}``.

    Parameters
    ----------
    coef_real : ndarray of shape ((n_max+1)**2,) or ((n_max+1)**2, D)
        Real-valued coefficients in flat ordering.

    Returns
    -------
    ndarray of same shape, complex128
        Complex coefficients.
    """
    coef_complex = np.empty_like(coef_real, dtype=np.complex128)
    n_coef = coef_real.shape[0]
    n_max = int(np.sqrt(n_coef)) - 1

    for n in range(n_max + 1):
        # m = 0
        idx_0 = n**2 + n
        coef_complex[idx_0] = coef_real[idx_0] + 0j

        # m > 0 and corresponding m < 0
        for m in range(1, n + 1):
            idx_pos = n**2 + n + m
            idx_neg = n**2 + n - m
            c_pos = (coef_real[idx_pos] - 1j * coef_real[idx_neg]) / np.sqrt(2)
            coef_complex[idx_pos] = c_pos
            coef_complex[idx_neg] = ((-1) ** m) * np.conj(c_pos)

    return coef_complex
