"""Spherical Harmonic (SPHARM) Analysis"""

# Copyright 2020 Koji Noshita
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
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
)
from sklearn.utils.parallel import Parallel, delayed

# Tolerance for detecting pole singularity in xyz2spherical.
_POLE_TOL = 1e-12


class SphericalHarmonicAnalysis(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
):
    r"""Spherical Harmonic (SPHARM) Analysis

    Parameters
    ----------
    n_harmonics: int, default=10
        Number of harmonics to use ($l_\mathrm{max}$).
    n_jobs: int, default=None
        The number of jobs to run in parallel. None means 1 unless in a
        joblib.parallel_backend context. -1 means using all processors.
    verbose: int, default=0
        The verbosity level.


    Notes
    -----
    [Ritche_Kemp_1999]_, [Shen_etal_2009]_

    A surface point :math:`\mathbf{p}(\theta, \phi)` is expanded as

    .. math::
        \mathbf{p}(\theta, \phi) = \sum_{l=0}^{L} \sum_{m=-l}^l
            a_l^m \, S_l^m(\theta, \phi)

    where :math:`S_l^m` are real orthonormal spherical harmonics defined
    in terms of the complex harmonics :math:`Y_l^m`:

    * :math:`S_l^0 = Y_l^0`
    * :math:`S_l^m = \sqrt{2}\,(-1)^m\,\mathrm{Re}(Y_l^m)` for :math:`m > 0`
    * :math:`S_l^m = \sqrt{2}\,(-1)^{|m|}\,\mathrm{Im}(Y_l^{|m|})` for :math:`m < 0`

    The coefficients :math:`a_l^m` are real-valued, so ``transform``
    returns a ``float64`` array.  Conversion utilities
    ``_complex_to_real_sph_coef`` and ``_real_to_complex_sph_coef`` are
    available for interoperability with complex-basis representations.

    References
    ----------

    .. [Ritche_Kemp_1999] Ritchie, D.W., Kemp, G.J.L. (1999) Fast computation, rotation, and comparison of low resolution spherical harmonic molecular surfaces. J. Comput. Chem. 20: 383–395.
    .. [Shen_etal_2009] Shen, L., Farid, H., McPeek, M.A. (2009) Modeling three-dimensional morphological structures using spherical harmonics. Evolution (N. Y). 63: 1003–1016.



    """

    def __init__(
        self,
        n_harmonics=10,
        n_jobs=None,
        verbose=0,
    ):
        self.n_harmonics = n_harmonics
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

    def fit_transform(self, X, y=None, theta_phi=None):
        """Fit and transform in a single step.

        Overridden to support metadata routing of ``theta_phi``.

        Parameters
        ----------
        X : list of array-like of shape (n_coords_i, 3)
            Coordinate values of n_samples.
        y : ignored
        theta_phi : list of array-like of shape (n_coords_i, 2)
            Surface parameterization of n_samples.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_coefficients)
        """
        return self.fit(X, y).transform(X, theta_phi=theta_phi)

    def _transform_single(self, X, theta_phi):
        """Compute real SPHARM coefficients for a single sample.

        Parameters
        ----------
        X : array-like of shape (n_coords, 3)
            Coordinate values of a surface.
        theta_phi : array-like of shape (n_coords, 2)
            Parameters indicating the position on the surface.

        Returns
        -------
        X_transformed : ndarray of shape (3 * (l_max + 1)**2,), float64
            Flat real-valued SPHARM coefficient vector.
            Layout: ``[cx_0_0, ..., cy_0_0, ..., cz_0_0, ...]``.
        """
        l_max = self.n_harmonics
        theta = theta_phi[:, 0]
        phi = theta_phi[:, 1]

        n_coords = len(theta)
        n_coeffs = (l_max + 1) ** 2
        if n_coords < n_coeffs:
            warnings.warn(
                f"Underdetermined system: n_coords ({n_coords}) < "
                f"(n_harmonics+1)**2 ({n_coeffs}). "
                f"lstsq will return a least-norm solution, not a least-squares fit. "
                f"Consider reducing n_harmonics or providing more sample points.",
                UserWarning,
                stacklevel=2,
            )

        B = _real_sph_harm_basis_matrix(l_max, theta, phi)

        sol = sp.linalg.lstsq(B, X)
        X_transformed = sol[0].T.ravel()

        return X_transformed

    def transform(self, X, theta_phi=None):
        """Compute SPHARM coefficients.

        Parameters
        ----------
        X: list of array-like
            Coordinate values of n_samples.
            The i-th array-like whose shape (n_coords_i, 3) represents
            3D coordinate values of the i-th sample .

        theta_phi: list of array-like of shape (n_coords, 2)
            Surface parameter of n_samples.
            The i-th array-like of theta and phi values whose shape is (n_coords_i, 2).

        Returns
        -------
        X_transformed: array-like of shape (n_samples, n_coefficients)
            Returns the array-like of SPHARM coefficients.
        """
        if theta_phi is None:
            raise ValueError(
                "theta_phi is required for SphericalHarmonicAnalysis.transform(). "
                "Provide surface parameterization for each sample."
            )

        if isinstance(X, pd.DataFrame):
            X_ = [row.dropna().to_numpy().reshape(3, -1).T for idx, row in X.iterrows()]
        else:
            X_ = X

        if len(theta_phi) != len(X_):
            raise ValueError(
                f"theta_phi ({len(theta_phi)}) must have the same length "
                f"as X ({len(X_)})"
            )

        X_transformed = np.stack(
            Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(self._transform_single)(X_[i], theta_phi[i])
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

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        return 3 * (self.n_harmonics + 1) ** 2

    def _build_feature_names(self) -> list[str]:
        l_max = self.n_harmonics
        names = []
        for axis in ("cx", "cy", "cz"):
            for l in range(l_max + 1):
                for m in range(-l, l + 1):
                    names.append(f"{axis}_{l}_{m}")
        return names

    def _inverse_transform_single(
        self,
        X_transformed,
        theta_range,
        phi_range,
        l_max=None,
    ):
        """Reconstruct a single surface from SPHARM coefficients.

        Parameters
        ----------
        X_transformed : ndarray of shape (3 * (n_harmonics + 1)**2,)
            Flat SPHARM coefficient vector for one sample, in axis-major
            layout (cx block, cy block, cz block).
        theta_range : array-like of shape (n_theta,)
            Polar angle values (colatitude, 0 to pi).
        phi_range : array-like of shape (n_phi,)
            Azimuthal angle values (0 to 2*pi).
        l_max : int, optional
            Maximum degree of harmonics to use. Defaults to
            ``self.n_harmonics``. When less than ``self.n_harmonics``,
            the leading ``(l_max + 1) ** 2`` coefficients of each axis
            block are kept and higher-degree terms are dropped.

        Returns
        -------
        X_coords : ndarray of shape (n_theta, n_phi, 3)
            Reconstructed surface coordinates.
        """
        if l_max is None:
            l_max = self.n_harmonics

        n_per_lm_full = (self.n_harmonics + 1) ** 2
        n_per_lm = (l_max + 1) ** 2

        # Axis-major layout: (3, n_per_lm_full) → take leading n_per_lm cols.
        coef_per_lm = (
            np.asarray(X_transformed).reshape(3, n_per_lm_full)[:, :n_per_lm].T
        )
        x, y, z = spharm(
            l_max,
            cvt_spharm_coef_to_list(coef_per_lm),
            theta_range,
            phi_range,
        )
        X_coords = np.stack([x, y, z], axis=-1)
        return X_coords

    def inverse_transform(
        self,
        X_transformed,
        theta_range=None,
        phi_range=None,
        l_max=None,
    ):
        """Reconstruct surfaces from SPHARM coefficients.

        Parameters
        ----------
        X_transformed : array-like of shape (n_samples, 3 * (l_max + 1)**2)
            Flat SPHARM coefficient vectors as returned by
            :meth:`transform`.  Layout is axis-major:
            ``[cx_0_0, cx_1_-1, ..., cy_0_0, ..., cz_0_0, ...]``.
        theta_range : array-like of shape (n_theta,), optional
            Polar angle values (colatitude). Defaults to
            ``np.linspace(0, pi, 90)``.
        phi_range : array-like of shape (n_phi,), optional
            Azimuthal angle values. Defaults to
            ``np.linspace(0, 2*pi, 180)``.
        l_max : int, optional
            Maximum degree of harmonics to use. Defaults to
            ``self.n_harmonics``. When smaller, the input coefficient
            vector is truncated to the leading ``(l_max + 1) ** 2``
            terms per axis. Values greater than ``self.n_harmonics``
            raise ``ValueError``.

        Returns
        -------
        X_coords : ndarray of shape (n_samples, n_theta, n_phi, 3)
            Reconstructed surface coordinates.

        Raises
        ------
        ValueError
            If ``l_max`` is negative or greater than ``self.n_harmonics``.
        """
        if theta_range is None:
            theta_range = np.linspace(0, np.pi, 90)
        if phi_range is None:
            phi_range = np.linspace(0, 2 * np.pi, 180)
        if l_max is None:
            l_max = self.n_harmonics
        if l_max < 0:
            raise ValueError(f"l_max must be >= 0, got {l_max}")
        if l_max > self.n_harmonics:
            raise ValueError(
                f"l_max ({l_max}) cannot exceed n_harmonics "
                f"({self.n_harmonics})"
            )

        X_coords = np.stack(
            Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(self._inverse_transform_single)(
                    X_transformed[i], theta_range, phi_range, l_max
                )
                for i in range(len(X_transformed))
            )
        )

        return X_coords


###########################################################
#
#   utility functions
#
###########################################################


def _real_sph_harm_y(
    l: int,
    m: int,
    theta: npt.NDArray[np.float64],
    phi: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""Evaluate a real spherical harmonic :math:`S_l^m`.

    Defined via :func:`scipy.special.sph_harm_y`:

    * :math:`m = 0`:  :math:`S_l^0 = Y_l^0`
    * :math:`m > 0`:  :math:`S_l^m = \sqrt{2}\,(-1)^m\,\mathrm{Re}(Y_l^m)`
    * :math:`m < 0`:  :math:`S_l^m = \sqrt{2}\,(-1)^{|m|}\,\mathrm{Im}(Y_l^{|m|})`

    This evaluates to:

    * :math:`S_l^m = \sqrt{2}\,N_l^m\,P_l^m(\cos\theta)\,\cos(m\varphi)`
      for :math:`m > 0`
    * :math:`S_l^{-|m|} = \sqrt{2}\,N_l^{|m|}\,P_l^{|m|}(\cos\theta)\,
      \sin(|m|\varphi)` for :math:`m < 0`

    The :math:`(-1)^m` factor cancels the Condon-Shortley phase
    included in ``sph_harm_y``, yielding positive cosine/sine.
    """
    if m == 0:
        return sp.special.sph_harm_y(l, 0, theta, phi).real
    elif m > 0:
        return np.sqrt(2) * (-1) ** m * np.real(sp.special.sph_harm_y(l, m, theta, phi))
    else:
        return np.sqrt(2) * (-1) ** abs(m) * np.imag(
            sp.special.sph_harm_y(l, abs(m), theta, phi)
        )


def _real_sph_harm_basis_matrix(
    l_max: int,
    theta: npt.NDArray[np.float64],
    phi: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""Build real-valued spherical harmonic design matrix.

    Columns correspond to ``(l, m)`` pairs in the ordering
    ``(0,0), (1,-1), (1,0), (1,1), (2,-2), ...``.

    Parameters
    ----------
    l_max : int
        Maximum degree.
    theta : ndarray of shape (N,)
        Polar angle (colatitude) values.
    phi : ndarray of shape (N,)
        Azimuthal angle values.

    Returns
    -------
    ndarray of shape (N, (l_max+1)**2), float64
        Real-valued design matrix.
    """
    n_pts = len(theta)
    n_coeffs = (l_max + 1) ** 2
    B = np.empty((n_pts, n_coeffs))

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            idx = l**2 + l + m
            B[:, idx] = _real_sph_harm_y(l, m, theta, phi)

    return B


def _complex_to_real_sph_coef(
    coef_complex: npt.NDArray[np.complexfloating],
) -> npt.NDArray[np.float64]:
    r"""Convert complex SH coefficients to real SH coefficients.

    Includes the :math:`(-1)^m` Condon-Shortley phase factor,
    which makes this different from DHA's ``_complex_to_real_coef``.

    The mapping for degree ``l``, order ``m`` is:

    * ``m = 0``:  ``a_{l,0} = Re(c_{l,0})``
    * ``m > 0``:  ``a_{l,m} = \sqrt{2}\,(-1)^m\,Re(c_{l,m})``
    * ``m < 0``:  ``a_{l,m} = -\sqrt{2}\,(-1)^{|m|}\,Im(c_{l,|m|})``

    Parameters
    ----------
    coef_complex : ndarray of shape ((l_max+1)**2,) or ((l_max+1)**2, D)
        Complex coefficients in flat ordering.

    Returns
    -------
    ndarray of same shape, float64
        Real-valued coefficients.
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


def _real_to_complex_sph_coef(
    coef_real: npt.NDArray[np.float64],
) -> npt.NDArray[np.complexfloating]:
    r"""Convert real SH coefficients to complex SH coefficients.

    Inverse of :func:`_complex_to_real_sph_coef`.  The output satisfies
    conjugate symmetry: ``c_{l,-m} = (-1)^m \overline{c_{l,m}}``.

    Parameters
    ----------
    coef_real : ndarray of shape ((l_max+1)**2,) or ((l_max+1)**2, D)
        Real-valued coefficients in flat ordering.

    Returns
    -------
    ndarray of same shape, complex128
        Complex coefficients.
    """
    coef_complex = np.empty_like(coef_real, dtype=np.complex128)
    n_coef = coef_real.shape[0]
    l_max = int(np.sqrt(n_coef)) - 1

    for l in range(l_max + 1):
        # m = 0
        idx_0 = l**2 + l
        coef_complex[idx_0] = coef_real[idx_0] + 0j

        # m > 0 and corresponding m < 0
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


def xyz2spherical(xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    xyz : ndarray of shape (n, 3)
        Cartesian coordinates (x, y, z). Points are assumed to lie on or
        near the unit sphere.

    Returns
    -------
    theta_phi : ndarray of shape (n, 2)
        Spherical coordinates ``[theta, phi]`` where ``theta`` is the
        polar angle (colatitude, 0 to pi) and ``phi`` is the azimuthal
        angle (-pi to pi).
    """
    theta = np.arccos(xyz[:, 2])
    xy_norm = np.linalg.norm(xyz[:, 0:2], axis=1)
    is_pole = xy_norm < _POLE_TOL
    phi = np.where(
        is_pole,
        0.0,
        np.sign(xyz[:, 1])
        * np.arccos(xyz[:, 0] / np.where(is_pole, 1.0, xy_norm)),
    )
    return np.array([theta, phi]).T


def spharm(
    l_max: int,
    coef: list[npt.ArrayLike],
    theta_range=None,
    phi_range=None,
):
    """Reconstruct surface coordinates from SPHARM coefficients.

    Parameters
    ----------
    l_max : int
        Maximum degree of spherical harmonics.
    coef : list of array-like
        Real SPHARM coefficients. ``coef[l]`` has shape ``(2*l+1, 3)``
        and ``coef[l][l+m]`` holds ``(c_x, c_y, c_z)`` for degree ``l``
        and order ``m``.
    theta_range : array-like of shape (n_theta,), optional
        Polar angle values (colatitude, 0 to pi). Defaults to
        ``np.linspace(0, pi, 90)``.
    phi_range : array-like of shape (n_phi,), optional
        Azimuthal angle values (0 to 2*pi). Defaults to
        ``np.linspace(0, 2*pi, 180)``.

    Returns
    -------
    x, y, z : ndarray of shape (n_theta, n_phi)
        Reconstructed surface coordinates.
    """
    if theta_range is None:
        theta_range = np.linspace(0, np.pi, 90)
    if phi_range is None:
        phi_range = np.linspace(0, 2 * np.pi, 180)

    theta_grid, phi_grid = np.meshgrid(theta_range, phi_range, indexing="ij")
    B = _real_sph_harm_basis_matrix(l_max, theta_grid.ravel(), phi_grid.ravel())

    coef_matrix = np.vstack(
        [coef[l] for l in range(l_max + 1)]
    )  # ((l_max+1)^2, 3)

    coords = B @ coef_matrix  # (N, 3)
    n_theta = len(theta_range)
    n_phi = len(phi_range)

    x = coords[:, 0].reshape(n_theta, n_phi)
    y = coords[:, 1].reshape(n_theta, n_phi)
    z = coords[:, 2].reshape(n_theta, n_phi)
    return x, y, z


def cvt_spharm_coef_to_list(
    coef: npt.NDArray[np.float64],
) -> list[npt.NDArray[np.float64]]:
    """Convert SPHARM coefficient matrix to a nested list by degree.

    Parameters
    ----------
    coef : ndarray of shape ((l_max+1)**2, D) or (D, (l_max+1)**2)
        SPHARM coefficient matrix. ``D`` is the number of components of
        the field expanded on the sphere (``D=3`` for 3D Cartesian
        coordinates).  Both orientations are accepted; if the second
        axis matches ``(l_max+1)**2``, the matrix is transposed.

    Returns
    -------
    coef_list : list of ndarray
        ``coef_list[l]`` has shape ``(2*l+1, D)`` for degree ``l``.

    Raises
    ------
    ValueError
        If ``coef`` is not 2-D, or neither axis is a perfect square
        (``(l_max+1)**2``).
    """
    coef_arr = np.asarray(coef)
    if coef_arr.ndim != 2:
        raise ValueError(
            f"coef must be 2-D ((n_lm, D) or (D, n_lm)); got shape {coef_arr.shape}."
        )

    n_rows, n_cols = coef_arr.shape
    rows_sqrt = np.sqrt(n_rows)
    cols_sqrt = np.sqrt(n_cols)
    if rows_sqrt.is_integer():
        coef_per_lm = coef_arr
        lmax = int(rows_sqrt) - 1
    elif cols_sqrt.is_integer():
        coef_per_lm = coef_arr.T
        lmax = int(cols_sqrt) - 1
    else:
        raise ValueError(
            f"Invalid coefficient shape {coef_arr.shape}: neither axis is a "
            f"perfect square ((l_max+1)**2)."
        )

    coef_list = [
        np.array([coef_per_lm[l**2 + l + m] for m in range(-l, l + 1, 1)])
        for l in range(0, lmax + 1, 1)
    ]
    return coef_list
