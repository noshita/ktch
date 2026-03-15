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

    .. math::
        \begin{align}
            \mathbf{p}(\theta, \phi) = \sum_{l=0}^{L} \sum_{m=-l}^l
            \left(
                c_l^m Y_l^m(\theta, \phi)
            \right)
        \end{align}

    , where :math:`Y_l^m(\theta, \phi)` are spherical harmonics:

    .. math::
        \begin{align}
            Y_l^m(\theta, \phi) = \sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}} P_l^m(\cos(\theta)) e^{im\phi}
        \end{align}

    , where :math:`P_n^m(x)` are associated Legendre polynomials:

    .. math::

        \begin{align}
            P_n^m(x) = (-1)^m (1-x^2)^{\frac{m}{2}} \frac{d^m}{dx^m} P_n(x)
        \end{align}

    , where :math:`P_n(x)` are Legendre polynomials, which are solutions of Legendre’s differential equation;

    .. math::
        (1-x^2)\frac{d^2 y}{dx^2} -2x \frac{dy}{dx} + n(n+1)y = 0.


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
        """Compute SPHARM coefficients for a single sample.

        Parameters
        ----------
        X: array-like of shape (n_coords, n_dim)
                Coordinate values of a surface.

        theta_phi: array-like of shape (n_coords,2)
                Parameters indicating the position on the surface.

        Returns
        -------
        X_transformed: array-like
            Returns the SPHARM coefficients.
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

        lm2j = np.array([[l, m] for l in range(l_max + 1) for m in range(-l, l + 1)])

        A_Mat = np.array([sp.special.sph_harm_y(l, m, theta, phi) for l, m in lm2j])

        sol = sp.linalg.lstsq(A_Mat.T, X)
        c_x, c_y, c_z = sol[0].T

        X_transformed = np.concatenate([c_x, c_y, c_z], axis=-1)

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
        X_transformed : ndarray of shape (n_coefficients,)
            Flat SPHARM coefficient vector for one sample.
        theta_range : array-like of shape (n_theta,)
            Polar angle values (colatitude, 0 to pi).
        phi_range : array-like of shape (n_phi,)
            Azimuthal angle values (0 to 2*pi).
        l_max : int, optional
            Maximum degree of harmonics to use. Defaults to
            ``self.n_harmonics``.

        Returns
        -------
        X_coords : ndarray of shape (n_theta, n_phi, 3)
            Reconstructed surface coordinates.
        """
        if l_max is None:
            l_max = self.n_harmonics

        x, y, z = spharm(
            l_max,
            cvt_spharm_coef_to_list(X_transformed.T),
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
        X_transformed : array-like of shape (n_samples, n_coefficients)
            SPHARM coefficients.
        theta_range : array-like of shape (n_theta,), optional
            Polar angle values (colatitude). Defaults to
            ``np.linspace(0, pi, 90)``.
        phi_range : array-like of shape (n_phi,), optional
            Azimuthal angle values. Defaults to
            ``np.linspace(0, 2*pi, 180)``.
        l_max : int, optional
            Maximum degree of harmonics to use. Defaults to
            ``self.n_harmonics``.

        Returns
        -------
        X_coords : ndarray of shape (n_samples, n_theta, n_phi, 3)
            Reconstructed surface coordinates.
        """
        if theta_range is None:
            theta_range = np.linspace(0, np.pi, 90)
        if phi_range is None:
            phi_range = np.linspace(0, 2 * np.pi, 180)
        if l_max is None:
            l_max = self.n_harmonics

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
    phi = np.where(
        xy_norm == 0,
        0.0,
        np.sign(xyz[:, 1])
        * np.arccos(xyz[:, 0] / np.where(xy_norm == 0, 1.0, xy_norm)),
    )
    return np.array([theta, phi]).T


def spharm(
    l_max: int,
    coef: list[npt.ArrayLike],
    theta_range=None,
    phi_range=None,
    threshold_imag_parts: float = 1e-10,
):
    """Reconstruct surface coordinates from SPHARM coefficients.

    Parameters
    ----------
    l_max : int
        Maximum degree of spherical harmonics.
    coef : list of array-like
        SPHARM coefficients. ``coef[l]`` has shape ``(2*l+1, 3)`` and
        ``coef[l][l+m]`` holds ``(c_x, c_y, c_z)`` for degree ``l``
        and order ``m``.
    theta_range : array-like of shape (n_theta,), optional
        Polar angle values (colatitude, 0 to pi). Defaults to
        ``np.linspace(0, pi, 90)``.
    phi_range : array-like of shape (n_phi,), optional
        Azimuthal angle values (0 to 2*pi). Defaults to
        ``np.linspace(0, 2*pi, 180)``.
    threshold_imag_parts : float, default=1e-10
        Tolerance for imaginary parts in the reconstructed coordinates.
        A warning is issued if the total imaginary magnitude exceeds
        this value.

    Returns
    -------
    x, y, z : ndarray of shape (n_theta, n_phi)
        Reconstructed surface coordinates.
    """
    if theta_range is None:
        theta_range = np.linspace(0, np.pi, 90)
    if phi_range is None:
        phi_range = np.linspace(0, 2 * np.pi, 180)

    l, m, theta, phi = np.meshgrid(
        np.arange(0, l_max + 1, 1),
        np.arange(-l_max, l_max + 1, 1),
        theta_range,
        phi_range,
        indexing="ij",
    )

    c = np.array(
        [np.pad(c_l, ((l_max - l, l_max - l), (0, 0))) for l, c_l in enumerate(coef)]
    )

    sph_mat = sp.special.sph_harm_y(l, m, theta, phi)

    coords = np.tensordot(c, sph_mat, axes=([0, 1], [0, 1]))

    total_imag_parts = np.abs(np.imag(coords)).sum()
    if total_imag_parts > threshold_imag_parts:
        warnings.warn(
            f"The coordinates have significant imaginary parts {total_imag_parts}.",
            UserWarning,
            stacklevel=2,
        )

    x, y, z = np.real(coords)
    return x, y, z


def cvt_spharm_coef_to_list(
    coef: npt.NDArray[np.float64],
) -> list[npt.NDArray[np.float64]]:
    """Convert flat SPHARM coefficient matrix to a nested list by degree.

    Parameters
    ----------
    coef : ndarray of shape (3, (l_max+1)**2) or ((l_max+1)**2, 3)
        SPHARM coefficient matrix.

    Returns
    -------
    coef_list : list of ndarray
        ``coef_list[l]`` has shape ``(2*l+1, 3)`` for degree ``l``.
    """
    coef_ = coef.reshape((-1, 3))
    lmax_plus_one = np.sqrt(coef_.shape[0])
    if not lmax_plus_one.is_integer():
        raise ValueError(
            f"Invalid coefficient count: {coef_.shape[0]} is not a perfect square "
            f"((lmax+1)^2)."
        )
    lmax = int(lmax_plus_one) - 1
    coef_list = [
        np.array([coef_[l**2 + l + m] for m in range(-l, l + 1, 1)])
        for l in range(0, lmax + 1, 1)
    ]
    return coef_list
