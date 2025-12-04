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

import dataclasses
import warnings
from typing import List

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.parallel import Parallel, delayed


class SphericalHarmonicAnalysis(TransformerMixin, BaseEstimator):
    r"""Spherical Harmonic (SPHARM) Analysis


    Notes
    ------------------------
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
    ------------------------

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

    def fit_transform(self, X, theta_phi):
        """SPHARM coefficients of outlines."""

        return self.transform(X, theta_phi)

    def _transform_single(self, X, theta_phi):
        """SPHARM coefficients of a single outline.

        Parameters
        ----------
        X: array-like of shape (n_coords, n_dim)
                Coordinate values of a surface.

        theta_phi: array-like of shape (n_coords,2)
                Parameters indicating the position on the surface.

        Returns
        ------------------------
        X_transformed: array-like
            Returns the SPHARM coefficients.
        """

        l_max = self.n_harmonics
        theta = theta_phi[:, 0]
        phi = theta_phi[:, 1]

        lm2j = np.array([[l, m] for l in range(l_max + 1) for m in range(-l, l + 1)])

        A_Mat = np.array([sp.special.sph_harm_y(l, m, theta, phi) for l, m in lm2j])

        sol = sp.linalg.lstsq(A_Mat.T, X)
        c_x, c_y, c_z = sol[0].T

        X_transformed = np.concatenate([c_x, c_y, c_z], axis=-1)

        return X_transformed

    def transform(self, X, theta_phi):
        """Transform X to a SPHARM coefficients.

        Parameters
        ------------------------
        X: list of array-like
            Coordinate values of n_samples.
            The i-th array-like whose shape (n_coords_i, 3) represents
            3D coordinate values of the i-th sample .

        theta_phi: list of array-like of shape (n_coords, 2)
            Surface parameter of n_samples.
            The i-th array-like of theta and phi values whose shape is (n_coords_i, 2).

        Returns
        ------------------------
        X_transformed: array-like of shape (n_samples, n_coefficients)
            Returns the array-like of SPHARM coefficients.
        """

        if isinstance(X, pd.DataFrame):
            X_ = [row.dropna().to_numpy().reshape(3, -1).T for idx, row in X.iterrows()]
        else:
            X_ = X

        X_transformed = np.stack(
            Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(self._transform_single)(X_[i], theta_phi[i])
                for i in range(len(X_))
            )
        )

        return X_transformed

    def _inverse_transform_single(
        self,
        X_transformed,
        theta_range,
        phi_range,
        l_max=None,
    ):
        """SPHARM

        Parameters
        ------------------------
        lmax: int
            Degree of SPHARM to use how far
        coef_list: list
            SPHARM coefficients
            coef_naive[l,m] corresponding to the coefficient of degree l and order m
        theta_range: array_like
        phi_range: array_like


        Returns
        ------------------------
        X_coords: array-like of shape (n_theta, n_phi, 3)
            Coordinate values of SPHARM.

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
        theta_range=np.linspace(0, np.pi, 90),
        phi_range=np.linspace(0, 2 * np.pi, 180),
        l_max=None,
    ):
        """Inverse SPHARM transform
        Parameters
        ------------------------
        X_transformed: array-like of shape (n_samples, n_coefficients)
            SPHARM coefficients.
        theta_range: array_like
        phi_range: array_like
        lmax: int
            Degree of SPHARM to use how far

        Returns
        ------------------------
        X_coords: array-like of shape (n_samples, n_theta, n_phi, 3)
            Coordinate values of reconstructed surfaces.
        """
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
    """Convert Cartesian coordinates to spherical coordinates (theta, phi)"""
    theta = np.arccos(xyz[:, 2])
    phi = np.sign(xyz[:, 1]) * np.arccos(
        xyz[:, 0] / np.linalg.norm(xyz[:, 0:2], axis=1)
    )
    return np.array([theta, phi]).T


def spharm(
    l_max: int,
    coef: List[npt.ArrayLike],
    theta_range=np.linspace(0, np.pi, 90),
    phi_range=np.linspace(0, 2 * np.pi, 180),
    threshold_imag_parts: float = 1e-10,
):
    """SPHARM

    Parameters
    ----------
    l_max: int
        Degree of SPHARM to use how far
    coef: List (l_max + 1) of array_like (2l + 1, 3) of complex
        SPHARM coefficients
        coef[l][m] is the coefficients (c_x, c_y, c_z) of degree l and order m.
    theta_range: array_like
    phi_range: array_like


    Returns
    ----------
    x, y, z: tuple of array_like
        Coordinate values of SPHARM.

    """

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
            UserWarning(
                f"The coordinates have significant imaginary parts {total_imag_parts}."
            )
        )

    x, y, z = np.real(coords)
    return x, y, z


def cvt_spharm_coef_to_list(coef):
    coef_ = coef.reshape((-1, 3))
    lmax = int(np.sqrt(coef_.shape)[0] - 1)
    coef_list = [
        np.array([coef_[l**2 + l + m] for m in range(-l, l + 1, 1)])
        for l in range(0, lmax + 1, 1)
    ]
    return coef_list
