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
import scipy as sp
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils import SymmetricIndexArray


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

    def __init__(self, n_harmonics=10, reflect=False, metric="", impute=False):
        # self.dtype = dtype
        self.n_harmonics = n_harmonics

    def fit_transform(self, X, theta=None, phi=None):
        """SPHARM coefficients of outlines."""

        spharm_coef = None

        return spharm_coef

    def _transform_single(self, X, theta):
        """SPHARM coefficients of a single outline.

        Parameters
        ----------
        X: array-like of shape (n_coords, n_dim)
                Coordinate values of an outline in n_dim (2 or 3).

        theta: array-like of shape (n_coords,2)
                Parameters indicating the position on the surface.


        Returns
        ------------------------
        spharm_coef: list of coeffients
            Returns the coefficients of Fourier series.

        ToDo
        ------------------------
        * EHN: 3D outline
        """

        spharm_coef = None

        return spharm_coef

    def transform(self, X, theta=None, phi=None):
        """Transform X to a SPHARM coefficients.

        Parameters
        ------------------------
        X: list of array-like
                Coordinate values of n_samples.
                The i-th array-like whose shape (n_coords_i, 3) represents
                3D coordinate values of the i-th sample .

        theta: array-like of shape (n_coords, )
            Array-like of theta values.

        phi: array-like of shape (n_coords, )
            Array-like of phi values.

        Returns
        ------------------------
        X_transformed: array-like of shape (n_samples, n_coefficients)
            Returns the array-like of SPHARM coefficients.
        """

        X_transformed = None

        return X_transformed

    def _inverse_transform_single(self, lmax, coef_list, theta_range, phi_range):
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
        x, y, z: tuple of array_like
            Coordinate values of SPHARM.

        """
        x = 0
        y = 0
        z = 0
        # coef_list = cvt_spharm_coef_from_SlicerSALT_to_list(coef_naive)
        for l in range(lmax + 1):
            m, theta, phi = np.meshgrid(np.arange(-l, l + 1, 1), theta_range, phi_range)
            coef = coef_list[l]
            x = x + np.sum(
                sp.special.sph_harm(m, l, theta, phi) * coef[:, 0].reshape((-1, 1)),
                axis=1,
            )
            y = y + np.sum(
                sp.special.sph_harm(m, l, theta, phi) * coef[:, 1].reshape((-1, 1)),
                axis=1,
            )
            z = z + np.sum(
                sp.special.sph_harm(m, l, theta, phi) * coef[:, 2].reshape((-1, 1)),
                axis=1,
            )

        return np.real(x), np.real(y), np.real(z)


###########################################################
#
#   utility functions
#
###########################################################


def xyz2spherical(xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
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
    print(sph_mat.shape)

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
