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
from abc import ABCMeta
from typing import Optional

import numpy as np
import numpy.typing as npt
import scipy as sp
from sklearn.base import (BaseEstimator, ClassNamePrefixFeaturesOutMixin,
                          TransformerMixin)
from sklearn.utils.parallel import Parallel, delayed


class SphericalHarmonicAnalysis(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator, metaclass=ABCMeta):
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
        n_harmonics: int = 10,
        reflect=False,
        metric="",
        impute=False,
        n_jobs: Optional[int] = None,
    ):
        self.n_harmonics = n_harmonics
        self.n_jobs = n_jobs

    def fit_transform(self, X, t=None, norm=True):
        """SPHARM coefficients of outlines."""

        spharm_coef = None

        return spharm_coef

    def _transform_single(self, X, t, norm=True):
        """SPHARM coefficients of a single outline.

        Parameters
        ----------
        X: array-like of shape (n_coords, 3)
                Coordinate values on a surface.

        t: array-like of shape (n_coords, 2), optional
                Parameters $\mathbf{t}=(\theta, \phi)$ indicating the position on the surface.
                If `t=None`, then theta and phi are calculated
                based on the coordinate values with `parameterization`.

        Returns
        ------------------------
        c: array of shape (1, 3*(n_harmonics+1)(n_harmonics+2)/2)
            Returns the coefficients of SPHARM series.

        """

        theta, phi = t.T
        lmax = self.n_harmonics
        lm2j = np.array([[l, m] for l in range(lmax + 1) for m in range(-l, l + 1)])

        A_Mat = np.array([sp.special.sph_harm_y(l, m, theta, phi) for l, m in lm2j])

        sol = sp.linalg.lstsq(A_Mat.T, X)
        c = sol[0].T.reshape(1,-1)

        return c

    def transform(self, X, t, norm=True):
        """Transform coordinates with parameters to SPHARM coefficients.

        Parameters
        ------------------------
        X: list (n_samples) of array-like of shape (n_coords_i, 3)
            Coordinate values of n_samples. The i-th array-like whose shape (n_coords_i, 3) represents 3D coordinate values of the i-th sample .

        t: array-like of shape (n_samples, n_coords_i, 2)
            Parameters of the i-th sample.
            The i-th array-like whose shape (n_coords_i, 2) represents
            the spherical parameters (theta, phi) of the i-th sample.

        norm: bool, default=True
            If True, the SPHARM coefficients are normalized.

        Returns
        ------------------------
        X_transformed: array-like of shape (n_samples, n_coefficients)
            Returns the array-like of SPHARM coefficients.
        """

        # X_transformed = Parallel(
        #     n_jobs=self.n_jobs)(
        #         delayed(self._ripser_diagram)(x) for x, (theta, phi) in zip(X, t)
        #         )

        X_transformed = np.array(
            [self._transform_single(x, theta, phi) for x, (theta, phi) in zip(X, t)]
        )

        return X_transformed

    def _inverse_transform_single(self, coef_list, lmax, theta_range, phi_range):
        """SPHARM

        Parameters
        ------------------------
        coef_list: list
            SPHARM coefficients
            coef_naive[l,m] corresponding to the coefficient of degree l and order m
        lmax: int
            Degree of SPHARM to use how far
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

    def inverse_transform(self, X_transformed, n_theta, n_phi):
        """Inverse transform SPHARM coefficients to coordinates.

        Parameters
        ----------
        X_transformed : Array-like of shape (n_samples, n_coefficients)
            SPHARM coefficients to be transformed back to coordinates.
        n_theta : int
            Number of theta values to generate.
        n_phi : int
            Number of phi values to generate.

        Returns
        -------
        X_coords : Array-like of shape (n_samples, n_theta, n_phi, 3)
            Returns the coordinates reconstructed from the SPHARM coefficients.
        """

        lmax = self.n_harmonics
        theta_range = np.linspace(0, np.pi, n_theta)
        phi_range = np.linspace(0, 2 * np.pi, n_phi)

        X_coords = np.array(
            [
                self._inverse_transform_single(coef, lmax, theta_range, phi_range)
                for coef in X_transformed
            ]
        )

        return X_coords



###########################################################
#
#   utility functions
#
###########################################################


def spharm(
    n_degree,
    coefficients,
    theta_range=np.linspace(0, 2 * np.pi, 90),
    phi_range=np.linspace(0, np.pi, 180),
):
    """SPHARM

    Parameters
    ----------
    n_degree: int
        Degree (l) of SPHARM
    coefficients: list (of length 3) of array-like of shape (, ) or SPHARMCoefficients
        SPHARM coefficients
        coef[l,m] corresponding to the coefficient of degree l and order m
    theta_range: array_like
    phi_range: array_like


    Returns
    ----------
    x, y, z: tuple of array_like
        Coordinate values of surface.

    """
    x = 0
    y = 0
    z = 0
    coef_x = SPHARMCoefficients()
    coef_y = SPHARMCoefficients()
    coef_z = SPHARMCoefficients()
    if type(coefficients[0]) is np.ndarray:
        coef_x.from_array(coefficients[0])
        coef_y.from_array(coefficients[1])
        coef_z.from_array(coefficients[2])
    elif type(coefficients[0]) is list:
        coef_x.from_list(coefficients[0])
        coef_y.from_list(coefficients[1])
        coef_z.from_list(coefficients[2])
    elif type(coefficients[0]) is SPHARMCoefficients:
        coef_x, coef_y, coef_z = coefficients
    else:
        raise TypeError(
            "coefficients must be list of array, list, or SPHARMCoefficients"
        )

    for l in range(n_degree + 1):
        m, theta, phi = np.meshgrid(np.arange(-l, l + 1, 1), theta_range, phi_range)

        x = x + np.sum(
            sp.special.sph_harm(m, l, theta, phi) * coef_x[l].reshape((-1, 1)), axis=1
        )
        y = y + np.sum(
            sp.special.sph_harm(m, l, theta, phi) * coef_y[l].reshape((-1, 1)), axis=1
        )
        z = z + np.sum(
            sp.special.sph_harm(m, l, theta, phi) * coef_z[l].reshape((-1, 1)), axis=1
        )

    return np.real(x), np.real(y), np.real(z)


@dataclasses.dataclass
class SPHARMCoefficients:
    """SPHARM coefficients"""

    coef_arr: npt.ArrayLike = None
    n_degree: int = None

    def from_list(self, coef_list: list):
        """SPHARM coefficients from list.

        Parameters
        ----------
        coef_list : list
            `coef_list[l][m+l]` ($-l \geq m \geq l$) corresponds to $c_l^m$.
            `coef_list[l]` has $2l+1$ elements.

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        """
        if len(coef_list) == 0:
            raise ValueError("coef_list must be non-empty")

        n_degree = len(coef_list) - 1
        size_m_lmax = len(coef_list[-1])

        for l, coef_l in enumerate(coef_list):
            if len(coef_l) != 2 * l + 1:
                raise ValueError("coef_list must be (n_degree, 2*n_degree+1)")

        coef_arr = np.zeros([n_degree + 1, size_m_lmax], dtype=np.complex128)
        # print(coef_arr.shape)
        for l, coef_l in enumerate(coef_list):
            # print(coef_l, coef_arr[l])
            coef_arr[l, (-l + n_degree) : (l + n_degree + 1)] = coef_l

        self.coef_arr = coef_arr
        self.n_degree = n_degree

    def from_array(self, coef_arr: npt.NDArray):
        """SPHARM coefficients from array.

        Parameters
        ----------
        coef_arr : npt.NDArray of shape (lmax+1, 2lmax+1)
            `coef_arr[l, m+l]`  ($-l \geq m \geq l$) corresponds to $c_l^m$.
            `` `coef_arr[l, m+l]==0`.

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        ValueError
            _description_
        """
        if coef_arr.ndim != 2:
            raise ValueError("coef_arr must be 2-dimensional")

        n_degree_, size_m_lmax = coef_arr.shape
        n_degree = n_degree_ - 1

        if size_m_lmax != 2 * n_degree + 1:
            raise ValueError("coef_arr.shape must be (n_degree, 2*n_degree+1)")

        for l in range(n_degree):
            for m_ in range(2 * n_degree + 1):
                m = m_ - n_degree
                if abs(m) > l and coef_arr[l, m_] != 0:
                    raise ValueError(
                        "coef_arr[l, m_] when abs(m+n_degree) > l must be 0"
                    )

        self.coef_arr = coef_arr
        self.n_degree = n_degree

    def as_list(self) -> list:
        coef_list = [
            [row[m + self.n_degree] for m in range(-l, l + 1)]
            for l, row in enumerate(self.coef)
        ]
        return coef_list

    def as_array(self) -> npt.NDArray:
        return self.coef

    def __getitem__(self, lm):
        if type(lm) is tuple:
            if len(lm) > 2:
                raise ValueError("Indices must be less than two")
            else:
                l, m = lm

            if abs(m) > l:
                raise ValueError("abs(m) must be less than l")

            if type(m) is int:
                m = m + self.n_degree
            elif type(m) is slice:
                m = slice(m.start + self.n_degree, m.stop + self.n_degree, m.step)
            else:
                raise ValueError("m must be int or slice")

            return self.coef[(l, m)]

        elif type(lm) is int:
            l = lm

            return self.coef[l, (-l + self.n_degree) : (l + self.n_degree + 1)]

        else:
            raise ValueError("Indices must be int")

    def __setitem__(self, lm, value):
        if type(lm) is tuple:
            if len(lm) > 2:
                raise ValueError("Indices must be less than two")
            else:
                l, m = lm

            if l > len(self.coef_arr):
                raise ValueError(f"l must be less than {len(self.coef_arr)}")

            if abs(m) > l:
                raise ValueError(f"abs(m) must be less than {l}")

            self.coef_arr[l][-l + self.n_degree + m] = value

        elif type(lm) is int:
            l = lm

            if len(value) != 2 * l + 1:
                raise ValueError(f"len(value) must be {2 * l + 1}")

            if l > len(self.coef_arr):
                raise ValueError(f"l must be less than {len(self.coef_arr)}")

            row = np.zeros(2 * self.n_degree + 1)
            row[(-l + self.n_degree) : (l + self.n_degree + 1)] = value
            self.coef_arr[l] = row

        else:
            raise ValueError("Indices must be int or tuple of int")
