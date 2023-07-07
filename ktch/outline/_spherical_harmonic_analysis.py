"""Spherical Harmonic (SPHARM) Analysis """

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

from typing import List
import dataclasses

import numpy as np
import scipy as sp

import numpy.typing as npt

from sklearn.base import BaseEstimator, TransformerMixin


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

    def _transform_single(self, X, theta=None, phi=None):
        """SPHARM coefficients of a single outline.

        Parameters
        ----------
        X: array-like of shape (n_coords, n_dim)
                Coordinate values of an outline in n_dim (2 or 3).

        t: array-like of shape (n_coords, ), optional
                A parameter indicating the position on the outline.
                If `t=None`, then t is calculated based on the coordinate values with the linear interpolation.

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
                Coordinate values of n_samples. The i-th array-like whose shape (n_coords_i, 3) represents 3D coordinate values of the i-th sample .

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


class PCContribDisplay:
    """Shape variation visualization along PC axes.


    Parameters
    ------------------------
    pca:

    n_PCs:

    sd_values:

    lmax:

    theta:

    standardized:


    dpi:

    morph_color:


    morph_alpha:




    Attributes
    ------------------------
    XXX : matplotlib Artist
        .
    ax_ : matplotlib Axes
        Axes with shape variation.
    figure_ : matplotlib Figure
        Figure containing the shape variation.


    """

    def __init__(self):
        pass

    def plot(
        self,
        *,
        include_values=True,
        cmap="viridis",
        xticks_rotation="horizontal",
        values_format=None,
        ax=None,
        colorbar=True,
    ):
        """Plot visualization.
        Parameters
        ------------------------
        include_values : bool, default=True
            Includes values in confusion matrix.
        cmap : str or matplotlib Colormap, default='viridis'
            Colormap recognized by matplotlib.
        xticks_rotation : {'vertical', 'horizontal'} or float, \
                         default='horizontal'
            Rotation of xtick labels.
        values_format : str, default=None
            Format specification for values in confusion matrix. If `None`,
            the format specification is 'd' or '.2g' whichever is shorter.
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.
        colorbar : bool, default=True
            Whether or not to add a colorbar to the plot.
        Returns
        ------------------------
        display : :class:`~ktch.outline.PCContribDisplay`
        """

        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        self.figure_ = fig
        self.ax_ = ax

        return self

    @classmethod
    def from_estimator(self):
        """Create a"""
        pass


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
    if type(coefficients[0]) is npt.NDArray:
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

    coef: npt.ArrayLike = None
    n_degree: int = None

    def from_list(self, coef_list: list):
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

        self.coef = coef_arr
        self.n_degree = n_degree

    def from_array(self, coef_arr: npt.NDArray):
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

        self.coef = coef_arr
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
                m = m + l
            elif type(m) is slice:
                m = slice(m.start + l, m.stop + l, m.step)
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

            if l > len(self.coef):
                raise ValueError(f"l must be less than {len(self.coef)}")

            if abs(m) > l:
                raise ValueError(f"abs(m) must be less than {l}")

            self.coef[l][-l + self.n_degree + m] = value

        elif type(lm) is int:
            l = lm

            if len(value) != 2 * l + 1:
                raise ValueError(f"len(value) must be {2*l+1}")

            if l > len(self.coef):
                raise ValueError(f"l must be less than {len(self.coef)}")

            row = np.zeros(2 * self.n_degree + 1)
            row[(-l + self.n_degree) : (l + self.n_degree + 1)] = value
            self.coef[l] = row

        else:
            raise ValueError("Indices must be int or tuple of int")
