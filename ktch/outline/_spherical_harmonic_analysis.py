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

import numpy as np
import scipy as sp

from sklearn.base import BaseEstimator, TransformerMixin


class SphericalHarmonicAnalysis(TransformerMixin, BaseEstimator):
    r"""Spherical Harmonic (SPHARM) Analysis


    Notes
    ------------------------
    [Ritche_Kemp_1999]_, [Shen_etal_2009]_

    .. math::
        \begin{align}
            \mathbf{p}(\theta, \phi) = \sum_{l=0}^{l_\mathrm{max}} \sum_{m=-l}^l
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

    def __init__(self, n_harmonics=20, reflect=False, metric="", impute=False):
        # self.dtype = dtype
        self.n_harmonics = n_harmonics

    def fit_transform(self, X, t=None):
        """Fit the model with X.

        Parameters
        ------------------------
        X: list of array-like
                Coordinate values of n_samples. The i-th array-like whose shape (n_coords_i, 2) represents 2D coordinate values of the i-th sample .

        t: list of array-like, optional
                Parameters indicating the position on the outline of n_samples. The i-th ndarray whose shape (n_coords_i, ) corresponds to each coordinate value in the i-th element of X. If `t=None`, then t is calculated based on the coordinate values with the linear interpolation.

        Returns
        ------------------------
        spharm_coef: array-like of shape (n_samples, (1+2*n_harmonics)*n_dim)
            Returns the array-like of coefficients.
        """

        spharm_coef = None

        return spharm_coef

    def _fit_transform_single(self, X, t=None):
        """Fit the model with a signle outline.

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

    # def transform(self, X):

    #     return X_transformed

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
