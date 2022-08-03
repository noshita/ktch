"""Elliptic Fourier Analysis"""

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


class EllipticFourierAnalysis(TransformerMixin, BaseEstimator):
    r"""Elliptic Fourier Analysis (EFA)
    

    Notes
    ------------

    :cite:`EFA-KUHL:1982bq` 

    .. math:: 

        \begin{align}
            x(l) &= \frac{a_0}{2} + \sum_{i=1}^{n} \left[ a_i \cos\left(\frac{2\pi i t}{T}\right) + b_i \sin\left(\frac{2\pi i t}{T}\right) \right]\\
            y(l) &= \frac{c_0}{2} + \sum_{i=1}^{n} \left[ c_i \cos\left(\frac{2\pi i t}{T}\right) + d_i \sin\left(\frac{2\pi i t}{T}\right) \right]\\
        \end{align}


    See the examples of implimentation :cite:`EFA-Claude2008,EFA-Bonhomme2013`.


    References
    ------------
    .. bibliography::
         :keyprefix: EFA-


    """

    def __init__(self, n_harmonics=20, reflect=False, metric="", impute=False):
        # self.dtype = dtype
        self.n_harmonics = n_harmonics

    def fit_transform(self, X, t=None):
        """Fit the model with X.

        Parameters
        ------------
        X: list of array-like
                Coordinate values of n_samples. The i-th array-like whose shape (n_coords_i, 2) represents 2D coordinate values of the i-th sample .

        t: list of array-like, optional
                Parameters indicating the position on the outline of n_samples. The i-th ndarray whose shape (n_coords_i, ) corresponds to each coordinate value in the i-th element of X. If `t=None`, then t is calculated based on the coordinate values with the linear interpolation.

        Returns
        ------------
        X_transformed: array-like of shape (n_samples, (1+2*n_harmonics)*n_dim)
            Returns the array-like of coefficients.
        """

        if t is None:
            t = [
                np.array(
                    [0]
                    + [
                        sp.spatial.distance.euclidean(x[i], x[i + 1])
                        for i in range(len(x) - 1)
                    ]
                )
                for x in X
            ]

        if len(t) != len(X):
            raise ValueError("t must have a same length of X ")

        T = np.array([np.sum(t_) for t_ in t])

        X_transformed = np.array(
            [self._fit_transform_single(X[i], t[i]) for i in range(len(X))]
        )

        return X_transformed

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
        -------
        X_transformed: list of coeffients
            Returns the coefficients of Fourier series.

        ToDo
        -------
        * EHN: 3D outline
        """

        X_arr = np.array(X)
        n_harmonics = self.n_harmonics

        if t is None:
            dt = np.array(
                [distance.euclidean(x[i], x[i + 1]) for i in range(len(x) - 1)]
            )
        else:
            dt = np.append(0, t[1:] - t[:-1])

        tp = np.cumsum(dt)

        dx = np.append(X_arr[0, 0] - X_arr[-1, 0], X_arr[1:, 0] - X_arr[:-1, 0])
        dy = np.append(X_arr[0, 1] - X_arr[-1, 1], X_arr[1:, 1] - X_arr[:-1, 1])

        if len(t) != len(X_arr):
            raise ValueError("t must have a same length of X ")

        T = np.sum(t)

        a0 = 2 / T * np.sum(X_arr[:, 0] * t)
        c0 = 2 / T * np.sum(X_arr[:, 1] * t)

        print(a0, c0)
        print(dx[1:], dt[1:])

        an = [
            (T / (2 * (np.pi**2) * (n**2)))
            * np.sum(
                (dx[1:] / dt[1:])
                * (
                    np.cos(2 * np.pi * n * tp[1:] / T)
                    - np.cos(2 * np.pi * n * tp[:-1] / T)
                )
            )
            for n in range(1, n_harmonics + 1, 1)
        ]
        bn = [
            (T / (2 * (np.pi**2) * (n**2)))
            * np.sum(
                (dx[1:] / dt[1:])
                * (
                    np.sin(2 * np.pi * n * tp[1:] / T)
                    - np.sin(2 * np.pi * n * tp[:-1] / T)
                )
            )
            for n in range(1, n_harmonics + 1, 1)
        ]
        cn = [
            (T / (2 * (np.pi**2) * (n**2)))
            * np.sum(
                (dy[1:] / dt[1:])
                * (
                    np.cos(2 * np.pi * n * tp[1:] / T)
                    - np.cos(2 * np.pi * n * tp[:-1] / T)
                )
            )
            for n in range(1, n_harmonics + 1, 1)
        ]
        dn = [
            (T / (2 * (np.pi**2) * (n**2)))
            * np.sum(
                (dy[1:] / dt[1:])
                * (
                    np.sin(2 * np.pi * n * tp[1:] / T)
                    - np.sin(2 * np.pi * n * tp[:-1] / T)
                )
            )
            for n in range(1, n_harmonics + 1, 1)
        ]

        X_transformed = [a0, c0, an, bn, cn, dn]
        return X_transformed

    # def transform(self, X):

    #     return X_transformed

    # def fit_transform(self, X):
    #     pass
