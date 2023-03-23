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

from operator import index
import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation as R
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import ClassNamePrefixFeaturesOutMixin


class EllipticFourierAnalysis(TransformerMixin, BaseEstimator):
    r"""
    Elliptic Fourier Analysis (EFA) 

    Parameters
    ------------
    n_harmonics: int, default=20
        harmonics

    reflect: bool, default=False
        reflect

    metric: str
        metric

    impute: bool, False
        impute

    Notes
    ------------
    EFA is widely applied for two-dimensional outline analysis [Kuhl_Giardina_1982]_.

    .. math::
        \begin{align}
            x(l) &=
            \frac{a_0}{2} + \sum_{i=1}^{n}
            \left[ a_i \cos\left(\frac{2\pi i t}{T}\right)
            + b_i \sin\left(\frac{2\pi i t}{T}\right) \right]\\
            y(l) &=
            \frac{c_0}{2} + \sum_{i=1}^{n}
            \left[ c_i \cos\left(\frac{2\pi i t}{T}\right)
            + d_i \sin\left(\frac{2\pi i t}{T}\right) \right]\\
        \end{align}

    References
    ------------
    .. [Kuhl_Giardina_1982] Kuhl, F.P., Giardina, C.R. (1982) Elliptic Fourier features of a closed contour. Comput. Graph. Image Process. 18: 236–258.


    """

    def __init__(self, n_harmonics=20, reflect=False, metric="", impute=False):
        # self.dtype = dtype
        self.n_harmonics = n_harmonics
        self.reflect = reflect
        self.metric = metric
        self.impute = impute

    def transform(self, X, t=None):
        return self.fit_transform(X, t)

    def fit_transform(self, X, t=None, as_frame=False):
        """

        Fit the model with X.

        Parameters
        ------------
        X: {list of array-like, array-like} of shape (n_samples, n_coords, 2)
            Coordinate values of n_samples.
            The i-th array-like of shape (n_coords_i, 2) represents
            2D coordinate values of the i-th sample.

        t: array-like of shape (n_samples, n_coords), optional
            Parameters indicating the position on the outline of n_samples.
            The i-th ndarray of shape (n_coords_i, ) corresponds to
            each coordinate value in the i-th element of X.
            If `t=None`, then t is calculated based on
            the coordinate values with the linear interpolation.

        Returns
        ------------
        X_transformed: array-like of shape (n_samples, (1+2*n_harmonics)*n_dim)
            Returns the array-like of coefficients.

        """

        n_harmonics = self.n_harmonics

        if t is None:
            t_ = [None for i in range(len(X))]

        if len(t_) != len(X):
            raise ValueError("t must have a same length of X ")

        if as_frame:
            X_transformed = pd.concat(
                [
                    self._fit_transform_single(X[i], t=t_[i], as_frame=True)
                    for i in range(len(X))
                ],
                axis=0,
            )
            X_transformed["specimen_id"] = [
                i for i in range(len(X)) for j in range(n_harmonics + 1)
            ]
            X_transformed = X_transformed.reset_index().set_index(
                ["specimen_id", "harmonics"]
            )
        else:
            if isinstance(X, pd.DataFrame):
                X_ = [x[0] for x in X.to_numpy()]
            else:
                X_ = X

            X_transformed = np.stack(
                [
                    self._fit_transform_single(np.array(X_[i]), t_[i], as_frame=False)
                    for i in range(len(X))
                ]
            )

        return X_transformed

    def _fit_transform_single(
        self,
        X,
        t=None,
        norm=True,
        duplicated_points="infinitesimal",
        as_frame=False,
    ):
        """Fit the model with a signle outline.

        Parameters
        ----------
        X: array-like of shape (n_coords, 2)
                Coordinate values of an 2D outline.

        t: array-like of shape (n_coords+1, ), optional
                A parameter indicating the position on the outline.
                Both t[0] and t[n_coords] corresponds to X[0].
                If `t=None`, then t is calculated based on
                the coordinate values with the linear interpolation.

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

        dx = np.append(
            X_arr[0, 0] - X_arr[-1, 0],
            X_arr[1:, 0] - X_arr[:-1, 0],
        )
        dy = np.append(
            X_arr[0, 1] - X_arr[-1, 1],
            X_arr[1:, 1] - X_arr[:-1, 1],
        )

        if t is None:
            dt = np.sqrt(dx**2 + dy**2)
            tp = np.append(0, np.cumsum(dt))
            T = np.sum(dt)
        else:
            # TODO: add test
            dt = t[1:] - t[:-1]
            tp = t
            T = t[-1]

        if duplicated_points == "infinitesimal":
            dt[dt < 10**-10] = 10**-10
        elif duplicated_points == "deletion":
            idx_duplicated_points = np.where(dt == 0)[0]
            if len(idx_duplicated_points) > 0:
                dx = np.delete(dx, idx_duplicated_points)
                dy = np.delete(dy, idx_duplicated_points)
                dt = np.delete(dt, idx_duplicated_points)
                tp = np.delete(
                    tp,
                    (np.array(idx_duplicated_points) + 1).tolist(),
                )
                X_arr = np.delete(X_arr, idx_duplicated_points, 0)
        else:
            raise ValueError(
                "'duplicated_points' must be 'infinitesimal' or 'deletion'"
            )

        if len(tp) != len(X_arr) + 1:
            raise ValueError(
                "len(t) must have a same len(X) + 1), len(t): "
                + str(len(tp))
                + ", len(X)+1: "
                + str(len(X_arr))
            )

        a0 = 2 / T * np.sum(X_arr[:, 0] * dt)
        c0 = 2 / T * np.sum(X_arr[:, 1] * dt)

        an = [
            (T / (2 * (np.pi**2) * (n**2)))
            * np.sum(
                (dx / dt)
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
                (dx / dt)
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
                (dy / dt)
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
                (dy / dt)
                * (
                    np.sin(2 * np.pi * n * tp[1:] / T)
                    - np.sin(2 * np.pi * n * tp[:-1] / T)
                )
            )
            for n in range(1, n_harmonics + 1, 1)
        ]

        an = [a0] + an
        bn = [0] + bn
        cn = [c0] + cn
        dn = [0] + dn

        if norm:
            an, bn, cn, dn = self._normalize(an, bn, cn, dn)

        if as_frame:
            harmonics = pd.Series([i for i in range(n_harmonics + 1)])
            df_a = pd.DataFrame(an, index=harmonics)
            df_b = pd.DataFrame(bn, index=harmonics)
            df_c = pd.DataFrame(cn, index=harmonics)
            df_d = pd.DataFrame(dn, index=harmonics)
            X_transformed = pd.concat([df_a, df_b, df_c, df_d], axis=1)
            X_transformed.columns = ["an", "bn", "cn", "dn"]
            X_transformed.index.name = "harmonics"
        else:
            # X_transformed = np.stack([an, bn, cn, dn], axis=1)
            X_transformed = np.concatenate([an, bn, cn, dn]).reshape(-1)

        return X_transformed

    def _normalize(self, an, bn, cn, dn):
        a1 = an[1]
        b1 = bn[1]
        c1 = cn[1]
        d1 = dn[1]

        theta = (1 / 2) * np.arctan2(
            2 * (a1 * b1 + c1 * d1), (a1**2 + c1**2 - b1**2 - d1**2)
        )
        if theta < 0:
            theta = theta + np.pi

        a_s = a1 * np.cos(theta) + b1 * np.sin(theta)
        c_s = c1 * np.cos(theta) + d1 * np.sin(theta)
        scale = np.sqrt(a_s**2 + c_s**2)
        psi = np.arctan2(c_s, a_s)
        if psi < 0:
            psi = 2 * np.pi + psi

        coef_norm_list = []
        r_psi = rotaion_matrix_2d(-psi)
        for n in range(1, len(an)):
            r_ntheta = rotaion_matrix_2d(n * theta)
            coef_orig = np.array([[an[n], bn[n]], [cn[n], dn[n]]])
            coef_norm_tmp = (1 / scale) * np.dot(np.dot(r_psi, coef_orig), r_ntheta)
            coef_norm_list.append(coef_norm_tmp.reshape(-1))

        coef_norm = np.stack(coef_norm_list)
        An = np.append(an[0], coef_norm[:, 0])
        Bn = np.append(bn[0], coef_norm[:, 1])
        Cn = np.append(cn[0], coef_norm[:, 2])
        Dn = np.append(dn[0], coef_norm[:, 3])

        return An, Bn, Cn, Dn

    def _inverse_transform_single(
        self,
        X_transformed,
        t_num=100,
        norm=True,
        as_frame=False,
    ):
        coef = X_transformed
        if as_frame:
            an = coef["an"][1:]
            bn = coef["bn"][1:]
            cn = coef["cn"][1:]
            dn = coef["dn"][1:]
        else:
            an, bn, cn, dn = coef.reshape([4, -1])
            an = an[1:]
            bn = bn[1:]
            cn = cn[1:]
            dn = dn[1:]

        n_max = len(an)

        theta = np.linspace(0, 2 * np.pi, t_num)

        cos = np.cos(np.tensordot(np.arange(1, n_max + 1, 1), theta, 0))
        sin = np.sin(np.tensordot(np.arange(1, n_max + 1, 1), theta, 0))

        x = np.dot(an, cos) + np.dot(bn, sin)
        y = np.dot(cn, cos) + np.dot(dn, sin)

        X_coords = np.stack([x, y], 1)

        return X_coords

    def inverse_transform(self, X_transformed, t_num=100, as_frame=False):
        X_list = []
        sp_num = X_transformed.shape[0]

        for i in range(sp_num):
            if as_frame:
                coef = X_transformed.loc[i]
                X = self._inverse_transform_single(coef, as_frame=as_frame)
                df_X = pd.DataFrame(X, columns=["x", "y"])
                df_X["coord_id"] = [coord_id for coord_id in range(len(X))]
                df_X["specimen_id"] = i
                X_list.append(df_X)
            else:
                coef = X_transformed[i]
                X = self._inverse_transform_single(coef, as_frame=as_frame)
                X_list.append(X)

        if as_frame:
            X_coords = pd.concat(X_list)
            X_coords = X_coords.set_index(["specimen_id", "coord_id"])
        else:
            X_coords = X_list

        return X_coords

    def get_feature_names_out(self):
        an = ["a_" + str(i) for i in range(self.n_harmonics + 1)]
        bn = ["b_" + str(i) for i in range(self.n_harmonics + 1)]
        cn = ["c_" + str(i) for i in range(self.n_harmonics + 1)]
        dn = ["d_" + str(i) for i in range(self.n_harmonics + 1)]
        return np.asarray(an + bn + cn + dn, dtype=object)

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        return (self.n_harmonics + 1) * 4


def rotaion_matrix_2d(theta):
    rot_mat = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return rot_mat
