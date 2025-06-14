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

from __future__ import annotations

from abc import ABCMeta
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import make_interp_spline
from sklearn.base import (BaseEstimator, ClassNamePrefixFeaturesOutMixin,
                          TransformerMixin)
from sklearn.decomposition import PCA
from sklearn.utils.parallel import Parallel, delayed


class EllipticFourierAnalysis(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator, metaclass=ABCMeta
):
    r"""
    Elliptic Fourier Analysis (EFA)

    Parameters
    ------------
    n_harmonics: int, default=20
        harmonics

    n_dim: int, default=2
        dimension

    reflect: bool, default=False
        reflect

    metric: str
        metric

    impute: bool, False
        impute

    Notes
    ------------
    EFA is widely applied for outline shape analysis
    in two-dimensional space [Kuhl_Giardina_1982]_.

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


    EFA is also applied for a closed curve in the three-dimensional space
    (e.g., [Lestrel_1997]_, [Lestrel_et_al_1997]_, and [Godefroy_et_al_2012]_).

    References
    ------------
    .. [Kuhl_Giardina_1982] Kuhl, F.P., Giardina, C.R. (1982) Elliptic Fourier features of a closed contour. Comput. Graph. Image Process. 18: 236–258. https://doi.org/10.1016/0146-664X(82)90034-X
    .. [Lestrel_1997]  Lestrel, P.E., 1997. Introduction and overview of Fourier descriptors, in: Fourier Descriptors and Their Applications in Biology. Cambridge University Press, pp. 22–44. https://doi.org/10.1017/cbo9780511529870.003
    .. [Lestrel_et_al_1997] Lestrel, P.E., Read, D.W., Wolfe, C., 1997. Size and shape of the rabbit orbit: 3-D Fourier descriptors, in: Lestrel, P.E. (Ed.), Fourier Descriptors and Their Applications in Biology. Cambridge University Press, pp. 359–378. https://doi.org/10.1017/cbo9780511529870.017
    .. [Godefroy_et_al_2012] Godefroy, J.E., Bornert, F., Gros, C.I., Constantinesco, A., 2012. Elliptical Fourier descriptors for contours in three dimensions: A new tool for morphometrical analysis in biology. C. R. Biol. 335, 205–213. https://doi.org/10.1016/j.crvi.2011.12.004

    """

    def __init__(
        self,
        n_harmonics: int = 20,
        n_dim: int = 2,
        reflect: bool = False,
        metric: str = "",
        impute: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        """
        ToDo
        -------
        * EHN: excluding position from the output
        """
        # self.dtype = dtype
        self.n_harmonics = n_harmonics
        if n_dim not in (2, 3):
            raise ValueError("n_dim must be 2 or 3")
        else:
            self.n_dim = n_dim
        self.reflect = reflect
        self.metric = metric
        self.impute = impute
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit_transform(self, X, t=None, norm=True, return_orientation_scale=False):
        return self.transform(
            X, t, norm=norm, return_orientation_scale=return_orientation_scale
        )

    def transform(
        self,
        X: list(npt.ArrayLike) | npt.ArrayLike,
        t: npt.ArrayLike = None,
        norm: bool = True,
        return_orientation_scale: bool = False,
    ) -> npt.ArrayLike:
        """EFA.

        Parameters
        ------------
        X: {list of array-like, array-like} of shape (n_samples, n_coords, dim)
            Coordinate values of n_samples.
            The i-th array-like of shape (n_coords_i, 2) represents
            2D coordinate values of the i-th sample.

        t: array-like of shape (n_samples, n_coords), optional
            Parameters indicating the position on the outline of n_samples.
            The i-th ndarray of shape (n_coords_i, ) corresponds to
            each coordinate value in the i-th element of X.
            If `t=None`, then t is calculated based on
            the coordinate values with the linear interpolation.

        norm: bool, default=True
            Normalize the elliptic Fourier coefficients
            by the major axis of the 1st ellipse.

        return_orientation_scale: bool, default=False
            Return orientation and scale of the outline.
            If `norm=True`, the orientation and scale are normalized by the 1st ellipse.
            Then, the orientation and scale are returned as the last columns of the output.

        Returns
        ------------
        X_transformed: array-like of shape (n_samples, (1+2*n_harmonics)*n_dim)
            Returns the array-like of coefficients.
            (a_0, a_1, ..., a_n, b_0, b_1, ..., b_n, ,
            c_0, c_1, ..., c_n, d_0, d_1, ..., d_n)

        """
        n_dim = self.n_dim

        if t is None:
            t_ = [None for i in range(len(X))]
        else:
            t_ = t

        if len(t_) != len(X):
            raise ValueError(
                "t ({t_len}) must have a same length of X ({X_len})".format(
                    t_len=len(t_), X_len=len(X)
                )
            )

        if isinstance(X, pd.DataFrame):
            X_ = [
                row.dropna().to_numpy().reshape(n_dim, -1).T
                for idx, row in X.iterrows()
            ]
        else:
            X_ = X

        if n_dim == 2:
            print(self.n_jobs)
            X_transformed = np.stack(
                Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                    delayed(self._transform_single_2d)(
                        X_[i], t_[i], norm, return_orientation_scale
                    )
                    for i in range(len(X_))
                )
            )
        elif n_dim == 3:
            X_transformed = np.stack(
                Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                    delayed(self._transform_single_3d)(
                        X_[i],
                        t_[i],
                        norm=norm,
                        return_orientation_scale=return_orientation_scale,
                    )
                    for i in range(len(X_))
                )
            )

        return X_transformed

    def inverse_transform(self, X_transformed, t_num=100, norm=True, as_frame=False):
        """Inverse analysis of elliptic Fourier analysis.

        Parameters
        ----------
        X_transformed : array-like of shape (n_samples, n_features)
            Elliptic Fourier coefficients.
        t_num : int, default = 100
            Number of coordinate values.
        as_frame : bool, default = False
            If True, return pd.DataFrame.

        Returns
        -------
        X_coords : array-like of shape (n_samples, t_num, 2) or pd.DataFrame
            Coordinate values reconstructed from the elliptic Fourier coefficients.

        """
        X_list = []
        sp_num = X_transformed.shape[0]

        for i in range(sp_num):
            if as_frame:
                coef = X_transformed.loc[i]
                if self.n_dim == 2:
                    X = self._inverse_transform_single_2d(
                        coef, as_frame=as_frame, t_num=t_num, norm=norm
                    )
                    df_X = pd.DataFrame(X, columns=["x", "y"])
                elif self.n_dim == 3:
                    X = self._inverse_transform_single_3d(
                        coef, as_frame=as_frame, t_num=t_num, norm=norm
                    )
                    df_X = pd.DataFrame(X, columns=["x", "y", "z"])
                df_X["coord_id"] = [coord_id for coord_id in range(len(X))]
                df_X["specimen_id"] = i
                X_list.append(df_X)
            else:
                coef = X_transformed[i]
                if self.n_dim == 2:
                    X = self._inverse_transform_single_2d(
                        coef, as_frame=as_frame, t_num=t_num, norm=norm
                    )
                elif self.n_dim == 3:
                    X = self._inverse_transform_single_3d(
                        coef, as_frame=as_frame, t_num=t_num, norm=norm
                    )
                X_list.append(X)

        if as_frame:
            X_coords = pd.concat(X_list)
            X_coords = X_coords.set_index(["specimen_id", "coord_id"])
        else:
            X_coords = X_list

        return X_coords

    ###########################################################
    #
    #   2D
    #
    ###########################################################

    def _transform_single_2d(
        self,
        X: np.ndarray,
        t: np.ndarray | None = None,
        norm=True,
        return_orientation_scale: bool = False,
        duplicated_points="infinitesimal",
    ):
        """Fit the model with a signle outline.

        Parameters
        ----------
        X: ndarray of shape (n_coords, 2)
            Coordinate values of an 2D outline.

        t: ndarray of shape  (n_coords, ), optional
            A parameter indicating the position on the outline.
            If `t=None`, then t is calculated based on
            the coordinate values with the linear interpolation.

        Returns
        -------
        X_transformed: ndarray of shape (4*(n_harmonics+1), )
            Coefficients of Fourier series.

        """

        n_harmonics = self.n_harmonics

        X_arr = np.append(
            X[-1].reshape(1, self.n_dim), np.array(X), axis=0
        )  # 1 <= i <= k
        dx = X_arr[1:, 0] - X_arr[:-1, 0]  # 1 <= i <= k
        dy = X_arr[1:, 1] - X_arr[:-1, 1]  # 1 <= i <= k

        if t is None:
            dt = np.sqrt(dx**2 + dy**2)
            tp = np.cumsum(dt)
        else:
            t_ = np.append(0, t)
            dt = t_[1:] - t_[:-1]  # 1 <= i <= k
            tp = np.cumsum(dt)

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

        if len(tp) != len(X):
            raise ValueError(
                "len(t) must have a same len(X), len(t): "
                + str(len(tp))
                + ", len(X): "
                + str(len(X))
            )

        # Fourier series expansion
        T = tp[-1]
        a0 = 2 * np.sum(X_arr[1:, 0] * dt) / T
        c0 = 2 * np.sum(X_arr[1:, 1] * dt) / T
        an = np.append(a0, _cse(dx, dt, n_harmonics))
        bn = np.append(0, _sse(dx, dt, n_harmonics))
        cn = np.append(c0, _cse(dy, dt, n_harmonics))
        dn = np.append(0, _sse(dy, dt, n_harmonics))

        # Normalize
        if norm:
            an, bn, cn, dn, psi, scale = self._normalize_2d(an, bn, cn, dn)

        if return_orientation_scale:
            X_transformed = np.hstack([an, bn, cn, dn, psi, scale])
        else:
            X_transformed = np.hstack([an, bn, cn, dn])

        return X_transformed

    def _normalize_2d(self, an, bn, cn, dn, keep_start_point=False):
        """Normalize Fourier coefficients.

        Todo:
            - [x] 1st ellipse, major axis
            - [ ] 1st ellipse, area
            - [ ] Procrustes alignment -> in coordinate values?
        """
        a1 = an[1]
        b1 = bn[1]
        c1 = cn[1]
        d1 = dn[1]

        theta = (1 / 2) * np.arctan(
            2 * (a1 * b1 + c1 * d1) / (a1**2 + c1**2 - b1**2 - d1**2)
        )

        [[a_s, b_s], [c_s, d_s]] = np.array([[a1, b1], [c1, d1]]).dot(
            rotation_matrix_2d(theta)
        )
        s1 = a_s**2 + c_s**2
        s2 = b_s**2 + d_s**2

        if s1 < s2:
            if theta < 0:
                theta = theta + np.pi / 2
            else:
                theta = theta - np.pi / 2

        a_s = a1 * np.cos(theta) + b1 * np.sin(theta)
        c_s = c1 * np.cos(theta) + d1 * np.sin(theta)
        scale = np.sqrt(a_s**2 + c_s**2)
        psi = np.arctan2(c_s, a_s)

        if keep_start_point:
            theta = 0

        coef_norm_list = []
        r_psi = rotation_matrix_2d(-psi)
        for n in range(1, len(an)):
            r_ntheta = rotation_matrix_2d(n * theta)
            coef_orig = np.array([[an[n], bn[n]], [cn[n], dn[n]]])
            coef_norm_tmp = (1 / scale) * np.dot(np.dot(r_psi, coef_orig), r_ntheta)
            coef_norm_list.append(coef_norm_tmp.reshape(-1))

        coef_norm = np.stack(coef_norm_list)
        An = np.append(an[0], coef_norm[:, 0])
        Bn = np.append(bn[0], coef_norm[:, 1])
        Cn = np.append(cn[0], coef_norm[:, 2])
        Dn = np.append(dn[0], coef_norm[:, 3])

        return An, Bn, Cn, Dn, psi, scale

    def _inverse_transform_single_2d(
        self,
        X_transformed,
        t_num=100,
        norm=True,
        as_frame=False,
    ):
        coef = X_transformed
        if as_frame:
            a0 = coef["an"][0]
            c0 = coef["cn"][0]
            an = coef["an"][1:]
            bn = coef["bn"][1:]
            cn = coef["cn"][1:]
            dn = coef["dn"][1:]
        else:
            an, bn, cn, dn = coef.reshape([2 * self.n_dim, -1])
            a0 = an[0]
            c0 = cn[0]
            an = an[1:]
            bn = bn[1:]
            cn = cn[1:]
            dn = dn[1:]

        if norm:
            a0 = 0
            c0 = 0

        n_max = len(an)

        theta = np.linspace(2 * np.pi / t_num, 2 * np.pi, t_num)

        cos = np.cos(np.tensordot(np.arange(1, n_max + 1, 1), theta, 0))
        sin = np.sin(np.tensordot(np.arange(1, n_max + 1, 1), theta, 0))

        x = a0 / 2 + np.dot(an, cos) + np.dot(bn, sin)
        y = c0 / 2 + np.dot(cn, cos) + np.dot(dn, sin)

        X_coords = np.stack([x, y], 1)

        return X_coords

    ###########################################################
    #
    #   3D
    #
    ###########################################################

    def _transform_single_3d(
        self,
        X: np.ndarray,
        t: np.ndarray | None = None,
        norm: bool = False,
        return_orientation_scale: bool = False,
        duplicated_points: str = "infinitesimal",
    ):
        """Fit the model with a signle outline.

        Parameters
        ----------
        X: ndarray of shape (n_coords, 3)
            Coordinate values of an 3D outline.

        t: ndarray of shape (n_coords+1, ), optional
            A parameter indicating the position on the outline.
            Both t[0] and t[n_coords] corresponds to X[0].
            If `t=None`, then t is calculated based on
            the coordinate values with the linear interpolation.

        norm: bool, default=True
            Normalize the elliptic Fourier coefficients
            Note: Not implemented yet.

        Returns
        -------
        X_transformed: ndarray of shape (6*(n_harmonics+1), )
            Coefficients of Fourier series.

        ToDo
        -------
        * EHN: Normalize 3D Fourier coefficients
        """

        n_harmonics = self.n_harmonics

        X_arr = np.append(
            X[-1].reshape(1, self.n_dim), np.array(X), axis=0
        )  # 1 <= i <= k
        dx = X_arr[1:, 0] - X_arr[:-1, 0]  # 1 <= i <= k
        dy = X_arr[1:, 1] - X_arr[:-1, 1]  # 1 <= i <= k
        dz = X_arr[1:, 2] - X_arr[:-1, 2]  # 1 <= i <= k

        if t is None:
            dt = np.sqrt(dx**2 + dy**2)
            tp = np.cumsum(dt)
        else:
            t_ = np.append(0, t)
            dt = t_[1:] - t_[:-1]  # 1 <= i <= k
            tp = np.cumsum(dt)

        if duplicated_points == "infinitesimal":
            dt[dt < 10**-10] = 10**-10
        elif duplicated_points == "deletion":
            idx_duplicated_points = np.where(dt == 0)[0]
            if len(idx_duplicated_points) > 0:
                dx = np.delete(dx, idx_duplicated_points)
                dy = np.delete(dy, idx_duplicated_points)
                dz = np.delete(dz, idx_duplicated_points)

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

        if len(tp) != len(X):
            raise ValueError(
                "len(t) must have a same len(X), len(t): "
                + str(len(tp))
                + ", len(X): "
                + str(len(X))
            )

        # Fourier series expansion
        T = tp[-1]
        a0 = 2 / T * np.sum(X_arr[1:, 0])
        c0 = 2 / T * np.sum(X_arr[1:, 1])
        e0 = 2 / T * np.sum(X_arr[1:, 2])
        an = np.append(a0, _cse(dx, dt, n_harmonics))
        bn = np.append(0, _sse(dx, dt, n_harmonics))
        cn = np.append(c0, _cse(dy, dt, n_harmonics))
        dn = np.append(0, _sse(dy, dt, n_harmonics))
        en = np.append(e0, _cse(dz, dt, n_harmonics))
        fn = np.append(0, _sse(dz, dt, n_harmonics))

        # Normalize
        if norm:
            an, bn, cn, dn, en, fn = self._normalize_3d(an, bn, cn, dn, en, fn)

        X_transformed = np.hstack([an, bn, cn, dn, en, fn])

        return X_transformed

    def _normalize_3d(self, an, bn, cn, dn, en, fn):
        raise NotImplementedError("Not implemented yet")

    def _inverse_transform_single_3d(
        self,
        X_transformed,
        t_num: int = 100,
        norm=True,
        as_frame: bool = False,
    ):
        coef = X_transformed
        if as_frame:
            a0 = coef["an"][0]
            c0 = coef["cn"][0]
            e0 = coef["en"][0]
            an = coef["an"][1:]
            bn = coef["bn"][1:]
            cn = coef["cn"][1:]
            dn = coef["dn"][1:]
            en = coef["en"][1:]
            fn = coef["fn"][1:]
        else:
            an, bn, cn, dn, en, fn = coef.reshape([2 * self.n_dim, -1])
            a0 = an[0]
            c0 = cn[0]
            e0 = en[0]
            an = an[1:]
            bn = bn[1:]
            cn = cn[1:]
            dn = dn[1:]
            en = en[1:]
            fn = fn[1:]

        n_max = len(an)

        theta = np.linspace(2 * np.pi / t_num, 2 * np.pi, t_num)

        if norm:
            a0 = 0
            c0 = 0
            e0 = 0

        cos = np.cos(np.tensordot(np.arange(1, n_max + 1, 1), theta, 0))
        sin = np.sin(np.tensordot(np.arange(1, n_max + 1, 1), theta, 0))

        x = a0 / 2 + np.dot(an, cos) + np.dot(bn, sin)
        y = c0 / 2 + np.dot(cn, cos) + np.dot(dn, sin)
        z = e0 / 2 + np.dot(en, cos) + np.dot(fn, sin)

        X_coords = np.stack([x, y, z], 1)

        return X_coords

    ###########################################################
    #
    #   set_output API
    #
    ###########################################################

    def get_feature_names_out(
        self, input_features: None | npt.ArrayLike = None
    ) -> np.ndarray:
        """Get output feature names.

        Parameters
        ----------
        input_features : None | npt.ArrayLike, optional
            Input feature names, by default None

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.

        """
        an = ["a_" + str(i) for i in range(self.n_harmonics + 1)]
        bn = ["b_" + str(i) for i in range(self.n_harmonics + 1)]
        cn = ["c_" + str(i) for i in range(self.n_harmonics + 1)]
        dn = ["d_" + str(i) for i in range(self.n_harmonics + 1)]
        feature_names = an + bn + cn + dn
        if self.n_dim == 3:
            en = ["e_" + str(i) for i in range(self.n_harmonics + 1)]
            fn = ["f_" + str(i) for i in range(self.n_harmonics + 1)]
            feature_names = feature_names + en + fn
        feature_names_out = np.asarray(feature_names, dtype=str)
        return feature_names_out

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        return (self.n_harmonics + 1) * (2 * self.n_dim)


###########################################################
#
#   utility functions
#
###########################################################


def rotation_matrix_2d(theta):
    rot_mat = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return rot_mat


def _cse(dx: np.ndarray, dt: np.ndarray, n_harmonics: int) -> np.ndarray:
    """Cos series expansion n>=1

    Parameters
    ----------
    dx : np.ndarray
        differences of coordinates
    dt : np.ndarray
        differences of parameter
    n_harmonics : int
        number of harmonics

    Return
    ----------
    coef : np.ndarray
        coefficients of cos series expansion
    """

    # t = np.concatenate([[0], np.cumsum(dt)]) - dt[0]  # t_{i-1}
    # T = t[-1] + dt[0]
    t = np.concatenate([[0], np.cumsum(dt)])
    T = t[-1]

    cn = [
        (T / (2 * (np.pi**2) * (n**2)))
        * np.sum(
            (dx / dt)
            * (np.cos(2 * np.pi * n * t[1:] / T) - np.cos(2 * np.pi * n * t[:-1] / T))
        )
        for n in range(1, n_harmonics + 1, 1)
    ]

    coef = np.array(cn)

    return coef


def _sse(dx: np.ndarray, dt: np.ndarray, n_harmonics: int) -> np.ndarray:
    """Sin series expansion n>=1"""
    # t = np.concatenate([[np.sum(dt)], np.cumsum(dt)]) - dt[0]  # t_{i-1}
    # T = t[-1] + dt[0]
    t = np.concatenate([[0], np.cumsum(dt)])
    T = t[-1]

    cn = [
        (T / (2 * (np.pi**2) * (n**2)))
        * np.sum(
            (dx / dt)
            * (np.sin(2 * np.pi * n * t[1:] / T) - np.sin(2 * np.pi * n * t[:-1] / T))
        )
        for n in range(1, n_harmonics + 1, 1)
    ]

    coef = np.array(cn)

    return coef



class PositionAligner(BaseEstimator, TransformerMixin):
    """_summary_

    Parameters
    ----------
    BaseEstimator : _type_
        _description_
    TransformerMixin : _type_
        _description_
    """

    def __init__(self, approx="points", method="centroid"):
        self.approx = approx
        self.method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return align_position(X, method=self.method, origin=self.origin)

    def fit_transform(self, X, y=None):
        return align_position(X, method=self.method, origin=self.origin)

    def _align_centroid(
        self,
    ):
        pass


class OrientationAligner(BaseEstimator, TransformerMixin):
    def __init__(self, approx="points", method="pca"):
        self.method = method


class ScaleAligner(BaseEstimator, TransformerMixin):
    def __init__(self, approx="points", method="area"):
        self.method = method


class ProcrustesAligner(BaseEstimator, TransformerMixin):
    def __init__(self, scale=True):
        self.method = method


def _align_position(x, p0, method="centroid_points", origin=None):
    """Align positions of outline coordinate values."""

    if method == "centroid_points":
        X_aligned = X - np.mean(X, axis=0)
    elif method == "centroid_polygon":
        raise NotImplementedError("Not implemented yet")
    elif method == "centroid_closed_spline":
        n_dim = X[0].shape[1]
        for x in X:
            dx = x[1:] - x[:-1]
            dt = np.sqrt(dx * dx)
            t = np.concatenate([[0], np.cumsum(dt)])
            spl = make_interp_spline(t, np.c_[[x[:, i] for i in range(n_dim)]], k=3)

        X_aligned = X - np.mean(X, axis=1)[:, None]
    elif method == "centroid_minimal_surface":
        raise NotImplementedError("Not implemented yet")
    else:
        raise ValueError("Unknown method: {}".format(method))

    return X_aligned


def _align_orientation(x1, R0, method="pca"):
    """Align orientation of outline coordinate values."""

    if method == "pca":
        n_dim = X[0].shape[1]
        pca = PCA(n_components=n_dim)
        X_aligned = [pca.fit_transpose(x) for x in X]
    else:
        raise ValueError("Unknown method: {}".format(method))

    return X_aligned


def _align_scale(x, s, method="area"):
    """Align scale of outline coordinate values.
    Parameters
    ----------
    X : np.ndarray
        outline coordinate values
    method : str, optional
        method to align scale, by default "area"
    """

    if method == "area":
        X_aligned = [x / np.sqrt(np.sum(x**2)) for x in X]
    else:
        raise ValueError("Unknown method: {}".format(method))

    return X_aligned
