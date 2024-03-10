"""Procrustes Analysis"""

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

import numpy as np
import numpy.typing as npt
import scipy as sp
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin


class GeneralizedProcrustesAnalysis(
    OneToOneFeatureMixin,
    TransformerMixin,
    BaseEstimator,
    metaclass=ABCMeta,
):
    r"""
    Generalized Procrustes Analysis (GPA)

    Parameters
    ------------
    tol: float, default=10^-7
        Torelance for convergence of Procrustes analysis.
    n_dim: int, default=2
        Dimensions of the configurations.

    Attributes
    ------------
    mu_: ndarray, shape (n_landmarks, n_dim)
        The mean shape of the aligned shapes.
    n_dim_: int, 2 or 3
        Dimensions of the configurations.

    Notes
    ------------
    GPA for shape involves translating, rotating, and scaling  the configurations to each other to minimize the sum of the squared distances with respect to positional, rotational, and size parameters, subject to a size constraint [Gower_1975]_, [Goodall_1991]_.

    GPA for size-and-shape

    References
    ------------
    .. [Gower_1975] Gower, J.C., 1975. Generalized procrustes analysis. Psychometrika 40, 33–51.
    .. [Goodall_1991] Goodall, C., 1991. Procrustes Methods in the Statistical Analysis of Shape. J Royal Statistical Soc Ser B Methodol 53, 285–321.

    """

    def __init__(self, tol=10**-7, scaling=True, n_dim=2, debug=False):
        self.tol = tol
        self.scaling = scaling
        self.debug = debug
        self.mu_ = None
        self.n_dim_ = n_dim

    @property
    def n_dim(self):
        return self.n_dim_

    @n_dim.setter
    def n_dim(self, n_dim):
        if n_dim not in [2, 3]:
            raise ValueError("n_dim must be 2 or 3.")
        self.n_dim_ = n_dim

    def fit(self, X):
        """Fit GPA.

        Parameters
        ----------
        X : array-like, shape (n_specimens, n_landmarks, n_dim)
            /DataFrame, shape (n_specimens, n_landmarks * n_dim)
            Configurations to be aligned.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns

        return self

    def transform(
        self,
        X: npt.ArrayLike,
    ) -> npt.ArrayLike:
        """GPA for shapes/size-and-shapes.

        Parameters
        ----------
        X : array-like, shape (n_specimens, n_landmarks*n_dim)
            Configurations to be aligned.

        scaling: bool, default=True
            If True, the configurations are aligned by translation, rotation, and scaling.
            If False, the configurations are aligned by translation and rotation.

        Returns
        -------
        X_ : ndarray, shape (n_specimens, n_landmarks, n_dim)
            Shapes/Size-and-Shape (aligned configurations)
        """

        X_ = np.array(X)
        n_specimen = len(X_)
        n_dim = self.n_dim_
        if len(X_[0]) % n_dim != 0:
            raise ValueError("X must be n_specimens x n_landmarks*n_dim.")
        n_landmarks = int(len(X_[0]) / n_dim)
        X_ = X_.reshape(n_specimen, n_landmarks, n_dim)

        scaling = self.scaling
        if scaling:
            X_ = self._transform_shape(X_)
        else:
            X_ = self._transform_size_and_shape(X_)

        return X_.reshape(n_specimen, n_landmarks * n_dim)

    def _transform_shape(self, X):
        X_ = np.array(X, dtype=np.double, copy=True)
        X_ = self._center(X_)
        mu = np.sum(X_, axis=0) / len(X_)
        mu = mu / centroid_size(mu)

        diff_disp = np.inf
        total_disp_prev = np.inf
        while diff_disp > self.tol:
            results = [sp.spatial.procrustes(mu, x) for x in X]
            X_ = np.array([x_aligned for _, x_aligned, _ in results])
            total_disp = np.sum(np.array([disp for _, _, disp in results]))
            diff_disp = np.abs(total_disp_prev - total_disp)
            total_disp_prev = total_disp
            mu = np.sum(X_, axis=0) / len(X_)
            mu = mu / centroid_size(mu)
            if self.debug:
                print("total_disp: ", total_disp, "diff_disp: ", diff_disp)

        self.mu_ = mu

        return X_

    def _transform_size_and_shape(self, X):
        X_ = np.array(X, dtype=np.double, copy=True)
        X_ = self._center(X_)
        mu = np.sum(X_, axis=0) / len(X_)

        diff_disp = np.inf
        total_disp_prev = np.inf
        while diff_disp > self.tol:
            total_disp = 0
            X_new = np.zeros(X_.shape)
            for i, x in enumerate(X):
                R, scale = sp.linalg.orthogonal_procrustes(x, mu)
                x_ = x @ R
                total_disp += np.sum((x_ - mu) ** 2)
                X_new[i] = x_
            X_ = X_new

            diff_disp = np.abs(total_disp_prev - total_disp)
            total_disp_prev = total_disp
            mu = np.sum(X_, axis=0) / len(X_)

            if self.debug:
                print("total_disp: ", total_disp, "diff_disp: ", diff_disp)

        self.mu_ = mu

        return X_.reshape(self.n_specimen_, self.n_landmarks_ * self.n_dim_)

    def fit_transform(self, X):
        """GPA for shapes/size-and-shapes.

        Parameters
        ----------
        X : array-like, shape (n_specimens, n_landmarks * n_dim)
            /DataFrame, shape (n_specimens, n_landmarks * n_dim)
            Configurations to be aligned.

        Returns
        -------
        X_ : ndarray, shape (n_specimens, n_landmarks, n_dim)
            Shapes/Size-and-Shape (aligned configurations)

        """
        self.fit(X)
        X_ = self.transform(X)
        return X_

    def _center(self, X):
        X_centered = np.array([x - np.mean(x, axis=0) for x in X])
        return X_centered

    def _scale(self, X):
        X_scaled = np.array([x / centroid_size(x) for x in X])
        return X_scaled

    # @property
    # def _n_features_out(self):
    #     """Number of transformed output features."""
    #     return self.n_landmarks_ * self.n_dim_


class LandmarkImputer(TransformerMixin, BaseEstimator):
    def __init__(self, missing_values=np.nan):
        pass


def centroid_size(x):
    """Calculate the centroid size.

    Parameters
    ----------
    x : array-like, shape (n_landmarks, n_dim)
        Configuration, pre-shape, shape, etc.

    Returns
    -------
    centroid_size : float
        Centroid size of the input.
    """
    x = np.array(x)
    x_c = x - np.mean(x, axis=0)
    centroid_size = np.sqrt(np.trace(np.dot(x_c, x_c.T)))
    return centroid_size


def _thin_plate_spline_2d(x_reference, x_target):
    """Thin-plate spline in 2D.

    Parameters
    ----------
    x_reference : array-like, shape (n_landmarks, n_dim)
        Reference configuration.
    x_target : array-like, shape (n_landmarks, n_dim)
        Target configuration.

    Returns
    -------
    W : ndarray, shape (n_landmarks, n_dim)
    c : ndarray, shape (n_dim)
    A : ndarray, shape (n_dim, n_dim)
    """

    n_dim = 2

    x_r = np.array(x_reference).reshape(-1, n_dim)
    x_t = np.array(x_target).reshape(-1, n_dim)

    n_landmarks = x_r.shape[0]

    if not x_r.shape == x_t.shape:
        raise ValueError("x_reference and x_target must have the same shape.")

    r = sp.spatial.distance.cdist(x_r, x_r, "euclidean")

    SMat = r**2 * np.log(r, out=np.zeros_like(r), where=(r != 0))
    QMat = np.concatenate([np.ones(n_landmarks).reshape(-1, 1), x_r], 1)
    zero_mat = np.zeros([n_dim + 1, n_dim + 1])

    GammaMat = np.concatenate(
        [
            np.concatenate([SMat, QMat], 1),
            np.concatenate([QMat.T, zero_mat], 1),
        ]
    )
    GammaInvMat = np.linalg.inv(GammaMat)
    sol = np.dot(GammaInvMat, np.concatenate([x_t, np.zeros([n_dim + 1, n_dim])], 0))

    W = sol[0:n_landmarks, :]
    c = sol[n_landmarks, :]
    A = sol[n_landmarks + 1 :, :]

    return W, c, A


def _tps_2d(x, y, T, W, c, A):

    t = np.array([x, y])

    r = np.apply_along_axis(lambda v: np.sqrt(np.dot(v, v)), 1, t - T)

    pred = c + np.dot(A, t) + np.dot(W.T, np.where(r == 0, 0, r**2 * np.log(r)))

    return pred
