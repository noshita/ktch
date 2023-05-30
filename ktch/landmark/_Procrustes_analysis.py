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

from abc import ABCMeta, abstractmethod

import numpy as np
import scipy as sp

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    ClassNamePrefixFeaturesOutMixin,
)


class GeneralizedProcrustesAnalysis(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator, metaclass=ABCMeta
):
    r"""
    Generalized Procrustes Analysis (GPA)

    Parameters
    ------------
    tol: float, default=10^-7
        Torelance for convergence of Procrustes analysis.

    Attributes
    ------------
    mu_: ndarray, shape (n_landmarks, n_dim)
        The mean shape of the aligned shapes.

    Notes
    ------------
    GPA for shape involves translating, rotating, and scaling  the configurations to each other to minimize the sum of the squared distances with respect to positional, rotational, and size parameters, subject to a size constraint [Gower_1975]_, [Goodall_1991]_.

    GPA for size-and-shape

    References
    ------------
    .. [Gower_1975] Gower, J.C., 1975. Generalized procrustes analysis. Psychometrika 40, 33–51.
    .. [Goodall_1991] Goodall, C., 1991. Procrustes Methods in the Statistical Analysis of Shape. J Royal Statistical Soc Ser B Methodol 53, 285–321.

    """

    def __init__(self, tol=10**-7, scaling=True, debug=False):
        self.tol = tol
        self.scaling = scaling
        self.debug = debug
        self.mu_ = None

    def fit(self, X):
        return self

    def transform(self, X):
        """GPA for shapes/size-and-shapes.

        Parameters
        ----------
        X : array-like, shape (n_shapes, n_landmarks, n_dim)
            Configurations to be aligned.

        scaling: bool, default=True
            If True, the configurations are aligned by translation, rotation, and scaling.
            If False, the configurations are aligned by translation and rotation.

        Returns
        -------
        X_ : ndarray, shape (n_shapes, n_landmarks, n_dim)
            Shapes/Size-and-Shape (aligned configurations)
        """

        scaling = self.scaling

        if scaling:
            X_ = self._transform_shape(X)
        else:
            X_ = self._transform_size_and_shape(X)

        return X_

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

        return X_

    def fit_transform(self, X):
        return self.transform(X)

    def _center(self, X):
        X_centered = np.array([x - np.mean(x, axis=0) for x in X])
        return X_centered

    def _scale(self, X):
        X_scaled = np.array([x / centroid_size(x) for x in X])
        return X_scaled


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
