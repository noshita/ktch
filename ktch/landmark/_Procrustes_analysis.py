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

import numpy as np
import scipy as sp

from sklearn.base import BaseEstimator, TransformerMixin


class GeneralizedProcrustesAnalysis(TransformerMixin, BaseEstimator):
    r"""
    Generalized Procrustes Analysis (GPA)

    Parameters
    ------------
    tol: float, default=10**-10
        Torelance for convergence of Procrustes analysis.

    Notes
    ------------
    GPA involves translating, rotating, and scaling  the configurations to each other to minimize the sum of the squared distances with respect to positional, rotational, and size parameters, subject to a size constraint [Gower_1975]_, [Goodall_1991]_.

    References
    ------------
    .. [Gower_1975] Gower, J.C., 1975. Generalized procrustes analysis. Psychometrika 40, 33–51.
    .. [Goodall_1991] Goodall, C., 1991. Procrustes Methods in the Statistical Analysis of Shape. J Royal Statistical Soc Ser B Methodol 53, 285–321.

    """

    def __init__(self, tol=10**-10):
        self.tol = tol

    def fit(self, X):
        Gamma = 0

        return self

    def transform(self, X):
        X_ = X
        X_ = self._translate(X_)
        X_ = self._scale(X_)
        mu = np.sum(X_, axis=0) / len(X_)

        diff_Procrustes_dist = 10 * self.tol
        total_Procrustes_dist_prev = 10**10
        while diff_Procrustes_dist > self.tol:
            results = [sp.spatial.procrustes(mu, x) for x in X]
            X_ = np.array([result[1] for result in results])
            total_Procrustes_dist = np.sum(np.array([result[2] for result in results]))
            diff_Procrustes_dist = total_Procrustes_dist_prev - total_Procrustes_dist
            total_Procrustes_dist_prev = total_Procrustes_dist
            mu = np.sum(X_, axis=0) / len(X_)

        return X_

    def fit_transform(self, X):
        return self.transform(X)

    def _translate(self, X):
        X_translated = np.array([x - np.mean(x, axis=0) for x in X])
        return X_translated

    def _scale(self, X):
        X_scaled = np.array(
            [x / np.sqrt(np.trace(np.dot(x, x.transpose()))) for x in X]
        )
        return X_scaled


class LandmarkImputer(TransformerMixin, BaseEstimator):
    def __init__(self, missing_values=np.nan):
        pass
