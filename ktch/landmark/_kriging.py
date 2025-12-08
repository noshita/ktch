"""Kriging"""

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
import scipy as sp
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin

###########################################################
#
#   utility functions
#
###########################################################


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
