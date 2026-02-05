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

from ._kernels import tps_coefficients, tps_warp

###########################################################
#
#   utility functions
#
###########################################################


def _thin_plate_spline_2d(x_reference, x_target):
    """Thin-plate spline in 2D.

    This is a backward-compatible wrapper around tps_coefficients().

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

    See Also
    --------
    ktch.landmark._kernels.tps_coefficients : The underlying implementation.
    """
    return tps_coefficients(x_reference, x_target)


def _tps_2d(x, y, T, W, c, A):
    """Apply TPS transformation to a single 2D point.

    This is a backward-compatible wrapper around tps_warp().

    Parameters
    ----------
    x, y : float
        Coordinates of the point to warp.
    T : ndarray, shape (n_landmarks, 2)
        Source landmarks (control points).
    W : ndarray, shape (n_landmarks, 2)
        Non-affine coefficients.
    c : ndarray, shape (2,)
        Translation coefficients.
    A : ndarray, shape (2, 2)
        Affine transformation matrix.

    Returns
    -------
    ndarray, shape (2,)
        Warped point coordinates.

    See Also
    --------
    ktch.landmark._kernels.tps_warp : The underlying implementation.
    """
    point = np.array([[x, y]])
    warped = tps_warp(point, T, W, c, A)
    return warped[0]
