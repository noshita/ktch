"""Base IO code for small sample datasets"""

# Copyright 2023 Koji Noshita
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
import pandas as pd

from sklearn.utils import check_random_state


def make_landmarks_from_reference(
    reference,
    n_samples=30,
    sd=1.0,
    random_state=None,
    allow_collinearity=False,
    allow_dup=False,
    as_frame=False,
):
    """Generate landmark dataset around a reference configuration.

    Parameters
    ----------
    reference : array-like, shape (n_landmarks, n_dim)
        Reference configuration.
    n_samples : int, default=30
        Number of samples to generate.
    sd : float, default=1.0
        Standard deviation of the Gaussian noise added to the reference.
    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible output across multiple function calls.
    allow_dup : bool, default=False
        If True, allow duplicate configurations in the generated dataset.
    as_frame : bool, default=False
        If True, return a pandas DataFrame.

    Returns
    -------
    X : array-like, shape (n_samples, n_landmarks, n_dim)
        Generated landmark dataset.
    """

    ref = np.asarray(reference)

    generator = check_random_state(random_state)

    rand_size = (n_samples,) + ref.shape
    X = ref + sd * generator.standard_normal(rand_size)

    num = 0
    while n_samples > num:
        rand_size = (n_samples - X.shape[0],) + ref.shape
        X_comp = ref + sd * generator.standard_normal(rand_size)
        X = np.concatenate([X, X_comp])
        if not allow_dup:
            X = _remove_duplicated_configurations(X)
            num = X.shape[0]

    if as_frame:
        n_landmarks = X.shape[1]
        n_dim = X.shape[2]
        X = pd.DataFrame(X.reshape(n_samples * n_landmarks, n_dim))
        X["id"] = [i for i in range(n_samples) for _ in range(n_landmarks)]
        X["coord_id"] = [j for _ in range(n_samples) for j in range(n_landmarks)]
        X = X.set_index(["id", "coord_id"])
        if n_dim == 2:
            X.columns = ["x", "y"]
        elif n_dim == 3:
            X.columns = ["x", "y", "z"]
        else:
            raise ValueError("reference must be 2D or 3D.")

    return X


def _remove_duplicated_configurations(X):
    """Remove duplicated configurations in a dataset."""

    X_flatten = np.unique(X.reshape(X.shape[0], -1), axis=0)
    X_new = X_flatten.reshape(X.shape)

    return X_new
