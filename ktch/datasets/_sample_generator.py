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

from sklearn.utils import check_random_state


def make_landmarks_from_reference(reference, n_samples=100, sd=1.0, random_state=None):
    """Generate landmark dataset around a reference configuration.

    Parameters
    ----------
    reference : array-like, shape (n_landmarks, n_dim)
        Reference configuration.
    n_samples : int, default=100
        Number of samples to generate.
    sd : float, default=1.0
        Standard deviation of the Gaussian noise added to the reference.

    Returns
    -------
    X : array-like, shape (n_samples, n_landmarks, n_dim)
        Generated landmark dataset.
    """

    generator = check_random_state(random_state)

    rand_size = (n_samples,) + reference.shape
    X = reference + sd * generator.standard_normal(rand_size)

    return X
