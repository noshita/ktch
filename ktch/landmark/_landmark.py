"""Landmark-based morphometrics"""

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

from sklearn.base import BaseEstimator, TransformerMixin


class PositionAligner(TransformerMixin, BaseEstimator):
    """Align landmark configurations by centroid position.

    Placeholder; not implemented yet.
    """

    def __init__(self, copy=True):
        self.copy = copy

    def transform(self, X, reference_point=None):
        raise NotImplementedError("PositionAligner is not implemented yet.")


class SizeScaler(TransformerMixin, BaseEstimator):
    """Scale landmark configurations by centroid size.

    Placeholder; not implemented yet.
    """

    def __init__(self):
        pass

    def transform(self, X):
        raise NotImplementedError("SizeScaler is not implemented yet.")


class OrientationAligner(TransformerMixin, BaseEstimator):
    """Align landmark configurations by orientation.

    Placeholder; not implemented yet.
    """

    def __init__(self):
        pass

    def transform(self, X):
        raise NotImplementedError("OrientationAligner is not implemented yet.")


class ProcrustesSuperImposer(TransformerMixin, BaseEstimator):
    """Procrustes superimposition.

    Placeholder; not implemented yet. Use
    :class:`ktch.landmark.GeneralizedProcrustesAnalysis` instead.
    """

    def __init__(self):
        pass

    def transform(self, X):
        raise NotImplementedError(
            "ProcrustesSuperImposer is not implemented yet; "
            "use GeneralizedProcrustesAnalysis."
        )
