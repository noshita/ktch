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


class OrdinaryProcrustesAnalysis(TransformerMixin, BaseEstimator):
	"""Ordinary Procrustes Analysis

	"""
	def __init__(
		self, dtype = np.float64,
		scale = True,
		reflect = False,
		metric = "",
		impute = False
		):
		self.dtype = dtype

	def fit(self, X):
		Gamma = 0


		return self

	def transform(self, X):

		return X_transformed


	def fit_transform(self, X):
		pass

		


class GPA(TransformerMixin, BaseEstimator):
	def __init__(self):
		pass
		

class LandmarkImputer(TransformerMixin, BaseEstimator):
	def __init__(self, missing_values=np.nan):
		pass