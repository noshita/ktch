"""Procrustes Analysis"""

# Author: Koji Noshita <noshita@morphometrics.jp>
# License: ISC

import numpy as np
import scipy as sp

class OPA(TransformerMixin, BaseEstimator):
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

		


class GPA(TransformerMixin, BaseEstimator):
	def __init__(self):
		pass
		

class LandmarkImputer(TransformerMixin, BaseEstimator):
	def __init__(self, missing_values=np.nan):
		pass