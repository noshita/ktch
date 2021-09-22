"""Landmark-based Morphometrics"""

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
	"""Align the coordinate values .
	This estimator scales and translates each feature individually such
	that it is in the given range on the training set, e.g. between
	zero and one.
	The transformation is given by::
	    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
	    X_scaled = X_std * (max - min) + min
	where min, max = feature_range.
	This transformation is often used as an alternative to zero mean,
	unit variance scaling.
	Read more in the :ref:`User Guide <preprocessing_scaler>`.
	Parameters
	----------
	copy : bool, default=True
	    Set to False to perform inplace row normalization and avoid a
	    copy (if the input is already a numpy array).
	Attributes
	----------
	data_range_ : ndarray of shape (n_features,)
	    Per feature range ``(data_max_ - data_min_)`` seen in the data
	    .. versionadded:: 0.17
	       *data_range_*
	n_samples_seen_ : int
	    The number of samples processed by the estimator.
	    It will be reset on new calls to fit, but increments across
	    ``partial_fit`` calls.
	Examples
	--------
	>>> from sklearn.preprocessing import MinMaxScaler
	>>> data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
	>>> scaler = MinMaxScaler()
	>>> print(scaler.fit(data))
	MinMaxScaler()
	>>> print(scaler.data_max_)
	[ 1. 18.]
	>>> print(scaler.transform(data))
	[[0.   0.  ]
	 [0.25 0.25]
	 [0.5  0.5 ]
	 [1.   1.  ]]
	>>> print(scaler.transform([[2, 2]]))
	[[1.5 0. ]]
	See Also
	--------
	minmax_scale : Equivalent function without the estimator API.
	Notes
	-----
	NaNs are treated as missing values: disregarded in fit, and maintained in
	transform.
	For a comparison of the different scalers, transformers, and normalizers,
	see :ref:`examples/preprocessing/plot_all_scaling.py
	<sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
	"""
	def __init__(self, copy=True):
		self.copy = copy

	def transform(self, X, reference_point=None):
	"""Perform alignment by centroid position
	Parameters
	----------
	X : {array-like, dense matrix of shape (n_samples, n_landmarks, n_dims)
	    The coordinate values to be aligned
	reference_point: {array-like}
 		The coordinate value of the reference point to be aligned onto the origin. 
 		If `reference_point` is `None`, the centroid computed from `X` is used as the reference point. 
	Returns
	-------
	X_tr : {ndarray, sparse matrix} of shape (n_samples, n_landmarks,n_dims)
	    Transformed array, which is removed the information about the position.
	"""
	check_is_fitted(self)

	n_samples, _, n_dims = X.shape
	X_tr = X - X.mean(axis=1).reshape(n_samples,1,n_dims)

	return X_tr

class SizeScaler(TransformerMixin, BaseEstimator):
	def __init__(self):
		self.copy = copy

	def transform(self, X):
	"""Perform scaling by centroid size
	Parameters
	----------
	X : {array-like, dense matrix of shape (n_samples, n_landmarks, n_dims)
	    The coordinate values used to align by the centroid position onto the origin.
	Returns
	-------
	X_tr : {ndarray, sparse matrix} of shape (n_samples, n_landmarks,n_dims)
	    Transformed array, which is removed the information about the position.
	"""


class OrientationAligner(TransformerMixin, BaseEstimator):
	def __init__(self):


class ProcrustesSuperImposer(TransformerMixin, BaseEstimator):
	def __init__(self):

