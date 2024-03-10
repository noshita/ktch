import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from ktch.datasets import load_landmark_mosquito_wings
from ktch.landmark import GeneralizedProcrustesAnalysis, centroid_size

data_landmark_mosquito_wings = load_landmark_mosquito_wings(as_frame=True)
data_landmark_mosquito_wings.coords

X = data_landmark_mosquito_wings.coords.to_numpy().reshape(-1, 18 * 2)


def test_gpa_shape():
    gpa = GeneralizedProcrustesAnalysis()
    gpa.fit_transform(X)
    X_transformed = gpa.fit_transform(X)

    assert X.shape == X_transformed.shape


@pytest.mark.parametrize("n_dim", [2, 3])
def test_centroid_size(n_dim):
    x = np.random.uniform(0, 100, (10, n_dim))
    cs_r = np.sqrt(np.sum((x - x.mean(axis=0)) ** 2))
    cs_t = centroid_size(x)

    assert_array_almost_equal(cs_r, cs_t)
