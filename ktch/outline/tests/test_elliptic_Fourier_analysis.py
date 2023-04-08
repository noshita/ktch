import numpy as np

from numpy.testing import assert_array_almost_equal
import pytest

from ktch.datasets import load_outline_bottles, load_coefficient_bottles
from ktch.outline import EllipticFourierAnalysis

bottles = load_outline_bottles()


@pytest.mark.parametrize("norm", [False, True])
def test_efa(norm):
    n_harmonics = 6

    bottles_coef = load_coefficient_bottles(norm=norm)

    X = bottles.coords

    efa = EllipticFourierAnalysis(n_harmonics=n_harmonics)
    X_transformed = efa.fit_transform(X, norm=norm)

    coef = X_transformed.reshape(-1, 4, n_harmonics + 1)[:, :, 1:].reshape(
        -1, n_harmonics * 4
    )
    coef_val = bottles_coef.coef

    assert_array_almost_equal(coef, coef_val)
