from numpy.testing import assert_array_almost_equal
import pytest

from ktch.datasets import load_outline_bottles, load_coefficient_bottles
from ktch.outline import EllipticFourierAnalysis

bottles = load_outline_bottles()
bottles_coef = load_coefficient_bottles(norm=False)

# @pytest.mark.parameterize("norm", [False, True])
def test_efa():
    n_harmonics = 6
    X = [bottles.coords.loc[i].to_numpy() for i in range(1, 41, 1)]
    efa = EllipticFourierAnalysis(n_harmonics=n_harmonics)

    X_transformed = efa.fit_transform(X)
    coef = [X_transformed.loc[i].loc[1:].to_numpy() for i in range(40)]
    coef_val = [bottles_coef.coef.loc[i].to_numpy() for i in range(1, 41, 1)]

    assert_array_almost_equal(coef, coef_val)
