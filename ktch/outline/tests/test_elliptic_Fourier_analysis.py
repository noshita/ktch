import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from ktch.datasets import load_coefficient_bottles, load_outline_bottles
from ktch.outline import EllipticFourierAnalysis

bottles = load_outline_bottles()
bottles_frame = load_outline_bottles(as_frame=True)


def test_data_prep_length():
    X_frame = bottles_frame.coords.unstack().sort_index(axis=1)
    X_list = bottles.coords

    assert len(X_frame) == len(X_list)


def test_data_prep_align():
    X_frame = bottles_frame.coords.unstack().sort_index(axis=1)
    X_list = bottles.coords

    for i in range(len(X_frame)):
        x_frame = X_frame.iloc[i].dropna().to_numpy().reshape(2, -1).T
        x_list = np.array(X_list[i])
        assert_array_almost_equal(x_frame, x_list)


@pytest.mark.parametrize("norm", [False, True])
def test_efa_shape(norm):
    n_harmonics = 6

    X = bottles.coords
    efa = EllipticFourierAnalysis(n_harmonics=n_harmonics)
    X_transformed = efa.fit_transform(X, norm=norm)

    assert X_transformed.shape == (len(X), 4 * (n_harmonics + 1))


@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("set_output", [None, "pandas"])
def test_efa(norm, set_output):
    n_harmonics = 6

    bottles_coef = load_coefficient_bottles(norm=norm)

    if set_output == "pandas":
        X = bottles_frame.coords.unstack().sort_index(axis=1)
    else:
        X = bottles.coords

    efa = EllipticFourierAnalysis(n_harmonics=n_harmonics)
    efa.set_output(transform=set_output)
    X_transformed = efa.fit_transform(X, norm=norm)

    if set_output == "pandas":
        coef = (
            X_transformed.to_numpy()
            .reshape(-1, 4, n_harmonics + 1)[:, :, 1:]
            .reshape(-1, n_harmonics * 4)
        )
        coef_val = bottles_coef.coef
    else:
        coef = X_transformed.reshape(-1, 4, n_harmonics + 1)[:, :, 1:].reshape(
            -1, n_harmonics * 4
        )
        coef_val = bottles_coef.coef

    assert_array_almost_equal(coef, coef_val)
