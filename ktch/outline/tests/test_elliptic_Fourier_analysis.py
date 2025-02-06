import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpl_toolkits.mplot3d import axes3d
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
def test_transform(norm, set_output):
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


def test_transform_exact():
    n_harmonics = 6
    t_num = 360

    t = np.linspace(2 * np.pi / t_num, 2 * np.pi, t_num)

    a0, c0 = np.random.rand(2)
    an, bn, cn, dn = np.random.rand(4, n_harmonics)
    coef_exact = np.array([a0, *an, 0, *bn, c0, *cn, 0, *dn]).reshape(
        4, n_harmonics + 1
    )

    cos = np.cos(np.tensordot(np.arange(1, n_harmonics + 1, 1), t, 0))
    sin = np.sin(np.tensordot(np.arange(1, n_harmonics + 1, 1), t, 0))

    x = a0 / 2 + np.dot(an, cos) + np.dot(bn, sin)
    y = c0 / 2 + np.dot(cn, cos) + np.dot(dn, sin)
    X_coords = np.stack([x, y], 1)

    # fig, ax = plt.subplots()
    # ax.plot(X_coords[:, 0], X_coords[:, 1])
    # fig.savefig("X_exact.png")

    efa = EllipticFourierAnalysis(n_harmonics=n_harmonics)
    coef_est = efa.fit_transform([X_coords], t=[t], norm=False)[0]
    coef_est = coef_est.reshape(4, n_harmonics + 1)

    # Ignore a0, c0 (and b0, d0)
    # due to the sampling rate for calculating the mean coordinate
    coef_exact = coef_exact[:, 1:]
    coef_est = coef_est[:, 1:]

    assert_array_almost_equal(coef_exact, coef_est, decimal=3)


def test_transform_3d_exact():
    n_harmonics = 6
    t_num = 360

    t = np.linspace(2 * np.pi / t_num, 2 * np.pi, t_num)

    a0, c0, e0 = np.random.rand(3)
    an, bn, cn, dn, en, fn = np.random.rand(6, n_harmonics)
    coef_exact = np.array([a0, *an, 0, *bn, c0, *cn, 0, *dn, e0, *en, 0, *fn]).reshape(
        6, n_harmonics + 1
    )

    cos = np.cos(np.tensordot(np.arange(1, n_harmonics + 1, 1), t, 0))
    sin = np.sin(np.tensordot(np.arange(1, n_harmonics + 1, 1), t, 0))

    x = a0 / 2 + np.dot(an, cos) + np.dot(bn, sin)
    y = c0 / 2 + np.dot(cn, cos) + np.dot(dn, sin)
    z = e0 / 2 + np.dot(en, cos) + np.dot(fn, sin)
    X_coords = np.stack([x, y, z], 1)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.plot(X_coords[:, 0], X_coords[:, 1], X_coords[:, 2])
    # fig.savefig("X_3d_exact.png")

    efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics)
    coef_est = efa.fit_transform([X_coords], t=[t], norm=False)[0]
    coef_est = coef_est.reshape(6, n_harmonics + 1)

    # Ignore a0, c0, e0 (and b0, d0, f0)
    # due to the sampling rate for calculating the mean coordinate
    coef_exact = coef_exact[:, 1:]
    coef_est = coef_est[:, 1:]

    assert_array_almost_equal(coef_exact, coef_est, decimal=3)


def test_inverse_transform():
    n_harmonics = 6
    t_num = 360

    X = bottles.coords
    efa = EllipticFourierAnalysis(n_harmonics=n_harmonics)
    X_transformed = efa.fit_transform(X, norm=True)
    X_adj = np.array(efa.inverse_transform(X_transformed, t_num=t_num))
    T = [np.linspace(2 * np.pi / t_num, 2 * np.pi, t_num) for i in range(len(X))]

    print(X_adj.shape, T[0].shape)

    X_transformed = efa.fit_transform(
        X_adj,
        t=T,
        norm=False,
    )
    X_reconstructed = np.array(
        efa.inverse_transform(X_transformed, t_num=t_num, norm=False)
    )

    # fig, ax = plt.subplots()
    # ax.plot(X_adj[0][:, 0], X_adj[0][:, 1])
    # ax.plot(X_reconstructed[0][:, 0], X_reconstructed[0][:, 1])
    # fig.savefig("X_inv.png")

    assert_array_almost_equal(X_adj, X_reconstructed, decimal=4)
