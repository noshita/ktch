import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal

from ktch.harmonic import SphericalHarmonicAnalysis, spharm, xyz2spherical
from ktch.harmonic._spherical_harmonic_analysis import cvt_spharm_coef_to_list


def test_xyz2spherical_axes():
    sqrt_half = np.sqrt(0.5)
    coords = np.array(
        [
            [sqrt_half, sqrt_half, 0.0],
            [sqrt_half, -sqrt_half, 0.0],
            [0.0, sqrt_half, sqrt_half],
            [0.0, sqrt_half, -sqrt_half],
        ]
    )

    theta_phi = xyz2spherical(coords)

    expected = np.array(
        [
            [np.pi / 2, np.pi / 4],
            [np.pi / 2, -np.pi / 4],
            [np.pi / 4, np.pi / 2],
            [3 * np.pi / 4, np.pi / 2],
        ]
    )
    assert_allclose(theta_phi, expected)


def test_cvt_spharm_coef_to_list_mapping():
    l_max = 2
    n_terms = (l_max + 1) ** 2
    coef_matrix = np.arange(n_terms * 3).reshape(n_terms, 3)

    coef_list = cvt_spharm_coef_to_list(coef_matrix)

    assert len(coef_list) == l_max + 1
    assert coef_list[0].shape == (1, 3)
    assert coef_list[1].shape == (3, 3)
    assert coef_list[2].shape == (5, 3)

    expected_l1 = coef_matrix[1:4]
    expected_l2 = coef_matrix[4:9]

    assert_array_almost_equal(coef_list[0][0], coef_matrix[0])
    assert_array_almost_equal(coef_list[1], expected_l1)
    assert_array_almost_equal(coef_list[2], expected_l2)


def test_transform_and_inverse_roundtrip():
    l_max = 2
    theta_range = np.linspace(0, np.pi, 5)
    phi_range = np.linspace(0, 2 * np.pi, 9)

    coef_list = [
        np.array([[1.0, 0.2, -0.1]]),
        np.zeros((3, 3)),
        np.zeros((5, 3)),
    ]

    x, y, z = spharm(l_max, coef_list, theta_range=theta_range, phi_range=phi_range)
    coords = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    theta_grid, phi_grid = np.meshgrid(theta_range, phi_range, indexing="ij")
    theta_phi = np.stack([theta_grid.ravel(), phi_grid.ravel()], axis=-1)

    sha = SphericalHarmonicAnalysis(n_harmonics=l_max, n_jobs=1)
    transformed = sha.fit_transform([coords], [theta_phi])

    assert transformed.shape == (1, 3 * (l_max + 1) ** 2)

    coef_matrix = transformed.reshape(3, -1).T
    assert_array_almost_equal(coef_matrix[0], coef_list[0][0], decimal=6)

    reconstructed = sha.inverse_transform(
        transformed.reshape(1, 3, -1),
        theta_range=theta_range,
        phi_range=phi_range,
    )

    expected_coords = np.stack([x, y, z], axis=-1)

    assert reconstructed.shape == (1, len(theta_range), len(phi_range), 3)
    assert_array_almost_equal(reconstructed[0], expected_coords, decimal=6)
