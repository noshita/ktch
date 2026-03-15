import numpy as np
import pytest
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
    transformed = sha.fit_transform([coords], theta_phi=[theta_phi])

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


def test_get_feature_names_out():
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
    sha.fit_transform([coords], theta_phi=[theta_phi])

    names = sha.get_feature_names_out()
    n_terms = (l_max + 1) ** 2
    assert len(names) == 3 * n_terms
    assert names[0] == "cx_0_0"
    assert names[n_terms] == "cy_0_0"
    assert names[2 * n_terms] == "cz_0_0"


def test_theta_phi_required():
    sha = SphericalHarmonicAnalysis(n_harmonics=2)
    with pytest.raises(ValueError, match="theta_phi is required"):
        sha.transform([np.zeros((10, 3))])


def test_underdetermined_system_warns():
    """Warn when n_coords < (l_max+1)**2."""
    l_max = 3
    n_coeffs = (l_max + 1) ** 2  # 16
    n_coords = 5  # < 16

    coords = np.random.default_rng(0).standard_normal((n_coords, 3))
    theta_phi = np.column_stack(
        [np.linspace(0.1, np.pi - 0.1, n_coords), np.linspace(0, 2 * np.pi, n_coords)]
    )

    sha = SphericalHarmonicAnalysis(n_harmonics=l_max, n_jobs=1)
    with pytest.warns(UserWarning, match="Underdetermined system"):
        sha.fit_transform([coords], theta_phi=[theta_phi])


class TestNHarmonicsOneSHA:
    """Tests for n_harmonics=1 (l_max=1) in SphericalHarmonicAnalysis.

    l_max=1 produces (1+1)^2 = 4 coefficients per axis (12 total).
    This is the minimum non-trivial harmonic count that includes both
    the constant term (l=0) and the first-order dipole (l=1).

    Test surfaces use parametric ellipsoids (exactly representable at
    l_max=1) to avoid passing real-only coefficients to spharm(), which
    would produce complex coordinates and trigger a UserWarning.
    """

    def test_shape(self):
        """SHA with n_harmonics=1 produces output shape (1, 3*4) = (1, 12)."""
        l_max = 1
        n_terms = (l_max + 1) ** 2  # 4
        theta_range = np.linspace(0.1, np.pi - 0.1, 6)
        phi_range = np.linspace(0, 2 * np.pi, 12)
        theta_grid, phi_grid = np.meshgrid(theta_range, phi_range, indexing="ij")

        x = 1.0 * np.sin(theta_grid) * np.cos(phi_grid)
        y = 0.5 * np.sin(theta_grid) * np.sin(phi_grid)
        z = 0.3 * np.cos(theta_grid)

        coords = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        theta_phi = np.stack([theta_grid.ravel(), phi_grid.ravel()], axis=-1)

        sha = SphericalHarmonicAnalysis(n_harmonics=l_max, n_jobs=1)
        transformed = sha.fit_transform([coords], theta_phi=[theta_phi])

        assert transformed.shape == (1, 3 * n_terms)

    def test_round_trip(self):
        """SHA with n_harmonics=1: forward + inverse recovers the surface."""
        l_max = 1
        theta_range = np.linspace(0.1, np.pi - 0.1, 6)
        phi_range = np.linspace(0, 2 * np.pi, 12)
        theta_grid, phi_grid = np.meshgrid(theta_range, phi_range, indexing="ij")

        x = 1.0 * np.sin(theta_grid) * np.cos(phi_grid)
        y = 0.5 * np.sin(theta_grid) * np.sin(phi_grid)
        z = 0.3 * np.cos(theta_grid)

        coords = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        theta_phi = np.stack([theta_grid.ravel(), phi_grid.ravel()], axis=-1)

        sha = SphericalHarmonicAnalysis(n_harmonics=l_max, n_jobs=1)
        transformed = sha.fit_transform([coords], theta_phi=[theta_phi])

        reconstructed = sha.inverse_transform(
            transformed.reshape(1, 3, -1),
            theta_range=theta_range,
            phi_range=phi_range,
        )

        expected = np.stack([x, y, z], axis=-1)
        assert reconstructed.shape == (1, len(theta_range), len(phi_range), 3)
        assert_array_almost_equal(reconstructed[0], expected, decimal=5)

    def test_feature_names(self):
        """SHA with n_harmonics=1 produces correct feature names."""
        l_max = 1
        theta_range = np.linspace(0.1, np.pi - 0.1, 6)
        phi_range = np.linspace(0, 2 * np.pi, 12)

        coef_list = [
            np.array([[1.0, 0.0, 0.0]]),
            np.zeros((3, 3)),
        ]

        x, y, z = spharm(l_max, coef_list, theta_range=theta_range, phi_range=phi_range)
        coords = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        theta_grid, phi_grid = np.meshgrid(theta_range, phi_range, indexing="ij")
        theta_phi = np.stack([theta_grid.ravel(), phi_grid.ravel()], axis=-1)

        sha = SphericalHarmonicAnalysis(n_harmonics=l_max, n_jobs=1)
        sha.fit_transform([coords], theta_phi=[theta_phi])

        names = sha.get_feature_names_out()
        n_terms = (l_max + 1) ** 2  # 4
        assert len(names) == 3 * n_terms
        assert names[0] == "cx_0_0"
        assert names[1] == "cx_1_-1"


class TestXyz2SphericalDegenerate:
    """Tests for xyz2spherical with degenerate (pole) inputs."""

    def test_north_pole(self):
        """North pole (0, 0, 1): theta=0, phi=0 (convention)."""
        coords = np.array([[0.0, 0.0, 1.0]])
        result = xyz2spherical(coords)
        assert result[0, 0] == pytest.approx(0.0)
        assert np.isfinite(result[0, 1])
        assert result[0, 1] == pytest.approx(0.0)

    def test_south_pole(self):
        """South pole (0, 0, -1): theta=pi, phi=0 (convention)."""
        coords = np.array([[0.0, 0.0, -1.0]])
        result = xyz2spherical(coords)
        assert result[0, 0] == pytest.approx(np.pi)
        assert np.isfinite(result[0, 1])
        assert result[0, 1] == pytest.approx(0.0)

    def test_mixed_poles_and_equator(self):
        """Mix of poles and equatorial points all produce finite results."""
        coords = np.array([
            [0.0, 0.0, 1.0],   # north pole
            [0.0, 0.0, -1.0],  # south pole
            [1.0, 0.0, 0.0],   # equator x-axis
            [0.0, 1.0, 0.0],   # equator y-axis
        ])
        result = xyz2spherical(coords)
        assert np.all(np.isfinite(result))
        assert result.shape == (4, 2)
        # Equatorial points
        assert result[2, 0] == pytest.approx(np.pi / 2)  # theta
        assert result[2, 1] == pytest.approx(0.0)         # phi (x-axis)
        assert result[3, 0] == pytest.approx(np.pi / 2)  # theta
        assert result[3, 1] == pytest.approx(np.pi / 2)  # phi (y-axis)

    def test_near_pole(self):
        """Points very close to poles produce correct theta values."""
        eps = 1e-15
        coords = np.array([
            [eps, 0.0, 1.0],
            [0.0, eps, -1.0],
        ])
        result = xyz2spherical(coords)
        assert np.all(np.isfinite(result))
        # Near north pole: theta ≈ 0
        assert result[0, 0] == pytest.approx(0.0, abs=1e-10)
        # Near south pole: theta ≈ pi
        assert result[1, 0] == pytest.approx(np.pi, abs=1e-10)


class TestSHARoundTripMixedSpectrum:
    """Round-trip test with a surface that spans multiple harmonic degrees.

    Uses an asymmetrically scaled ellipsoid with an axis-dependent offset,
    generating non-trivial content in both l=0, l=1, and l=2 harmonics.
    The coordinate-level round-trip (coords -> coefficients -> inverse ->
    coords) is the correct invariant.
    """

    def test_mixed_spectrum_round_trip(self):
        """coords -> SHA -> inverse recovers the real-valued surface."""
        l_max = 2
        theta_range = np.linspace(0.1, np.pi - 0.1, 8)
        phi_range = np.linspace(0, 2 * np.pi, 16)
        theta_grid, phi_grid = np.meshgrid(theta_range, phi_range, indexing="ij")

        # Asymmetric ellipsoid with offset
        x = 2.0 * np.sin(theta_grid) * np.cos(phi_grid) + 0.5
        y = 1.5 * np.sin(theta_grid) * np.sin(phi_grid) - 0.3
        z = 1.0 * np.cos(theta_grid) + 0.2

        coords = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        theta_phi = np.stack([theta_grid.ravel(), phi_grid.ravel()], axis=-1)

        sha = SphericalHarmonicAnalysis(n_harmonics=l_max, n_jobs=1)
        transformed = sha.fit_transform([coords], theta_phi=[theta_phi])

        n_terms = (l_max + 1) ** 2
        assert transformed.shape == (1, 3 * n_terms)

        reconstructed = sha.inverse_transform(
            transformed.reshape(1, 3, -1),
            theta_range=theta_range,
            phi_range=phi_range,
        )
        expected = np.stack([x, y, z], axis=-1)
        assert_array_almost_equal(reconstructed[0], expected, decimal=5)


class TestSHAInverseTransformDefaults:
    """Test inverse_transform with default theta_range/phi_range."""

    def test_default_grid(self):
        """inverse_transform with no theta/phi args uses default 90x180 grid."""
        l_max = 2
        theta_range = np.linspace(0.1, np.pi - 0.1, 10)
        phi_range = np.linspace(0, 2 * np.pi, 20)

        coef_list = [
            np.array([[1.0, 0.5, -0.2]]),
            np.zeros((3, 3)),
            np.zeros((5, 3)),
        ]

        x, y, z = spharm(l_max, coef_list, theta_range=theta_range, phi_range=phi_range)
        coords = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        theta_grid, phi_grid = np.meshgrid(theta_range, phi_range, indexing="ij")
        theta_phi = np.stack([theta_grid.ravel(), phi_grid.ravel()], axis=-1)

        sha = SphericalHarmonicAnalysis(n_harmonics=l_max, n_jobs=1)
        transformed = sha.fit_transform([coords], theta_phi=[theta_phi])

        # Call without theta_range/phi_range -> uses defaults (90, 180)
        result = sha.inverse_transform(transformed.reshape(1, 3, -1))
        assert result.shape == (1, 90, 180, 3)

        # Input has l=0 only (constant function), so the reconstructed
        # surface should be approximately constant across the grid.
        for dim in range(3):
            surface = result[0, :, :, dim]
            assert np.std(surface) < 1e-5, (
                f"Dimension {dim}: l=0-only surface should be constant, "
                f"but std={np.std(surface):.2e}"
            )


class TestSHANHarmonicsZero:
    """Tests for n_harmonics=0 (l=0 only, constant function).

    l_max=0 produces a single coefficient per axis (3 total).
    The surface is a constant point in 3D space.
    """

    def test_shape(self):
        """SHA with n_harmonics=0 produces output shape (1, 3)."""
        theta_range = np.linspace(0.1, np.pi - 0.1, 10)
        phi_range = np.linspace(0, 2 * np.pi, 20)

        coef_list = [np.array([[1.0, 0.5, -0.2]])]
        x, y, z = spharm(0, coef_list, theta_range=theta_range, phi_range=phi_range)
        coords = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        theta_grid, phi_grid = np.meshgrid(theta_range, phi_range, indexing="ij")
        theta_phi = np.stack([theta_grid.ravel(), phi_grid.ravel()], axis=-1)

        sha = SphericalHarmonicAnalysis(n_harmonics=0, n_jobs=1)
        transformed = sha.fit_transform([coords], theta_phi=[theta_phi])

        assert transformed.shape == (1, 3)

    def test_coefficient_recovery(self):
        """SHA with n_harmonics=0 recovers the l=0 coefficient."""
        theta_range = np.linspace(0.1, np.pi - 0.1, 10)
        phi_range = np.linspace(0, 2 * np.pi, 20)

        coef_list = [np.array([[1.0, 0.5, -0.2]])]
        x, y, z = spharm(0, coef_list, theta_range=theta_range, phi_range=phi_range)
        coords = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        theta_grid, phi_grid = np.meshgrid(theta_range, phi_range, indexing="ij")
        theta_phi = np.stack([theta_grid.ravel(), phi_grid.ravel()], axis=-1)

        sha = SphericalHarmonicAnalysis(n_harmonics=0, n_jobs=1)
        transformed = sha.fit_transform([coords], theta_phi=[theta_phi])

        assert_allclose(np.real(transformed[0]), coef_list[0][0], atol=1e-10)

    def test_feature_names(self):
        """SHA with n_harmonics=0 produces 3 feature names."""
        sha = SphericalHarmonicAnalysis(n_harmonics=0)
        names = sha.get_feature_names_out()
        assert list(names) == ["cx_0_0", "cy_0_0", "cz_0_0"]

    def test_inverse_transform(self):
        """SHA with n_harmonics=0: inverse produces a constant surface."""
        theta_range = np.linspace(0.1, np.pi - 0.1, 5)
        phi_range = np.linspace(0, 2 * np.pi, 10)

        coef_list = [np.array([[1.0, 0.5, -0.2]])]
        x, y, z = spharm(0, coef_list, theta_range=theta_range, phi_range=phi_range)
        coords = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        theta_grid, phi_grid = np.meshgrid(theta_range, phi_range, indexing="ij")
        theta_phi = np.stack([theta_grid.ravel(), phi_grid.ravel()], axis=-1)

        sha = SphericalHarmonicAnalysis(n_harmonics=0, n_jobs=1)
        transformed = sha.fit_transform([coords], theta_phi=[theta_phi])

        recon = sha.inverse_transform(
            transformed.reshape(1, 3, -1),
            theta_range=theta_range,
            phi_range=phi_range,
        )
        expected = np.stack([x, y, z], axis=-1)
        assert recon.shape == (1, len(theta_range), len(phi_range), 3)
        assert_array_almost_equal(recon[0], expected, decimal=10)


class TestSHANonFiniteInput:
    """Tests for NaN/Inf input to SHA.

    scipy.linalg.lstsq raises ValueError for non-finite input.
    """

    def test_nan_input_raises(self):
        """SHA with NaN in coordinates raises ValueError."""
        theta_phi = np.column_stack([
            np.linspace(0.1, np.pi - 0.1, 10),
            np.linspace(0, 2 * np.pi, 10),
        ])
        coords = np.ones((10, 3))
        coords[3, 0] = np.nan

        sha = SphericalHarmonicAnalysis(n_harmonics=2, n_jobs=1)
        with pytest.raises(ValueError, match="infs or NaNs"):
            sha.fit_transform([coords], theta_phi=[theta_phi])

    def test_inf_input_raises(self):
        """SHA with Inf in coordinates raises ValueError."""
        theta_phi = np.column_stack([
            np.linspace(0.1, np.pi - 0.1, 10),
            np.linspace(0, 2 * np.pi, 10),
        ])
        coords = np.ones((10, 3))
        coords[5, 1] = np.inf

        sha = SphericalHarmonicAnalysis(n_harmonics=2, n_jobs=1)
        with pytest.raises(ValueError, match="infs or NaNs"):
            sha.fit_transform([coords], theta_phi=[theta_phi])
