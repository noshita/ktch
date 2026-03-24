"""Tests for Disk Harmonic Analysis."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

from ktch.harmonic._disk_harmonic_analysis import (
    DiskHarmonicAnalysis,
    _calc_eigenvalues,
    _complex_to_real_coef,
    _cvt_dha_coef_to_list,
    _disk_harm_basis_matrix,
    _real_to_complex_coef,
    disk_harm,
    xy2polar,
)

#
#   Eigenvalue table
#


class TestCalcEigenvalues:
    def test_shape(self):
        for n_max in (0, 1, 3, 5):
            table = _calc_eigenvalues(n_max)
            assert table.shape == (n_max + 1, n_max + 1)

    def test_origin_is_zero(self):
        table = _calc_eigenvalues(5)
        assert table[0, 0] == 0.0

    def test_lower_triangular(self):
        """Entries with m > n should be zero."""
        n_max = 5
        table = _calc_eigenvalues(n_max)
        for n in range(n_max + 1):
            for m in range(n + 1, n_max + 1):
                assert table[n, m] == 0.0

    def test_positive_valid_entries(self):
        """Valid entries (m <= n, except [0,0]) should be positive."""
        n_max = 5
        table = _calc_eigenvalues(n_max)
        for n in range(1, n_max + 1):
            for m in range(n + 1):
                assert table[n, m] > 0, f"table[{n},{m}] = {table[n, m]}"

    def test_column_monotonicity(self):
        """Within each column m, valid eigenvalues increase with n."""
        n_max = 5
        table = _calc_eigenvalues(n_max)
        for m in range(n_max + 1):
            valid = [table[n, m] for n in range(m, n_max + 1) if table[n, m] > 0]
            if len(valid) > 1:
                assert all(valid[i] < valid[i + 1] for i in range(len(valid) - 1)), (
                    f"column m={m}: {valid}"
                )


#
#   Coordinate conversion
#


class TestXy2Polar:
    def test_origin(self):
        rtheta = xy2polar(np.array([[0.0, 0.0]]))
        assert rtheta[0, 0] == pytest.approx(0.0)

    def test_unit_circle_boundary(self):
        rtheta = xy2polar(np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))
        assert_allclose(rtheta[:, 0], 1.0, atol=1e-14)

    def test_theta_range(self):
        """All theta values should be in [0, 2*pi]."""
        rng = np.random.default_rng(42)
        xy = rng.uniform(-1, 1, size=(100, 2))
        rtheta = xy2polar(xy)
        assert np.all(rtheta[:, 1] >= 0)
        assert np.all(rtheta[:, 1] <= 2 * np.pi + 1e-14)

    def test_centered_false(self):
        """centered=False shifts [0,1] to [-1,1]."""
        rtheta_c = xy2polar(np.array([[0.0, 0.0]]), centered=True)
        rtheta_u = xy2polar(np.array([[0.5, 0.5]]), centered=False)
        assert_allclose(rtheta_c, rtheta_u, atol=1e-14)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="must have shape"):
            xy2polar(np.array([1.0, 2.0]))


#
#   Basis orthonormality
#


class TestBasisOrthonormality:
    def _numerical_inner_product(self, B, r, dr, dtheta):
        """Compute Gram matrix via numerical integration on the unit disk.

        inner product: int_0^1 int_0^{2pi} f_i * f_j * r dr dtheta
        """
        # weight = r * dr * dtheta for each sample point
        w = r * dr * dtheta
        return B.T @ np.diag(w) @ B

    def test_orthonormality(self):
        n_max = 3
        n_r, n_theta = 150, 300
        r_1d = np.linspace(0, 1, n_r)
        theta_1d = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        R, Theta = np.meshgrid(r_1d, theta_1d)
        r = R.ravel()
        theta = Theta.ravel()
        dr = r_1d[1] - r_1d[0]
        dtheta = theta_1d[1] - theta_1d[0]

        B = _disk_harm_basis_matrix(n_max, r, theta)
        gram = self._numerical_inner_product(B, r, dr, dtheta)

        n_coeffs = (n_max + 1) ** 2
        assert_allclose(gram, np.eye(n_coeffs), atol=0.02)


#
#   Coefficient list conversion
#


class TestCvtDhaCoefToList:
    def test_structure(self):
        n_max = 3
        n_coef = (n_max + 1) ** 2
        coef = np.arange(n_coef * 3, dtype=float).reshape(n_coef, 3)
        result = _cvt_dha_coef_to_list(coef)

        assert len(result) == n_max + 1
        for n in range(n_max + 1):
            assert result[n].shape == (2 * n + 1, 3)

    def test_values_preserved(self):
        n_max = 2
        n_coef = (n_max + 1) ** 2
        coef = np.arange(n_coef * 3, dtype=float).reshape(n_coef, 3)
        result = _cvt_dha_coef_to_list(coef)

        # n=0: index 0
        assert_array_almost_equal(result[0][0], coef[0])
        # n=1: indices 1,2,3
        assert_array_almost_equal(result[1], coef[1:4])
        # n=2: indices 4..8
        assert_array_almost_equal(result[2], coef[4:9])

    def test_non_perfect_square_raises(self):
        with pytest.raises(ValueError, match="not a perfect square"):
            _cvt_dha_coef_to_list(np.zeros((10, 3)))

    def test_1d_input(self):
        coef = np.array([1.0, 2.0, 3.0, 4.0])  # n_max=1
        result = _cvt_dha_coef_to_list(coef)
        assert len(result) == 2
        assert result[0].shape == (1,)
        assert result[1].shape == (3,)


#
#   Complex <-> Real conversion
#


class TestComplexRealConversion:
    def test_roundtrip_1d(self):
        """real -> complex -> real is identity."""
        n_max = 3
        n_coef = (n_max + 1) ** 2
        rng = np.random.default_rng(42)
        coef_real = rng.standard_normal(n_coef)

        coef_complex = _real_to_complex_coef(coef_real)
        coef_back = _complex_to_real_coef(coef_complex)
        assert_allclose(coef_back, coef_real, atol=1e-14)

    def test_roundtrip_2d(self):
        """real -> complex -> real is identity for (K, 3) arrays."""
        n_max = 2
        n_coef = (n_max + 1) ** 2
        rng = np.random.default_rng(7)
        coef_real = rng.standard_normal((n_coef, 3))

        coef_complex = _real_to_complex_coef(coef_real)
        coef_back = _complex_to_real_coef(coef_complex)
        assert_allclose(coef_back, coef_real, atol=1e-14)

    def test_conjugate_symmetry(self):
        """Complex coefficients satisfy c_{n,-m} = (-1)^m * conj(c_{n,m})."""
        n_max = 3
        n_coef = (n_max + 1) ** 2
        rng = np.random.default_rng(99)
        coef_real = rng.standard_normal(n_coef)
        coef_complex = _real_to_complex_coef(coef_real)

        for n in range(n_max + 1):
            for m in range(1, n + 1):
                idx_pos = n**2 + n + m
                idx_neg = n**2 + n - m
                expected = ((-1) ** m) * np.conj(coef_complex[idx_pos])
                assert_allclose(coef_complex[idx_neg], expected, atol=1e-14)

    def test_m0_is_real(self):
        """m=0 complex coefficients should be real."""
        n_max = 3
        rng = np.random.default_rng(11)
        coef_real = rng.standard_normal((n_max + 1) ** 2)
        coef_complex = _real_to_complex_coef(coef_real)

        for n in range(n_max + 1):
            idx = n**2 + n
            assert np.imag(coef_complex[idx]) == pytest.approx(0.0, abs=1e-15)

    def test_known_value(self):
        """Manual check: n=1, m=1, c = 2+3j."""
        # n_max=1, 4 coefficients: (0,0),(1,-1),(1,0),(1,1)
        coef_complex = np.array([1 + 0j, 0j, 0.5 + 0j, 2 + 3j])
        # Set (1,-1) via conjugate symmetry: (-1)^1 * conj(2+3j) = -(2-3j) = -2+3j
        coef_complex[1] = -2 + 3j

        coef_real = _complex_to_real_coef(coef_complex)

        # m=0: a = Re(c)
        assert coef_real[0] == pytest.approx(1.0)
        assert coef_real[2] == pytest.approx(0.5)
        # m=1: a = sqrt(2)*Re(c) = sqrt(2)*2
        assert coef_real[3] == pytest.approx(np.sqrt(2) * 2)
        # m=-1: a = -sqrt(2)*Im(c_{1,1}) = -sqrt(2)*3
        assert coef_real[1] == pytest.approx(-np.sqrt(2) * 3)


#
#   DiskHarmonicAnalysis class
#


def _generate_synthetic_surface(n_max, n_points, n_dim=3, seed=42):
    """Generate a synthetic surface with known real coefficients.

    Returns (vertices, r_theta, coef_true) where coef_true is the
    flat real coefficient vector of shape (n_dim*(n_max+1)^2,).
    """
    rng = np.random.default_rng(seed)
    n_coeffs = (n_max + 1) ** 2

    coef_matrix = rng.standard_normal((n_coeffs, n_dim))

    # Random points inside the unit disk
    r = np.sqrt(rng.uniform(0, 1, n_points))  # sqrt for uniform area
    theta = rng.uniform(0, 2 * np.pi, n_points)

    B = _disk_harm_basis_matrix(n_max, r, theta)
    vertices = B @ coef_matrix

    r_theta = np.column_stack([r, theta])
    coef_true = coef_matrix.T.ravel()

    return vertices, r_theta, coef_true


class TestDHAShape:
    def test_transform_shape(self):
        n_max = 3
        vertices, r_theta, _ = _generate_synthetic_surface(n_max, 200)

        dha = DiskHarmonicAnalysis(n_harmonics=n_max, n_jobs=1)
        transformed = dha.fit_transform([vertices], r_theta=[r_theta])

        assert transformed.shape == (1, 3 * (n_max + 1) ** 2)

    def test_multiple_samples(self):
        n_max = 2
        v1, rt1, _ = _generate_synthetic_surface(n_max, 100, seed=1)
        v2, rt2, _ = _generate_synthetic_surface(n_max, 150, seed=2)

        dha = DiskHarmonicAnalysis(n_harmonics=n_max, n_jobs=1)
        transformed = dha.fit_transform([v1, v2], r_theta=[rt1, rt2])

        assert transformed.shape == (2, 3 * (n_max + 1) ** 2)


class TestDHARoundTrip:
    def test_coefficient_recovery(self):
        """Transform synthetic surface and recover exact coefficients."""
        n_max = 3
        n_points = 500

        vertices, r_theta, coef_true = _generate_synthetic_surface(n_max, n_points)

        dha = DiskHarmonicAnalysis(n_harmonics=n_max, n_jobs=1)
        transformed = dha.fit_transform([vertices], r_theta=[r_theta])

        assert_allclose(transformed[0], coef_true, atol=1e-8)

    def test_transform_inverse_roundtrip(self):
        """transform -> inverse_transform recovers the surface."""
        n_max = 3
        n_points = 500

        vertices, r_theta, _ = _generate_synthetic_surface(n_max, n_points)
        r_range = np.linspace(0, 1, 30)
        theta_range = np.linspace(0, 2 * np.pi, 60)

        dha = DiskHarmonicAnalysis(n_harmonics=n_max, n_jobs=1)
        transformed = dha.fit_transform([vertices], r_theta=[r_theta])

        reconstructed = dha.inverse_transform(
            transformed, r_range=r_range, theta_range=theta_range
        )

        assert reconstructed.shape == (1, len(theta_range), len(r_range), 3)

        # Verify by evaluating the known coefficients at the same grid
        r_grid, theta_grid = np.meshgrid(r_range, theta_range)
        B = _disk_harm_basis_matrix(n_max, r_grid.ravel(), theta_grid.ravel())
        n_coeffs = (n_max + 1) ** 2
        coef_matrix = transformed[0].reshape(3, n_coeffs).T
        expected_coords = (B @ coef_matrix).reshape(len(theta_range), len(r_range), 3)

        assert_allclose(reconstructed[0], expected_coords, atol=1e-10)


class TestDHAFeatureNames:
    def test_feature_names(self):
        n_max = 2
        dha = DiskHarmonicAnalysis(n_harmonics=n_max)
        names = dha.get_feature_names_out()
        n_terms = (n_max + 1) ** 2

        assert len(names) == 3 * n_terms
        assert names[0] == "cx_0_0"
        assert names[1] == "cx_1_-1"
        assert names[n_terms] == "cy_0_0"
        assert names[2 * n_terms] == "cz_0_0"

    def test_n_features_out(self):
        dha = DiskHarmonicAnalysis(n_harmonics=5)
        assert dha._n_features_out == 3 * 36


class TestDHAValidation:
    def test_r_theta_required(self):
        dha = DiskHarmonicAnalysis(n_harmonics=2)
        with pytest.raises(ValueError, match="r_theta is required"):
            dha.transform([np.zeros((10, 3))])

    def test_length_mismatch_raises(self):
        dha = DiskHarmonicAnalysis(n_harmonics=2, n_jobs=1)
        with pytest.raises(ValueError, match="same length"):
            dha.transform(
                [np.zeros((10, 3))],
                r_theta=[np.zeros((10, 2)), np.zeros((5, 2))],
            )

    def test_underdetermined_warns(self):
        n_max = 3
        n_coeffs = (n_max + 1) ** 2  # 16
        n_coords = 5  # < 16
        rng = np.random.default_rng(0)
        coords = rng.standard_normal((n_coords, 3))
        r_theta = np.column_stack(
            [np.linspace(0.1, 0.9, n_coords), np.linspace(0, 2 * np.pi, n_coords)]
        )

        dha = DiskHarmonicAnalysis(n_harmonics=n_max, n_jobs=1)
        with pytest.warns(UserWarning, match="Underdetermined system"):
            dha.fit_transform([coords], r_theta=[r_theta])


#
#   Edge cases
#


class TestDHANHarmonicsZero:
    """n_harmonics=0: only the constant term, 3 features total."""

    def test_shape(self):
        n_max = 0
        rng = np.random.default_rng(0)
        r = np.sqrt(rng.uniform(0, 1, 50))
        theta = rng.uniform(0, 2 * np.pi, 50)
        r_theta = np.column_stack([r, theta])

        B = _disk_harm_basis_matrix(0, r, theta)
        coef = rng.standard_normal((1, 3))
        vertices = B @ coef

        dha = DiskHarmonicAnalysis(n_harmonics=n_max, n_jobs=1)
        transformed = dha.fit_transform([vertices], r_theta=[r_theta])

        assert transformed.shape == (1, 3)

    def test_coefficient_recovery(self):
        n_max = 0
        rng = np.random.default_rng(0)
        r = np.sqrt(rng.uniform(0, 1, 50))
        theta = rng.uniform(0, 2 * np.pi, 50)
        r_theta = np.column_stack([r, theta])

        B = _disk_harm_basis_matrix(0, r, theta)
        coef = np.array([[1.0, 0.5, -0.2]])
        vertices = B @ coef

        dha = DiskHarmonicAnalysis(n_harmonics=n_max, n_jobs=1)
        transformed = dha.fit_transform([vertices], r_theta=[r_theta])

        assert_allclose(transformed[0], [1.0, 0.5, -0.2], atol=1e-10)

    def test_feature_names(self):
        dha = DiskHarmonicAnalysis(n_harmonics=0)
        names = dha.get_feature_names_out()
        assert list(names) == ["cx_0_0", "cy_0_0", "cz_0_0"]

    def test_inverse_transform(self):
        n_max = 0
        rng = np.random.default_rng(0)
        r = np.sqrt(rng.uniform(0, 1, 50))
        theta = rng.uniform(0, 2 * np.pi, 50)
        r_theta = np.column_stack([r, theta])

        B = _disk_harm_basis_matrix(0, r, theta)
        coef = np.array([[1.0, 0.5, -0.2]])
        vertices = B @ coef

        dha = DiskHarmonicAnalysis(n_harmonics=n_max, n_jobs=1)
        transformed = dha.fit_transform([vertices], r_theta=[r_theta])

        r_range = np.linspace(0, 1, 10)
        theta_range = np.linspace(0, 2 * np.pi, 20)
        recon = dha.inverse_transform(
            transformed, r_range=r_range, theta_range=theta_range
        )

        assert recon.shape == (1, len(theta_range), len(r_range), 3)
        # Constant surface: should be uniform across the grid
        for dim in range(3):
            surface = recon[0, :, :, dim]
            assert np.std(surface) < 1e-10


class TestDHANHarmonicsOne:
    """n_harmonics=1: 4 coefficients per axis, 12 total."""

    def test_shape(self):
        n_max = 1
        vertices, r_theta, _ = _generate_synthetic_surface(n_max, 50)

        dha = DiskHarmonicAnalysis(n_harmonics=n_max, n_jobs=1)
        transformed = dha.fit_transform([vertices], r_theta=[r_theta])

        assert transformed.shape == (1, 12)

    def test_round_trip(self):
        n_max = 1
        vertices, r_theta, coef_true = _generate_synthetic_surface(n_max, 100)

        dha = DiskHarmonicAnalysis(n_harmonics=n_max, n_jobs=1)
        transformed = dha.fit_transform([vertices], r_theta=[r_theta])

        assert_allclose(transformed[0], coef_true, atol=1e-8)


#
#   Standalone disk_harm reconstruction
#


class TestDiskHarmReconstruction:
    def test_output_shape(self):
        n_max = 3
        n_coeffs = (n_max + 1) ** 2
        rng = np.random.default_rng(0)
        coef_matrix = rng.standard_normal((n_coeffs, 3))
        coef_list = _cvt_dha_coef_to_list(coef_matrix)

        r_range = np.linspace(0, 1, 50)
        theta_range = np.linspace(0, 2 * np.pi, 100)

        x, y, z = disk_harm(n_max, coef_list, r_range, theta_range)

        assert x.shape == (len(theta_range), len(r_range))
        assert y.shape == (len(theta_range), len(r_range))
        assert z.shape == (len(theta_range), len(r_range))

    def test_truncation(self):
        """Using smaller n_max for reconstruction works."""
        n_max_full = 5
        n_max_trunc = 2
        n_coeffs = (n_max_full + 1) ** 2
        rng = np.random.default_rng(0)
        coef_matrix = rng.standard_normal((n_coeffs, 3))
        coef_list = _cvt_dha_coef_to_list(coef_matrix)

        x, y, z = disk_harm(n_max_trunc, coef_list)

        assert x.shape[0] > 0
        assert np.all(np.isfinite(x))

    def test_default_grid(self):
        n_max = 2
        n_coeffs = (n_max + 1) ** 2
        rng = np.random.default_rng(0)
        coef_matrix = rng.standard_normal((n_coeffs, 3))
        coef_list = _cvt_dha_coef_to_list(coef_matrix)

        x, y, z = disk_harm(n_max, coef_list)

        assert x.shape == (180, 100)


class TestDHAInverseTransformDefaults:
    def test_default_grid(self):
        n_max = 2
        vertices, r_theta, _ = _generate_synthetic_surface(n_max, 100)

        dha = DiskHarmonicAnalysis(n_harmonics=n_max, n_jobs=1)
        transformed = dha.fit_transform([vertices], r_theta=[r_theta])

        result = dha.inverse_transform(transformed)
        assert result.shape == (1, 180, 100, 3)


#
#   2D mode (n_dim=2)
#


class TestDHA2DShape:
    def test_transform_shape(self):
        n_max = 3
        vertices, r_theta, _ = _generate_synthetic_surface(n_max, 200, n_dim=2)

        dha = DiskHarmonicAnalysis(n_harmonics=n_max, n_dim=2, n_jobs=1)
        transformed = dha.fit_transform([vertices], r_theta=[r_theta])

        assert transformed.shape == (1, 2 * (n_max + 1) ** 2)

    def test_multiple_samples(self):
        n_max = 2
        v1, rt1, _ = _generate_synthetic_surface(n_max, 100, n_dim=2, seed=1)
        v2, rt2, _ = _generate_synthetic_surface(n_max, 150, n_dim=2, seed=2)

        dha = DiskHarmonicAnalysis(n_harmonics=n_max, n_dim=2, n_jobs=1)
        transformed = dha.fit_transform([v1, v2], r_theta=[rt1, rt2])

        assert transformed.shape == (2, 2 * (n_max + 1) ** 2)


class TestDHA2DRoundTrip:
    def test_coefficient_recovery(self):
        n_max = 3
        vertices, r_theta, coef_true = _generate_synthetic_surface(n_max, 500, n_dim=2)

        dha = DiskHarmonicAnalysis(n_harmonics=n_max, n_dim=2, n_jobs=1)
        transformed = dha.fit_transform([vertices], r_theta=[r_theta])

        assert_allclose(transformed[0], coef_true, atol=1e-8)

    def test_transform_inverse_roundtrip(self):
        n_max = 3
        vertices, r_theta, _ = _generate_synthetic_surface(n_max, 500, n_dim=2)
        r_range = np.linspace(0, 1, 30)
        theta_range = np.linspace(0, 2 * np.pi, 60)

        dha = DiskHarmonicAnalysis(n_harmonics=n_max, n_dim=2, n_jobs=1)
        transformed = dha.fit_transform([vertices], r_theta=[r_theta])

        reconstructed = dha.inverse_transform(
            transformed, r_range=r_range, theta_range=theta_range
        )

        assert reconstructed.shape == (1, len(theta_range), len(r_range), 2)

        # Verify against direct computation
        r_grid, theta_grid = np.meshgrid(r_range, theta_range)
        B = _disk_harm_basis_matrix(n_max, r_grid.ravel(), theta_grid.ravel())
        n_coeffs = (n_max + 1) ** 2
        coef_matrix = transformed[0].reshape(2, n_coeffs).T
        expected = (B @ coef_matrix).reshape(len(theta_range), len(r_range), 2)

        assert_allclose(reconstructed[0], expected, atol=1e-10)


class TestDHA2DFeatureNames:
    def test_feature_names(self):
        dha = DiskHarmonicAnalysis(n_harmonics=2, n_dim=2)
        names = dha.get_feature_names_out()
        n_terms = (2 + 1) ** 2

        assert len(names) == 2 * n_terms
        assert names[0] == "cx_0_0"
        assert names[n_terms] == "cy_0_0"

    def test_n_features_out(self):
        dha = DiskHarmonicAnalysis(n_harmonics=5, n_dim=2)
        assert dha._n_features_out == 2 * 36


class TestDHA2DEdgeCases:
    def test_n_harmonics_zero(self):
        """n_dim=2, n_harmonics=0: 2 features total."""
        rng = np.random.default_rng(0)
        r = np.sqrt(rng.uniform(0, 1, 50))
        theta = rng.uniform(0, 2 * np.pi, 50)
        r_theta = np.column_stack([r, theta])

        B = _disk_harm_basis_matrix(0, r, theta)
        coef = np.array([[1.0, -0.5]])
        vertices = B @ coef

        dha = DiskHarmonicAnalysis(n_harmonics=0, n_dim=2, n_jobs=1)
        transformed = dha.fit_transform([vertices], r_theta=[r_theta])

        assert transformed.shape == (1, 2)
        assert_allclose(transformed[0], [1.0, -0.5], atol=1e-10)

    def test_feature_names_zero(self):
        dha = DiskHarmonicAnalysis(n_harmonics=0, n_dim=2)
        assert list(dha.get_feature_names_out()) == ["cx_0_0", "cy_0_0"]

    def test_invalid_n_dim_raises(self):
        dha = DiskHarmonicAnalysis(n_harmonics=2, n_dim=4, n_jobs=1)
        with pytest.raises(ValueError, match="n_dim must be 2 or 3"):
            dha.transform([np.zeros((10, 4))], r_theta=[np.zeros((10, 2))])


class TestDiskHarm2DReconstruction:
    def test_output_shape(self):
        n_max = 3
        n_coeffs = (n_max + 1) ** 2
        rng = np.random.default_rng(0)
        coef_matrix = rng.standard_normal((n_coeffs, 2))
        coef_list = _cvt_dha_coef_to_list(coef_matrix)

        r_range = np.linspace(0, 1, 50)
        theta_range = np.linspace(0, 2 * np.pi, 100)

        result = disk_harm(n_max, coef_list, r_range, theta_range)

        assert len(result) == 2
        assert result[0].shape == (len(theta_range), len(r_range))
        assert result[1].shape == (len(theta_range), len(r_range))

    def test_3d_unpack_still_works(self):
        """Backward compat: x, y, z = disk_harm(...) with 3D coefficients."""
        n_max = 2
        n_coeffs = (n_max + 1) ** 2
        rng = np.random.default_rng(0)
        coef_matrix = rng.standard_normal((n_coeffs, 3))
        coef_list = _cvt_dha_coef_to_list(coef_matrix)

        x, y, z = disk_harm(n_max, coef_list)

        assert x.shape == (180, 100)
