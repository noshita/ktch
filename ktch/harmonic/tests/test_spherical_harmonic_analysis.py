import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

from ktch.harmonic import SphericalHarmonicAnalysis, spharm, xyz2spherical
from ktch.harmonic._spherical_harmonic_analysis import (
    _complex_to_real_sph_coef,
    _real_sph_harm_basis_matrix,
    _real_to_complex_sph_coef,
    cvt_spharm_coef_to_list,
)


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
        transformed,
        theta_range=theta_range,
        phi_range=phi_range,
    )

    expected_coords = np.stack([x, y, z], axis=-1)

    assert reconstructed.shape == (1, len(theta_range), len(phi_range), 3)
    assert_array_almost_equal(reconstructed[0], expected_coords, decimal=6)


def test_get_feature_names_out():
    l_max = 2
    sha = SphericalHarmonicAnalysis(n_harmonics=l_max)
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
        coords = np.array(
            [
                [0.0, 0.0, 1.0],  # north pole
                [0.0, 0.0, -1.0],  # south pole
                [1.0, 0.0, 0.0],  # equator x-axis
                [0.0, 1.0, 0.0],  # equator y-axis
            ]
        )
        result = xyz2spherical(coords)
        assert np.all(np.isfinite(result))
        assert result.shape == (4, 2)
        # Equatorial points
        assert result[2, 0] == pytest.approx(np.pi / 2)  # theta
        assert result[2, 1] == pytest.approx(0.0)  # phi (x-axis)
        assert result[3, 0] == pytest.approx(np.pi / 2)  # theta
        assert result[3, 1] == pytest.approx(np.pi / 2)  # phi (y-axis)

    def test_near_pole(self):
        """Points very close to poles produce correct theta values."""
        eps = 1e-15
        coords = np.array(
            [
                [eps, 0.0, 1.0],
                [0.0, eps, -1.0],
            ]
        )
        result = xyz2spherical(coords)
        assert np.all(np.isfinite(result))
        assert result[0, 0] == pytest.approx(0.0, abs=1e-10)
        assert result[1, 0] == pytest.approx(np.pi, abs=1e-10)


class TestSHARoundTripMixedSpectrum:
    """Round-trip test with a surface that spans multiple harmonic degrees.

    Uses an asymmetrically scaled ellipsoid with an axis-dependent offset,
    generating non-trivial content in both l=0, l=1, and l=2 harmonics.
    """

    def test_mixed_spectrum_round_trip(self):
        """coords -> SHA -> inverse recovers the surface."""
        l_max = 2
        theta_range = np.linspace(0.1, np.pi - 0.1, 8)
        phi_range = np.linspace(0, 2 * np.pi, 16)
        theta_grid, phi_grid = np.meshgrid(theta_range, phi_range, indexing="ij")

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
            transformed,
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

        result = sha.inverse_transform(transformed)
        assert result.shape == (1, 90, 180, 3)

        for dim in range(3):
            surface = result[0, :, :, dim]
            assert np.std(surface) < 1e-5, (
                f"Dimension {dim}: l=0-only surface should be constant, "
                f"but std={np.std(surface):.2e}"
            )


class TestSHANHarmonicsZero:
    """Tests for n_harmonics=0 (l=0 only, constant function)."""

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

        assert transformed.shape == (1, 3)
        assert_allclose(transformed[0], coef_list[0][0], atol=1e-10)

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
            transformed,
            theta_range=theta_range,
            phi_range=phi_range,
        )
        expected = np.stack([x, y, z], axis=-1)
        assert recon.shape == (1, len(theta_range), len(phi_range), 3)
        assert_array_almost_equal(recon[0], expected, decimal=10)


class TestSHAInverseTransformLMaxTruncation:
    """Tests for ``inverse_transform(l_max=L)`` with ``L < n_harmonics``.

    Before the fix, passing ``l_max < n_harmonics`` raised
    ``ValueError: cannot reshape ...`` because the flat coefficient
    vector (length ``3 * (n_harmonics + 1) ** 2``) was reshaped using
    ``(l_max + 1) ** 2`` as the column count.
    """

    @staticmethod
    def _make_axis_asymmetric_sha(n_harmonics, n_th=40, n_ph=80):
        """Build an axis-asymmetric surface fitted by SHA at full degree."""
        th = np.linspace(0.05, np.pi - 0.05, n_th)
        ph = np.linspace(0, 2 * np.pi, n_ph, endpoint=False)
        tg, pg = np.meshgrid(th, ph, indexing="ij")
        tf, pf = tg.ravel(), pg.ravel()
        r = 1.0 + 0.05 * np.sin(2 * tf) * np.cos(3 * pf)
        coords = np.stack(
            [
                1.0 * r * np.sin(tf) * np.cos(pf),
                0.5 * r * np.sin(tf) * np.sin(pf),
                2.0 * r * np.cos(tf),
            ],
            axis=-1,
        )
        theta_phi = np.stack([tf, pf], axis=-1)
        sha = SphericalHarmonicAnalysis(n_harmonics=n_harmonics, n_jobs=1)
        flat = sha.transform([coords], theta_phi=[theta_phi])
        return sha, flat, th, ph

    def test_l_max_less_than_n_harmonics_runs(self):
        """l_max < n_harmonics no longer raises and returns a valid surface."""
        sha, flat, th, ph = self._make_axis_asymmetric_sha(n_harmonics=6)
        recon = sha.inverse_transform(flat, theta_range=th, phi_range=ph, l_max=3)
        assert recon.shape == (1, len(th), len(ph), 3)
        assert np.all(np.isfinite(recon))

    def test_l_max_truncation_preserves_axis_layout(self):
        """Per-axis std of truncated reconstruction matches full reconstruction.

        Confirms the truncation respects the axis-major layout: the cx,
        cy, cz blocks are sliced independently rather than mixed.
        """
        sha, flat, th, ph = self._make_axis_asymmetric_sha(n_harmonics=6)
        full = sha.inverse_transform(flat, theta_range=th, phi_range=ph)[0]
        trunc = sha.inverse_transform(
            flat, theta_range=th, phi_range=ph, l_max=3
        )[0]
        for k in range(3):
            assert_allclose(full[..., k].std(), trunc[..., k].std(), rtol=0.05)

    def test_l_max_default_equals_n_harmonics(self):
        """l_max=None and l_max=n_harmonics produce the same output."""
        sha, flat, th, ph = self._make_axis_asymmetric_sha(n_harmonics=4)
        a = sha.inverse_transform(flat, theta_range=th, phi_range=ph)
        b = sha.inverse_transform(
            flat, theta_range=th, phi_range=ph, l_max=sha.n_harmonics
        )
        assert_array_almost_equal(a, b)

    def test_l_max_gt_n_harmonics_raises(self):
        sha, flat, th, ph = self._make_axis_asymmetric_sha(n_harmonics=4)
        with pytest.raises(ValueError, match="cannot exceed"):
            sha.inverse_transform(flat, theta_range=th, phi_range=ph, l_max=5)

    def test_l_max_negative_raises(self):
        sha, flat, th, ph = self._make_axis_asymmetric_sha(n_harmonics=4)
        with pytest.raises(ValueError, match=">= 0"):
            sha.inverse_transform(flat, theta_range=th, phi_range=ph, l_max=-1)


class TestSHAFlatRoundTripAxisOrder:
    """Regression test for axis-order layout bug in inverse_transform.

    Uses an axis-asymmetric surface so that mis-ordered coefficients
    produce gross reconstruction errors, not subtle ones.  Before the
    fix, ``inverse_transform`` interpreted the flat output of
    ``transform`` as index-major instead of axis-major and returned
    a reconstruction with ~127% relative error.
    """

    def test_flat_output_round_trip(self):
        l_max = 6
        n_th, n_ph = 60, 120
        th = np.linspace(0.01, np.pi - 0.01, n_th)
        ph = np.linspace(0, 2 * np.pi, n_ph, endpoint=False)
        theta_grid, phi_grid = np.meshgrid(th, ph, indexing="ij")
        tf, pf = theta_grid.ravel(), phi_grid.ravel()

        # Distinct per-axis scales so any axis-mixup is detectable.
        r = 1.0 + 0.1 * np.sin(2 * tf) * np.cos(3 * pf)
        x = 1.0 * r * np.sin(tf) * np.cos(pf)
        y = 0.5 * r * np.sin(tf) * np.sin(pf)
        z = 2.0 * r * np.cos(tf)
        surface = np.stack([x, y, z], axis=-1)
        theta_phi = np.stack([tf, pf], axis=-1)

        sha = SphericalHarmonicAnalysis(n_harmonics=l_max, n_jobs=1)
        flat = sha.transform([surface], theta_phi=[theta_phi])

        # Pass the flat output directly (no manual reshape).
        recon = sha.inverse_transform(flat, theta_range=th, phi_range=ph)[0]
        sg = surface.reshape(n_th, n_ph, 3)

        rel_err = np.linalg.norm(recon - sg) / np.linalg.norm(sg)
        assert rel_err < 0.05, (
            f"flat round-trip relative error too large: {rel_err:.3e}"
        )

        # Per-axis range must also be preserved within the truncation tolerance.
        for ax, expected in zip(range(3), [1.0, 0.5, 2.0]):
            obs = max(-recon[..., ax].min(), recon[..., ax].max())
            assert abs(obs - expected) / expected < 0.1, (
                f"axis {ax}: reconstructed range {obs:.3f} far from {expected}"
            )


class TestSHANonFiniteInput:
    """Tests for NaN/Inf input to SHA.

    scipy.linalg.lstsq raises ValueError for non-finite input.
    """

    def test_nan_input_raises(self):
        """SHA with NaN in coordinates raises ValueError."""
        theta_phi = np.column_stack(
            [
                np.linspace(0.1, np.pi - 0.1, 10),
                np.linspace(0, 2 * np.pi, 10),
            ]
        )
        coords = np.ones((10, 3))
        coords[3, 0] = np.nan

        sha = SphericalHarmonicAnalysis(n_harmonics=2, n_jobs=1)
        with pytest.raises(ValueError, match="infs or NaNs"):
            sha.fit_transform([coords], theta_phi=[theta_phi])

    def test_inf_input_raises(self):
        """SHA with Inf in coordinates raises ValueError."""
        theta_phi = np.column_stack(
            [
                np.linspace(0.1, np.pi - 0.1, 10),
                np.linspace(0, 2 * np.pi, 10),
            ]
        )
        coords = np.ones((10, 3))
        coords[5, 1] = np.inf

        sha = SphericalHarmonicAnalysis(n_harmonics=2, n_jobs=1)
        with pytest.raises(ValueError, match="infs or NaNs"):
            sha.fit_transform([coords], theta_phi=[theta_phi])


#
# Real spherical harmonic basis and coefficient conversion
#


class TestRealSHOrthonormality:
    """Verify the real SH basis is orthonormal on the sphere."""

    @pytest.mark.parametrize("l_max,n_theta,n_phi", [(2, 50, 100), (5, 80, 160)])
    def test_orthonormality(self, l_max, n_theta, n_phi):
        nodes, weights = np.polynomial.legendre.leggauss(n_theta)
        theta_gl = np.arccos(nodes)
        phi_uniform = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        dphi = 2 * np.pi / n_phi

        theta_grid, phi_grid = np.meshgrid(theta_gl, phi_uniform, indexing="ij")
        B = _real_sph_harm_basis_matrix(l_max, theta_grid.ravel(), phi_grid.ravel())
        w_grid = np.outer(weights, np.full(n_phi, dphi)).ravel()
        gram = B.T @ np.diag(w_grid) @ B

        n_coeffs = (l_max + 1) ** 2
        assert_allclose(gram, np.eye(n_coeffs), atol=1e-12)


class TestComplexRealSphCoefRoundtrip:
    """complex -> real -> complex and real -> complex -> real."""

    def test_complex_to_real_to_complex(self):
        rng = np.random.RandomState(99)
        l_max = 4
        n_coeffs = (l_max + 1) ** 2

        coef_complex = np.zeros((n_coeffs, 3), dtype=np.complex128)
        for l in range(l_max + 1):
            idx_0 = l**2 + l
            coef_complex[idx_0] = rng.randn(3)
            for m in range(1, l + 1):
                idx_pos = l**2 + l + m
                idx_neg = l**2 + l - m
                c = rng.randn(3) + 1j * rng.randn(3)
                coef_complex[idx_pos] = c
                coef_complex[idx_neg] = (-1) ** m * np.conj(c)

        coef_real = _complex_to_real_sph_coef(coef_complex)
        coef_complex_rt = _real_to_complex_sph_coef(coef_real)
        assert_allclose(coef_complex_rt, coef_complex, atol=1e-14)

    def test_real_to_complex_to_real(self):
        rng = np.random.RandomState(77)
        l_max = 4
        n_coeffs = (l_max + 1) ** 2
        coef_real = rng.randn(n_coeffs, 3)

        coef_complex = _real_to_complex_sph_coef(coef_real)
        coef_real_rt = _complex_to_real_sph_coef(coef_complex)
        assert_allclose(coef_real_rt, coef_real, atol=1e-14)
