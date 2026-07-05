import numpy as np
import pytest
import scipy as sp
from numpy.testing import assert_allclose, assert_array_almost_equal

from ktch.harmonic import (
    SphericalHarmonicAnalysis,
    SphericalHarmonicRegistration,
    spharm,
    xyz2spherical,
)
from ktch.harmonic._spherical_harmonic_analysis import (
    _WIGNER_D_LMAX,
    _axis_third_moment_signs,
    _complex_to_real_sph_coef,
    _real_sph_harm_basis_matrix,
    _real_to_complex_sph_coef,
    _third_moment_grid_basis,
    _wigner_d_small,
    cvt_spharm_coef_to_list,
    rotate_real_sph_coef,
)


def _spherical_to_xyz(theta_phi):
    theta, phi = theta_phi[:, 0], theta_phi[:, 1]
    return np.column_stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )


class TestRotateRealSphCoef:
    """Coefficient-domain SH rotation == rotate parameterization + re-fit."""

    def _setup(self, seed):
        rng = np.random.default_rng(seed)
        l_max = 3
        n_coeffs = (l_max + 1) ** 2
        theta = np.arccos(rng.uniform(-1, 1, 600))
        phi = rng.uniform(0, 2 * np.pi, 600)
        theta_phi = np.column_stack([theta, phi])
        basis = _real_sph_harm_basis_matrix(l_max, theta, phi)
        coords = basis @ rng.standard_normal((n_coeffs, 3))
        coef = sp.linalg.lstsq(basis, coords)[0]  # (n_coeffs, 3)
        return l_max, theta_phi, coords, coef

    def _refit_after_param_rotation(self, l_max, theta_phi, coords, R):
        xyz = _spherical_to_xyz(theta_phi)
        tp_rot = xyz2spherical(xyz @ R.T)  # point' = R point
        b_rot = _real_sph_harm_basis_matrix(l_max, tp_rot[:, 0], tp_rot[:, 1])
        return sp.linalg.lstsq(b_rot, coords)[0]

    def test_matches_refit_proper(self):
        l_max, theta_phi, coords, coef = self._setup(0)
        R = sp.spatial.transform.Rotation.from_rotvec([0.3, -0.5, 0.8]).as_matrix()
        expected = self._refit_after_param_rotation(l_max, theta_phi, coords, R)
        got = rotate_real_sph_coef(coef, R)
        assert_allclose(got, expected, atol=1e-7)

    def test_matches_refit_improper(self):
        # Reflection (det = -1) handled via parity.
        l_max, theta_phi, coords, coef = self._setup(1)
        R = sp.spatial.transform.Rotation.from_rotvec([0.2, 0.4, -0.1]).as_matrix()
        R = R @ np.diag([1.0, 1.0, -1.0])  # make it improper
        assert np.linalg.det(R) < 0
        expected = self._refit_after_param_rotation(l_max, theta_phi, coords, R)
        got = rotate_real_sph_coef(coef, R)
        assert_allclose(got, expected, atol=1e-7)

    def test_identity(self):
        _, _, _, coef = self._setup(2)
        got = rotate_real_sph_coef(coef, np.eye(3))
        assert_allclose(got, coef, atol=1e-10)

    def test_matches_refit_higher_degree(self):
        # The convention is degree-independent; confirm it holds (and stays
        # numerically sound) at a higher l_max.
        rng = np.random.default_rng(20)
        l_max = 8
        n_coeffs = (l_max + 1) ** 2
        theta = np.arccos(rng.uniform(-1, 1, 2000))
        phi = rng.uniform(0, 2 * np.pi, 2000)
        theta_phi = np.column_stack([theta, phi])
        basis = _real_sph_harm_basis_matrix(l_max, theta, phi)
        coords = basis @ rng.standard_normal((n_coeffs, 3))
        coef = sp.linalg.lstsq(basis, coords)[0]

        R = sp.spatial.transform.Rotation.from_rotvec([0.6, -0.2, 0.9]).as_matrix()
        expected = self._refit_after_param_rotation(l_max, theta_phi, coords, R)
        got = rotate_real_sph_coef(coef, R)
        assert_allclose(got, expected, atol=1e-6)

    def test_inverse_rotation_round_trip(self):
        _, _, _, coef = self._setup(5)
        R = sp.spatial.transform.Rotation.from_rotvec([0.4, 0.7, -0.3]).as_matrix()
        back = rotate_real_sph_coef(rotate_real_sph_coef(coef, R), R.T)
        assert_allclose(back, coef, atol=1e-9)

    def test_1d_input(self):
        _, _, _, coef = self._setup(3)
        R = sp.spatial.transform.Rotation.from_rotvec([0.1, 0.2, 0.3]).as_matrix()
        got = rotate_real_sph_coef(coef[:, 0], R)
        assert got.shape == (coef.shape[0],)

    def test_composition(self):
        # Group homomorphism: rotating by R1 then R2 equals rotating by R2 @ R1.
        # Guards against convention drift in the Wigner-D / real-complex layers.
        _, _, _, coef = self._setup(7)
        R1 = sp.spatial.transform.Rotation.from_rotvec([0.5, -0.2, 0.8]).as_matrix()
        R2 = sp.spatial.transform.Rotation.from_rotvec([0.1, 0.9, -0.4]).as_matrix()
        composed = rotate_real_sph_coef(rotate_real_sph_coef(coef, R1), R2)
        direct = rotate_real_sph_coef(coef, R2 @ R1)
        assert_allclose(composed, direct, atol=1e-9)


class TestWignerDSmall:
    """Wigner small-d invariants (guards the factorial-table form)."""

    @pytest.mark.parametrize("l", [0, 1, 3, 8, 20])
    @pytest.mark.parametrize("beta", [0.0, 0.7, np.pi / 3, np.pi])
    def test_orthogonal(self, l, beta):
        # Real small-d is orthogonal (d^T d = I) only if the series is correct.
        d = _wigner_d_small(l, beta)
        assert_allclose(d.T @ d, np.eye(2 * l + 1), atol=1e-9)

    def test_above_lmax_raises(self):
        with pytest.raises(NotImplementedError, match="exceeds the float64-safe"):
            _wigner_d_small(_WIGNER_D_LMAX + 1, 0.5)


class TestThirdMomentGridBasis:
    """Cached third-moment grid basis shared across samples."""

    def test_matches_fresh_build(self):
        l_max, n_theta, n_phi = 6, 30, 60
        basis, weights = _third_moment_grid_basis(l_max, n_theta, n_phi)
        theta_g = np.linspace(0.0, np.pi, n_theta)
        phi_g = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
        tg, pg = np.meshgrid(theta_g, phi_g, indexing="ij")
        expected = _real_sph_harm_basis_matrix(l_max, tg.ravel(), pg.ravel())
        assert_allclose(basis, expected)
        assert_allclose(weights, np.sin(tg).ravel())

    def test_read_only(self):
        basis, weights = _third_moment_grid_basis(5, 30, 60)
        assert not basis.flags.writeable
        assert not weights.flags.writeable

    def test_memoized_identity(self):
        a = _third_moment_grid_basis(4, 30, 60)
        b = _third_moment_grid_basis(4, 30, 60)
        assert a[0] is b[0]
        assert a[1] is b[1]


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

    sha = SphericalHarmonicAnalysis(n_harmonics=l_max, registration=None, n_jobs=1)
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
    n_coords = 5  # < (l_max+1)**2 = 16

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

        sha = SphericalHarmonicAnalysis(n_harmonics=l_max, registration=None, n_jobs=1)
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

        sha = SphericalHarmonicAnalysis(n_harmonics=l_max, registration=None, n_jobs=1)
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
        trunc = sha.inverse_transform(flat, theta_range=th, phi_range=ph, l_max=3)[0]
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

        sha = SphericalHarmonicAnalysis(n_harmonics=l_max, registration=None, n_jobs=1)
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


def _synthetic_sphere(l_max, n_points, n_dim, seed=0):
    """Generate a synthetic R^D-valued field on the sphere with known coefs.

    Returns (X, theta_phi, coef_true) where coef_true is the flat real
    coefficient vector of shape (n_dim*(l_max+1)^2,).
    """
    rng = np.random.default_rng(seed)
    n_coeffs = (l_max + 1) ** 2
    coef_matrix = rng.standard_normal((n_coeffs, n_dim))

    # Roughly uniform sampling on the sphere.
    theta = np.arccos(rng.uniform(-1.0, 1.0, n_points))
    phi = rng.uniform(0.0, 2.0 * np.pi, n_points)

    B = _real_sph_harm_basis_matrix(l_max, theta, phi)
    X = B @ coef_matrix

    theta_phi = np.column_stack([theta, phi])
    return X, theta_phi, coef_matrix.T.ravel()


class TestSPHARMNDim:
    """Codomain generalization to dims other than 3 (incl. scalar field)."""

    def test_default_n_dim_is_3(self):
        assert SphericalHarmonicAnalysis().n_dim == 3

    def test_scalar_field_round_trip(self):
        """n_dim=1 (scalar field on the sphere) recovers exact coefficients."""
        l_max = 3
        X, theta_phi, coef_true = _synthetic_sphere(l_max, 400, n_dim=1)
        sha = SphericalHarmonicAnalysis(n_harmonics=l_max, n_dim=1, n_jobs=1)
        transformed = sha.fit_transform([X], theta_phi=[theta_phi])

        assert transformed.shape == (1, (l_max + 1) ** 2)
        assert_allclose(transformed[0], coef_true, atol=1e-8)

    def test_4d_round_trip(self):
        l_max = 2
        X, theta_phi, coef_true = _synthetic_sphere(l_max, 400, n_dim=4)
        sha = SphericalHarmonicAnalysis(n_harmonics=l_max, n_dim=4, n_jobs=1)
        transformed = sha.fit_transform([X], theta_phi=[theta_phi])

        assert transformed.shape == (1, 4 * (l_max + 1) ** 2)
        assert_allclose(transformed[0], coef_true, atol=1e-8)

    def test_inverse_scalar_field_round_trip(self):
        """Grid-based transform -> inverse round trip for a scalar field."""
        l_max = 2
        theta_range = np.linspace(0, np.pi, 7)
        phi_range = np.linspace(0, 2 * np.pi, 9)

        rng = np.random.default_rng(1)
        coef_list = [rng.standard_normal((2 * l + 1, 1)) for l in range(l_max + 1)]

        coords_tuple = spharm(
            l_max, coef_list, theta_range=theta_range, phi_range=phi_range
        )
        assert len(coords_tuple) == 1
        scalar = coords_tuple[0]

        theta_grid, phi_grid = np.meshgrid(theta_range, phi_range, indexing="ij")
        theta_phi = np.stack([theta_grid.ravel(), phi_grid.ravel()], axis=-1)
        coords = scalar.reshape(-1, 1)

        sha = SphericalHarmonicAnalysis(n_harmonics=l_max, n_dim=1, n_jobs=1)
        transformed = sha.fit_transform([coords], theta_phi=[theta_phi])
        assert transformed.shape == (1, (l_max + 1) ** 2)

        reconstructed = sha.inverse_transform(
            transformed, theta_range=theta_range, phi_range=phi_range
        )
        assert reconstructed.shape == (
            1,
            len(theta_range),
            len(phi_range),
            1,
        )
        assert_array_almost_equal(reconstructed[0, ..., 0], scalar, decimal=6)

    def test_spharm_returns_tuple_len_d(self):
        l_max = 1
        coef_list = [np.zeros((2 * l + 1, 2)) for l in range(l_max + 1)]
        coef_list[0][0] = [1.0, -0.5]
        out = spharm(
            l_max, coef_list, np.linspace(0, np.pi, 4), np.linspace(0, 2 * np.pi, 5)
        )
        assert isinstance(out, tuple)
        assert len(out) == 2

    def test_feature_names_scalar(self):
        sha = SphericalHarmonicAnalysis(n_harmonics=1, n_dim=1)
        names = list(sha.get_feature_names_out())
        assert names == ["cx_0_0", "cx_1_-1", "cx_1_0", "cx_1_1"]

    def test_feature_names_4d(self):
        sha = SphericalHarmonicAnalysis(n_harmonics=1, n_dim=4)
        names = list(sha.get_feature_names_out())
        assert len(names) == 4 * 4
        assert names[0] == "c0_0_0"
        assert names[4] == "c1_0_0"

    def test_n_features_out_4d(self):
        sha = SphericalHarmonicAnalysis(n_harmonics=2, n_dim=4)
        assert sha._n_features_out == 4 * 9

    def test_data_dim_mismatch_raises(self):
        sha = SphericalHarmonicAnalysis(n_harmonics=2, n_dim=3, n_jobs=1)
        with pytest.raises(ValueError, match="n_dim=3"):
            sha.transform([np.zeros((10, 2))], theta_phi=[np.zeros((10, 2))])


class TestSPHARMRegistration:
    """Tests for registration modes of SphericalHarmonicAnalysis."""

    def test_default_is_auto(self):
        assert SphericalHarmonicAnalysis().registration == "auto"

    def test_moment_shape(self):
        l_max = 2
        X, theta_phi, _ = _synthetic_sphere(l_max, 300, n_dim=3, seed=3)
        sha = SphericalHarmonicAnalysis(
            n_harmonics=l_max, n_dim=3, registration="moment", n_jobs=1
        )
        out = sha.fit_transform([X], theta_phi=[theta_phi])
        assert out.shape == (1, 3 * (l_max + 1) ** 2)

    def test_moment_invariance(self):
        l_max = 2
        X, theta_phi, _ = _synthetic_sphere(l_max, 400, n_dim=3, seed=5)
        sha = SphericalHarmonicAnalysis(
            n_harmonics=l_max, n_dim=3, registration="moment", n_jobs=1
        )
        c1 = sha.fit_transform([X], theta_phi=[theta_phi])

        rng = np.random.default_rng(6)
        A = rng.standard_normal((3, 3))
        R, _ = np.linalg.qr(A)
        if np.linalg.det(R) < 0:
            R[:, 0] = -R[:, 0]
        X2 = 1.7 * (X @ R.T) + np.array([2.0, -1.0, 0.5])
        c2 = sha.fit_transform([X2], theta_phi=[theta_phi])
        assert_allclose(c1, c2, atol=1e-7)

    def test_first_order_shape(self):
        l_max = 3
        X, theta_phi, _ = _synthetic_sphere(l_max, 400, n_dim=3, seed=2)
        sha = SphericalHarmonicAnalysis(
            n_harmonics=l_max, n_dim=3, registration="first_order", n_jobs=1
        )
        out = sha.fit_transform([X], theta_phi=[theta_phi])
        assert out.shape == (1, 3 * (l_max + 1) ** 2)
        # constant (l=0) mode removed
        assert_allclose(out.reshape(3, -1)[:, 0], 0.0, atol=1e-10)

    def test_first_order_ellipsoid_canonical_form(self):
        # Theory (Brechbühler 1995): after first_order registration the
        # degree-1 ellipsoid is axis-aligned in object space (diagonal), with
        # descending positive semi-axes; with semi_major_axis scaling the
        # leading entry is 1.
        l_max = 3
        n_coeffs = (l_max + 1) ** 2
        X, theta_phi, _ = _synthetic_sphere(l_max, 500, n_dim=3, seed=3)
        sha = SphericalHarmonicAnalysis(
            n_harmonics=l_max, n_dim=3, registration="first_order", n_jobs=1
        )
        out = sha.fit_transform([X], theta_phi=[theta_phi])[0].reshape(3, n_coeffs)
        m1_xyz = out[:, [1, 2, 3]][:, [2, 0, 1]]  # l=1 columns x, y, z
        diag = np.diag(m1_xyz)
        assert_allclose(m1_xyz - np.diag(diag), 0.0, atol=1e-9)  # diagonal
        assert diag[0] == pytest.approx(1.0, abs=1e-9)  # semi_major scaled
        assert diag[0] >= diag[1] >= diag[2] > 0  # descending, positive

    def test_first_order_positive_skewness(self):
        # Klein-four disambiguation makes the per-axis third moment positive;
        # the first two axes keep positive skewness (the third is then fixed by
        # det=+1, either sign).
        l_max = 3
        n_coeffs = (l_max + 1) ** 2
        X, theta_phi, _ = _synthetic_sphere(l_max, 500, n_dim=3, seed=3)
        sha = SphericalHarmonicAnalysis(
            n_harmonics=l_max, n_dim=3, registration="first_order", n_jobs=1
        )
        out = sha.fit_transform([X], theta_phi=[theta_phi])[0].reshape(3, n_coeffs)
        signs = _axis_third_moment_signs(out, np.eye(3), l_max)
        assert signs[0] > 0  # positive skewness along x (no flip needed)
        assert signs[1] > 0  # positive skewness along y

    def test_first_order_codomain_invariance(self):
        l_max = 3
        X, theta_phi, _ = _synthetic_sphere(l_max, 600, n_dim=3, seed=4)
        sha = SphericalHarmonicAnalysis(
            n_harmonics=l_max, n_dim=3, registration="first_order", n_jobs=1
        )
        c1 = sha.fit_transform([X], theta_phi=[theta_phi])

        rng = np.random.default_rng(12)
        Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        X2 = 1.7 * (X @ Q.T) + np.array([2.0, -1.0, 0.5])
        c2 = sha.fit_transform([X2], theta_phi=[theta_phi])
        assert_allclose(c1, c2, atol=1e-6)

    def test_first_order_parameter_so3_invariance(self):
        # Rotating the sphere parameterization (SO(3)) and re-fitting must give
        # the same registered coefficients (group B).
        l_max = 3
        X, theta_phi, _ = _synthetic_sphere(l_max, 800, n_dim=3, seed=7)
        sha = SphericalHarmonicAnalysis(
            n_harmonics=l_max, n_dim=3, registration="first_order", n_jobs=1
        )
        c1 = sha.fit_transform([X], theta_phi=[theta_phi])

        rng = np.random.default_rng(13)
        R, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(R) < 0:
            R[:, 0] = -R[:, 0]
        theta, phi = theta_phi[:, 0], theta_phi[:, 1]
        xyz = np.column_stack(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )
        theta_phi_rot = xyz2spherical(xyz @ R.T)
        c2 = sha.fit_transform([X], theta_phi=[theta_phi_rot])
        assert_allclose(c1, c2, atol=1e-6)

    def test_first_order_requires_3d(self):
        X, theta_phi, _ = _synthetic_sphere(2, 200, n_dim=2, seed=1)
        sha = SphericalHarmonicAnalysis(
            n_harmonics=2, n_dim=2, registration="first_order", n_jobs=1
        )
        with pytest.raises(ValueError, match="n_dim=3"):
            sha.fit_transform([X], theta_phi=[theta_phi])

    def test_reserved_not_implemented(self):
        sha = SphericalHarmonicAnalysis(n_harmonics=2, registration="rotational_match")
        with pytest.raises(NotImplementedError, match="reserved"):
            sha.transform([np.zeros((10, 3))], theta_phi=[np.zeros((10, 2))])

    def test_return_transform_not_implemented(self):
        sha = SphericalHarmonicAnalysis(n_harmonics=2, return_transform=True)
        with pytest.raises(NotImplementedError, match="return_transform"):
            sha.transform([np.zeros((10, 3))], theta_phi=[np.zeros((10, 2))])

    def test_align_parameter_false_not_implemented(self):
        sha = SphericalHarmonicAnalysis(n_harmonics=2, align_parameter=False)
        with pytest.raises(NotImplementedError, match="align_parameter"):
            sha.transform([np.zeros((10, 3))], theta_phi=[np.zeros((10, 2))])

    def test_reflect_first_order_allowed(self):
        # reflect=True IS implemented for SPHARM first_order; it must not be gated.
        l_max = 3
        X, theta_phi, _ = _synthetic_sphere(l_max, 400, n_dim=3, seed=2)
        sha = SphericalHarmonicAnalysis(
            n_harmonics=l_max,
            n_dim=3,
            registration="first_order",
            reflect=True,
            n_jobs=1,
        )
        out = sha.fit_transform([X], theta_phi=[theta_phi])
        assert out.shape == (1, 3 * (l_max + 1) ** 2)

    def test_invalid_scale_method_for_moment(self):
        sha = SphericalHarmonicAnalysis(
            n_harmonics=2, registration="moment", scale_method="ellipsoid_volume"
        )
        # ellipsoid_volume is a first_order measure, not valid for moment
        with pytest.raises(ValueError, match="not valid for"):
            sha.transform([np.zeros((10, 3))], theta_phi=[np.zeros((10, 2))])


class TestSphericalHarmonicRegistration:
    """Public coefficient-only registration transformer."""

    def _raw_coeffs(self, l_max, n_dim=3, seed=0, n_samples=3):
        """Batch of raw (unregistered) SPHARM coefficient vectors."""
        rows = []
        for s in range(n_samples):
            X, theta_phi, _ = _synthetic_sphere(l_max, 500, n_dim=n_dim, seed=seed + s)
            sha = SphericalHarmonicAnalysis(
                n_harmonics=l_max, n_dim=n_dim, registration=None, n_jobs=1
            )
            rows.append(sha.transform([X], theta_phi=[theta_phi])[0])
        return np.stack(rows)

    def test_first_order_parity_with_analysis(self):
        # Registering raw coefficients must reproduce the analysis estimator's
        # internal first_order registration bit-for-bit (shared implementation).
        l_max = 3
        X, theta_phi, _ = _synthetic_sphere(l_max, 500, n_dim=3, seed=1)
        raw = SphericalHarmonicAnalysis(
            n_harmonics=l_max, registration=None, n_jobs=1
        ).transform([X], theta_phi=[theta_phi])
        registered = SphericalHarmonicAnalysis(
            n_harmonics=l_max, registration="first_order", scale=False, n_jobs=1
        ).transform([X], theta_phi=[theta_phi])
        out = SphericalHarmonicRegistration(
            method="first_order", scale=False
        ).fit_transform(raw)
        assert_allclose(out, registered, atol=1e-12)

    def test_moment_parity_with_analysis(self):
        l_max = 3
        X, theta_phi, _ = _synthetic_sphere(l_max, 500, n_dim=3, seed=2)
        raw = SphericalHarmonicAnalysis(
            n_harmonics=l_max, registration=None, n_jobs=1
        ).transform([X], theta_phi=[theta_phi])
        registered = SphericalHarmonicAnalysis(
            n_harmonics=l_max, registration="moment", n_jobs=1
        ).transform([X], theta_phi=[theta_phi])
        out = SphericalHarmonicRegistration(method="moment").fit_transform(raw)
        assert_allclose(out, registered, atol=1e-12)

    def test_none_is_passthrough(self):
        raw = self._raw_coeffs(3, seed=10, n_samples=2)
        out = SphericalHarmonicRegistration(method=None).fit_transform(raw)
        assert_allclose(out, raw, atol=0)

    def test_auto_resolves_first_order_for_3d(self):
        raw = self._raw_coeffs(3, seed=11, n_samples=2)
        auto = SphericalHarmonicRegistration(method="auto", scale=False).fit_transform(
            raw
        )
        first = SphericalHarmonicRegistration(
            method="first_order", scale=False
        ).fit_transform(raw)
        assert_allclose(auto, first, atol=1e-12)

    def test_lmax_inferred_from_width(self):
        for l_max in (1, 2, 4):
            raw = self._raw_coeffs(l_max, seed=20, n_samples=1)
            reg = SphericalHarmonicRegistration(method="first_order", scale=False).fit(
                raw
            )
            assert reg._l_max == l_max
            assert reg.n_features_in_ == 3 * (l_max + 1) ** 2

    def test_translation_removed(self):
        raw = self._raw_coeffs(3, seed=21, n_samples=2)
        out = SphericalHarmonicRegistration(
            method="first_order", scale=False
        ).fit_transform(raw)
        mat = out.reshape(out.shape[0], 3, -1)
        assert_allclose(mat[:, :, 0], 0.0, atol=1e-12)  # l=0 (constant) mode

    def test_scale_false_preserves_amplitude_spectrum(self):
        # scale=False keeps size; the per-degree amplitude ||c_l|| is invariant
        # under the orthogonal group-A / group-B rotations registration applies.
        raw = self._raw_coeffs(3, seed=22, n_samples=1)
        out = SphericalHarmonicRegistration(
            method="first_order", scale=False
        ).fit_transform(raw)

        def spectrum(v):
            m = v.reshape(3, -1)
            return np.array(
                [np.linalg.norm(m[:, l * l : (l + 1) ** 2]) for l in range(1, 4)]
            )

        assert_allclose(spectrum(out[0]), spectrum(raw[0]), rtol=1e-8)

    def test_feature_names_preserved(self):
        raw = self._raw_coeffs(2, seed=23, n_samples=1)
        reg = SphericalHarmonicRegistration(method="first_order", scale=False).fit(raw)
        assert len(reg.get_feature_names_out()) == 3 * (2 + 1) ** 2

    def test_pipeline_with_pca(self):
        from sklearn.decomposition import PCA
        from sklearn.pipeline import make_pipeline

        raw = self._raw_coeffs(3, seed=24, n_samples=6)
        pipe = make_pipeline(
            SphericalHarmonicRegistration(method="first_order", scale=False),
            PCA(n_components=2),
        )
        assert pipe.fit_transform(raw).shape == (6, 2)

    def test_transform_before_fit_raises(self):
        from sklearn.exceptions import NotFittedError

        raw = self._raw_coeffs(2, seed=25, n_samples=1)
        with pytest.raises(NotFittedError):
            SphericalHarmonicRegistration(method="first_order").transform(raw)

    @pytest.mark.parametrize("method", ["landmark", "rotational_match"])
    def test_reserved_methods_raise(self, method):
        raw = self._raw_coeffs(2, seed=26, n_samples=1)
        with pytest.raises(NotImplementedError, match="reserved"):
            SphericalHarmonicRegistration(method=method).fit(raw)

    def test_return_transform_reserved(self):
        raw = self._raw_coeffs(2, seed=27, n_samples=1)
        with pytest.raises(NotImplementedError, match="return_transform"):
            SphericalHarmonicRegistration(
                method="first_order", return_transform=True
            ).fit(raw)

    def test_align_parameter_false_reserved(self):
        raw = self._raw_coeffs(2, seed=28, n_samples=1)
        with pytest.raises(NotImplementedError, match="align_parameter"):
            SphericalHarmonicRegistration(
                method="first_order", align_parameter=False
            ).fit(raw)

    def test_first_order_requires_ndim3(self):
        # Valid n_dim=2 width so l_max inference passes before the n_dim check.
        raw = np.zeros((2, 2 * (3 + 1) ** 2))
        with pytest.raises(ValueError, match="requires n_dim=3"):
            SphericalHarmonicRegistration(method="first_order", n_dim=2).fit(raw)

    def test_invalid_scale_method(self):
        raw = self._raw_coeffs(2, seed=29, n_samples=1)
        with pytest.raises(ValueError, match="not valid for"):
            SphericalHarmonicRegistration(
                method="first_order", scale_method="centroid_size"
            ).fit(raw)

    def test_invalid_method(self):
        raw = self._raw_coeffs(2, seed=30, n_samples=1)
        with pytest.raises(ValueError, match="registration must be one of"):
            SphericalHarmonicRegistration(method="bogus").fit(raw)

    def test_bad_width_raises(self):
        with pytest.raises(ValueError, match="not divisible by n_dim"):
            SphericalHarmonicRegistration(method=None).fit(np.zeros((2, 47)))
        with pytest.raises(ValueError, match="perfect square"):
            SphericalHarmonicRegistration(method="first_order").fit(np.zeros((2, 15)))

    def test_degenerate_ellipsoid_raises(self):
        l_max = 3
        v = np.random.default_rng(0).standard_normal(3 * (l_max + 1) ** 2)
        mat = v.reshape(3, -1)
        mat[:, 1:4] = 0.0  # zero the l=1 ellipsoid
        with pytest.raises(ValueError, match="[Dd]egenerate"):
            SphericalHarmonicRegistration(method="first_order").fit_transform(
                mat.ravel()[None, :]
            )
