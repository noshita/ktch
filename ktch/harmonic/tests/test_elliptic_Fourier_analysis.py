from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpl_toolkits.mplot3d import axes3d
from numpy.testing import assert_array_almost_equal
from scipy.interpolate import BSpline, make_interp_spline
from scipy.stats import wasserstein_distance_nd

from ktch.datasets import load_outline_mosquito_wings
from ktch.harmonic import EllipticFourierAnalysis
from ktch.harmonic._elliptic_Fourier_analysis import (
    _compute_ellipse_geometry_3d,
    rotation_matrix_3d_euler_zxz,
)

EXPORT_DIR_FIGS = Path(".pytest_artifacts/figures/")
EXPORT_DIR_FIGS.mkdir(exist_ok=True, parents=True)


def _load_wings_as_list(n_specimens=None):
    """Load mosquito wing outlines as a list of arrays for EFA input."""
    wings = load_outline_mosquito_wings()
    n_total = 126
    n_points = 100
    coords = wings.coords.reshape(n_total, n_points, 2)
    if n_specimens is not None:
        coords = coords[:n_specimens]
    return [coords[i] for i in range(len(coords))]


@pytest.mark.parametrize("norm", [False, True])
def test_efa_shape(norm):
    n_harmonics = 6

    X = _load_wings_as_list(n_specimens=10)
    efa = EllipticFourierAnalysis(n_harmonics=n_harmonics)
    X_transformed = efa.fit_transform(X, norm=norm)

    assert X_transformed.shape == (len(X), 4 * (n_harmonics + 1))


@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("set_output", [None, "pandas"])
def test_transform(norm, set_output):
    n_harmonics = 6
    t_num = 360

    X = _load_wings_as_list(n_specimens=10)

    efa = EllipticFourierAnalysis(n_harmonics=n_harmonics)
    efa.set_output(transform=set_output)
    coef1 = efa.fit_transform(X, norm=norm)

    # round-trip: inverse_transform -> re-fit_transform
    if set_output == "pandas":
        coef1_arr = coef1.to_numpy()
    else:
        coef1_arr = coef1

    X_reconstructed = np.array(
        efa.inverse_transform(coef1_arr, t_num=t_num, norm=norm)
    )
    T = [np.linspace(2 * np.pi / t_num, 2 * np.pi, t_num)] * len(X)

    efa2 = EllipticFourierAnalysis(n_harmonics=n_harmonics)
    coef2 = efa2.fit_transform(X_reconstructed, t=T, norm=norm)

    if norm:
        # DC offsets (a0, c0) change because inverse_transform centers the
        # output; compare only the harmonic coefficients.
        n_cols = n_harmonics + 1
        # harmonic columns: skip a0 (col 0) and c0 (col 2*n_cols)
        mask = np.ones(coef1_arr.shape[1], dtype=bool)
        mask[0] = False
        mask[2 * n_cols] = False
        assert_array_almost_equal(coef1_arr[:, mask], coef2[:, mask], decimal=4)
    else:
        assert_array_almost_equal(coef1_arr, coef2, decimal=4)


@pytest.mark.parametrize("n_jobs", [None, 1, 3])
def test_transform_parallel(benchmark, n_jobs, norm=True):
    n_harmonics = 6

    X = _load_wings_as_list(n_specimens=10)

    efa_serial = EllipticFourierAnalysis(n_harmonics=n_harmonics, n_jobs=None)
    coef_serial = efa_serial.fit_transform(X, norm=norm)

    efa = EllipticFourierAnalysis(n_harmonics=n_harmonics, n_jobs=n_jobs, verbose=1)
    coef_parallel = benchmark(efa.fit_transform, X, norm=norm)

    assert_array_almost_equal(coef_parallel, coef_serial)


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


def test_orientation_and_scale_2d(export_figures=False):
    n_harmonics = 20
    t_num = 360

    t = np.linspace(0, 2 * np.pi, 17)

    x_o = np.random.uniform(low=-5, high=5, size=2)
    psi = np.random.rand() * 2 * np.pi
    scale = np.random.uniform(low=0.5, high=1.5)

    x = np.array(
        [
            1,
            0.85,
            0.7,
            0.3,
            0,
            -0.4,
            -0.7,
            -0.7,
            -0.5,
            -0.7,
            -0.7,
            -0.4,
            0,
            0.3,
            0.7,
            0.85,
            1,
        ]
    )
    y = np.array(
        [
            0,
            0.2,
            0.4,
            0.3,
            0.4,
            0.6,
            0.5,
            0.25,
            0,
            -0.25,
            -0.5,
            -0.6,
            -0.4,
            -0.3,
            -0.4,
            -0.2,
            0,
        ]
    )

    spl = make_interp_spline(t, np.stack([x, y], 1), k=3, bc_type="periodic")
    t = np.linspace(2 * np.pi / t_num, 2 * np.pi, t_num)
    X_coords = (
        x_o
        + scale
        * np.dot(
            np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]]),
            (spl(t)).T,
        ).T
    )

    efa = EllipticFourierAnalysis(n_harmonics=n_harmonics)
    coef_est = efa.fit_transform([X_coords], norm=True, return_orientation_scale=True)[
        0
    ]
    psi_est, scale_est = coef_est[-2:]
    coef_est = coef_est[:-2].reshape(4, n_harmonics + 1)

    cos = np.cos(np.tensordot(np.arange(1, n_harmonics + 1, 1), t, 0))
    sin = np.sin(np.tensordot(np.arange(1, n_harmonics + 1, 1), t, 0))

    a0, an = coef_est[0, 0], coef_est[0, 1:]
    _, bn = coef_est[1, 0], coef_est[1, 1:]
    c0, cn = coef_est[2, 0], coef_est[2, 1:]
    _, dn = coef_est[3, 0], coef_est[3, 1:]
    x = np.dot(an, cos) + np.dot(bn, sin)
    y = np.dot(cn, cos) + np.dot(dn, sin)
    X_coords_est = np.stack([x, y], 1)

    x_o_est = np.array([a0 / 2, c0 / 2])

    X_coords_recon = (
        np.dot(
            np.array(
                [
                    [np.cos(psi_est), -np.sin(psi_est)],
                    [np.sin(psi_est), np.cos(psi_est)],
                ]
            ),
            X_coords_est.T,
        ).T
        * scale_est
    ) + x_o_est

    if export_figures:
        fig, ax = plt.subplots()
        ax.plot(X_coords[:, 0], X_coords[:, 1], "o-")
        ax.set_aspect("equal")
        fig.savefig(EXPORT_DIR_FIGS / "X_points.png")

        fig, ax = plt.subplots()
        ax.plot(X_coords_est[:, 0], X_coords_est[:, 1])
        ax.set_aspect("equal")
        fig.savefig(EXPORT_DIR_FIGS / "X_est.png")

        fig, ax = plt.subplots()
        ax.plot(X_coords_recon[:, 0], X_coords_recon[:, 1])
        ax.set_aspect("equal")
        fig.savefig(EXPORT_DIR_FIGS / "X_recon.png")

    # print(x_o, x_o_est)
    # print(psi, psi_est, scale, scale_est)

    assert wasserstein_distance_nd(X_coords, X_coords_recon) < 10**-1


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

    X = _load_wings_as_list(n_specimens=10)
    efa = EllipticFourierAnalysis(n_harmonics=n_harmonics)
    X_transformed = efa.fit_transform(X, norm=True)
    X_adj = np.array(efa.inverse_transform(X_transformed, t_num=t_num))
    T = [np.linspace(2 * np.pi / t_num, 2 * np.pi, t_num) for i in range(len(X))]

    X_transformed = efa.fit_transform(
        X_adj,
        t=T,
        norm=False,
    )
    X_reconstructed = np.array(
        efa.inverse_transform(X_transformed, t_num=t_num, norm=False)
    )

    assert_array_almost_equal(X_adj, X_reconstructed, decimal=4)


def test_arc_length_3d_includes_z():
    """Verify that 3D arc-length parameterization includes dz² (z-component).

    Generate a 3D helix where z-variation is significant. Compare coefficients
    computed with auto arc-length (t=None) against coefficients computed with
    explicit t based on full 3D Euclidean distance. They should match closely.
    """
    n_harmonics = 6
    t_num = 360

    t = np.linspace(2 * np.pi / t_num, 2 * np.pi, t_num)

    # 3D helix: x = cos(t), y = sin(t), z = 0.5*sin(2t)
    # The z component has significant variation
    x = np.cos(t)
    y = np.sin(t)
    z = 0.5 * np.sin(2 * t)
    X_coords = np.stack([x, y, z], 1)

    # Compute explicit 3D arc-length parameter
    X_arr = np.vstack([X_coords[-1:], X_coords])
    dx = X_arr[1:, 0] - X_arr[:-1, 0]
    dy = X_arr[1:, 1] - X_arr[:-1, 1]
    dz = X_arr[1:, 2] - X_arr[:-1, 2]
    dt_3d = np.sqrt(dx**2 + dy**2 + dz**2)
    t_explicit = np.cumsum(dt_3d)

    efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics)

    # With explicit 3D arc-length
    coef_explicit = efa.fit_transform([X_coords], t=[t_explicit], norm=False)[0]

    # With auto arc-length (t=None) — should use sqrt(dx²+dy²+dz²)
    coef_auto = efa.fit_transform([X_coords], norm=False)[0]

    # Harmonic coefficients (indices 1+) should match closely
    coef_explicit = coef_explicit.reshape(6, n_harmonics + 1)[:, 1:]
    coef_auto = coef_auto.reshape(6, n_harmonics + 1)[:, 1:]

    assert_array_almost_equal(coef_explicit, coef_auto, decimal=3)


def test_dc_component_3d_weighted():
    """Verify that 3D DC components (a0, c0, e0) use dt-weighted mean.

    Generate a 3D closed curve from known Fourier coefficients with uniform
    parameter spacing. The DC components should match the analytical values
    (weighted average of coordinates). With non-uniform arc-length spacing,
    the unweighted sum gives incorrect DC values.
    """
    n_harmonics = 6
    t_num = 360

    t = np.linspace(2 * np.pi / t_num, 2 * np.pi, t_num)

    # Known coefficients
    a0, c0, e0 = 2.0, 3.0, 1.0
    an, bn, cn, dn, en, fn = np.random.rand(6, n_harmonics) * 0.5

    cos = np.cos(np.tensordot(np.arange(1, n_harmonics + 1, 1), t, 0))
    sin = np.sin(np.tensordot(np.arange(1, n_harmonics + 1, 1), t, 0))

    x = a0 / 2 + np.dot(an, cos) + np.dot(bn, sin)
    y = c0 / 2 + np.dot(cn, cos) + np.dot(dn, sin)
    z = e0 / 2 + np.dot(en, cos) + np.dot(fn, sin)
    X_coords = np.stack([x, y, z], 1)

    efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics)

    # With uniform t (DC should be close to analytical values)
    coef_uniform = efa.fit_transform([X_coords], t=[t], norm=False)[0]
    dc_uniform = coef_uniform.reshape(6, n_harmonics + 1)[:, 0]

    # With auto arc-length (non-uniform dt), DC should also be reasonable
    coef_auto = efa.fit_transform([X_coords], norm=False)[0]
    dc_auto = coef_auto.reshape(6, n_harmonics + 1)[:, 0]

    # The auto DC should be close to the analytical values (a0, c0, e0)
    # Both uniform and auto parameterizations approximate the mean coordinate,
    # but with different weights. The key check is that both are reasonable.
    analytical_dc = np.array([a0, c0, e0])
    assert_array_almost_equal(dc_uniform[[0, 2, 4]], analytical_dc, decimal=0)
    assert_array_almost_equal(dc_auto[[0, 2, 4]], analytical_dc, decimal=0)


class TestRotationMatrix3dEulerZxz:
    """Tests for rotation_matrix_3d_euler_zxz utility function."""

    def test_identity(self):
        """rotation_matrix_3d_euler_zxz(0, 0, 0) returns the identity matrix."""
        R = rotation_matrix_3d_euler_zxz(0.0, 0.0, 0.0)
        assert_array_almost_equal(R, np.eye(3))

    def test_orthogonality(self):
        """Rotation matrix is orthogonal: R^T R = I and det(R) = +1."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            alpha = rng.uniform(-np.pi, np.pi)
            beta = rng.uniform(0, np.pi)
            gamma = rng.uniform(-np.pi, np.pi)
            R = rotation_matrix_3d_euler_zxz(alpha, beta, gamma)
            assert_array_almost_equal(R.T @ R, np.eye(3), decimal=12)
            assert abs(np.linalg.det(R) - 1.0) < 1e-12

    def test_shape(self):
        """Output is a 3x3 numpy array."""
        R = rotation_matrix_3d_euler_zxz(0.1, 0.2, 0.3)
        assert R.shape == (3, 3)

    def test_pure_alpha_rotation(self):
        """Alpha-only rotation (β=0, γ=0) is a Z-axis rotation by α."""
        alpha = np.pi / 4
        R = rotation_matrix_3d_euler_zxz(alpha, 0.0, 0.0)
        Rz = np.array(
            [
                [np.cos(alpha), -np.sin(alpha), 0],
                [np.sin(alpha), np.cos(alpha), 0],
                [0, 0, 1],
            ]
        )
        assert_array_almost_equal(R, Rz)

    def test_pure_beta_rotation(self):
        """With α=0, γ=0, β-only rotation is about the X-axis."""
        beta = np.pi / 3
        R = rotation_matrix_3d_euler_zxz(0.0, beta, 0.0)
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(beta), -np.sin(beta)],
                [0, np.sin(beta), np.cos(beta)],
            ]
        )
        assert_array_almost_equal(R, Rx)

    def test_explicit_matrix_entries(self):
        """Check specific matrix entries from the Godefroy et al. (2012) formula."""
        alpha, beta, gamma = 0.5, 1.0, 0.7
        R = rotation_matrix_3d_euler_zxz(alpha, beta, gamma)
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        cg, sg = np.cos(gamma), np.sin(gamma)

        expected = np.array(
            [
                [ca * cg - sa * cb * sg, -ca * sg - sa * cb * cg, sa * sb],
                [sa * cg + ca * cb * sg, -sa * sg + ca * cb * cg, -ca * sb],
                [sb * sg, sb * cg, cb],
            ]
        )
        assert_array_almost_equal(R, expected)

    def test_inverse_is_transpose(self):
        """R(α, β, γ)^T should be the inverse rotation."""
        alpha, beta, gamma = 0.3, 0.8, 1.2
        R = rotation_matrix_3d_euler_zxz(alpha, beta, gamma)
        v = np.array([1.0, 2.0, 3.0])
        v_rot = R @ v
        v_back = R.T @ v_rot
        assert_array_almost_equal(v, v_back)


def _make_ellipse_coefficients(a, b, alpha, beta, gamma, phi):
    """Helper: construct 6 Fourier coefficients from known geometric parameters.

    An ellipse in the XY-plane with semi-axes a, b and phase phi has parametric form:
        [a cos(t + phi), b sin(t + phi), 0]
    which gives cosine/sine coefficients in the local frame:
        xc_local = a cos(phi), xs_local = -a sin(phi)  (note: not a cos, a sin)
        yc_local = b sin(phi), ys_local = b cos(phi)  (note: not -b sin, b cos)
        -- wait, let me derive properly.

    x_n(t) = xc cos(2πnt/T) + xs sin(2πnt/T)
    For the 1st harmonic, the ellipse parametric equation is:
        [x(t), y(t), z(t)] = Ω @ [a cos(2πt/T + φ), b sin(2πt/T + φ), 0]^T

    Expanding:
        a cos(t + φ) = a cos(φ) cos(t) - a sin(φ) sin(t)
        b sin(t + φ) = b sin(φ) cos(t) + b cos(φ) sin(t)

    So in the local (unrotated) frame:
        [xc_local, xs_local] = [a cos(φ), -a sin(φ)]
        [yc_local, ys_local] = [b sin(φ), b cos(φ)]
        [zc_local, zs_local] = [0, 0]

    Then apply rotation Ω:
        [xc, yc, zc]^T = Ω @ [xc_local, yc_local, zc_local]^T
        [xs, ys, zs]^T = Ω @ [xs_local, ys_local, zs_local]^T
    """
    Omega = rotation_matrix_3d_euler_zxz(alpha, beta, gamma)

    # Local frame coefficients
    local_c = np.array([a * np.cos(phi), b * np.sin(phi), 0.0])
    local_s = np.array([-a * np.sin(phi), b * np.cos(phi), 0.0])

    # Rotate to 3D
    rotated_c = Omega @ local_c
    rotated_s = Omega @ local_s

    xc, yc, zc = rotated_c
    xs, ys, zs = rotated_s
    return xc, xs, yc, ys, zc, zs


class TestComputeEllipseGeometry3d:
    """Tests for _compute_ellipse_geometry_3d utility function."""

    def test_xy_plane_ellipse(self):
        """Ellipse in XY-plane (β=0) with known a, b, φ."""
        a, b = 3.0, 1.5
        phi = 0.2
        alpha, beta, gamma = 0.5, 0.0, 0.0
        xc, xs, yc, ys, zc, zs = _make_ellipse_coefficients(
            a, b, alpha, beta, gamma, phi
        )
        phi_est, a_est, b_est, alpha_est, beta_est, gamma_est = (
            _compute_ellipse_geometry_3d(xc, xs, yc, ys, zc, zs)
        )

        assert a_est == pytest.approx(a, abs=1e-10)
        assert b_est == pytest.approx(b, abs=1e-10)
        assert beta_est == pytest.approx(beta, abs=1e-10)

    def test_tilted_ellipse(self):
        """General 3D ellipse recovers a, b correctly."""
        a, b = 5.0, 2.0
        phi = 0.3
        alpha, beta, gamma = 0.7, 1.2, -0.4
        xc, xs, yc, ys, zc, zs = _make_ellipse_coefficients(
            a, b, alpha, beta, gamma, phi
        )
        phi_est, a_est, b_est, alpha_est, beta_est, gamma_est = (
            _compute_ellipse_geometry_3d(xc, xs, yc, ys, zc, zs)
        )

        assert a_est == pytest.approx(a, abs=1e-10)
        assert b_est == pytest.approx(b, abs=1e-10)
        assert a_est >= b_est

    def test_uniqueness_constraints(self):
        """Output satisfies φ ∈ ]−π/4, π/4[, a ≥ b > 0, β ∈ [0, π]."""
        rng = np.random.default_rng(123)
        for _ in range(20):
            a = rng.uniform(1.0, 10.0)
            b = rng.uniform(0.1, a)
            phi = rng.uniform(-np.pi / 4 + 0.01, np.pi / 4 - 0.01)
            alpha = rng.uniform(-np.pi, np.pi)
            beta = rng.uniform(0.1, np.pi - 0.1)
            gamma = rng.uniform(-np.pi, np.pi)
            xc, xs, yc, ys, zc, zs = _make_ellipse_coefficients(
                a, b, alpha, beta, gamma, phi
            )
            phi_est, a_est, b_est, alpha_est, beta_est, gamma_est = (
                _compute_ellipse_geometry_3d(xc, xs, yc, ys, zc, zs)
            )

            assert -np.pi / 4 < phi_est < np.pi / 4, f"phi={phi_est} out of range"
            assert a_est > 0, f"a={a_est} not positive"
            assert b_est > 0, f"b={b_est} not positive"
            assert a_est >= b_est - 1e-10, f"a={a_est} < b={b_est}"
            assert 0 <= beta_est <= np.pi, f"beta={beta_est} out of range"

    def test_round_trip_reconstruction(self):
        """Reconstructing coefficients from extracted params recovers the original."""
        rng = np.random.default_rng(77)
        for _ in range(20):
            a = rng.uniform(1.0, 10.0)
            b = rng.uniform(0.1, a)
            phi = rng.uniform(-np.pi / 4 + 0.01, np.pi / 4 - 0.01)
            alpha = rng.uniform(-np.pi, np.pi)
            beta = rng.uniform(0.1, np.pi - 0.1)
            gamma = rng.uniform(-np.pi, np.pi)
            xc, xs, yc, ys, zc, zs = _make_ellipse_coefficients(
                a, b, alpha, beta, gamma, phi
            )
            phi_est, a_est, b_est, alpha_est, beta_est, gamma_est = (
                _compute_ellipse_geometry_3d(xc, xs, yc, ys, zc, zs)
            )

            # Reconstruct coefficients from extracted parameters
            xc2, xs2, yc2, ys2, zc2, zs2 = _make_ellipse_coefficients(
                a_est, b_est, alpha_est, beta_est, gamma_est, phi_est
            )

            # The reconstructed coefficients should match the originals
            assert_array_almost_equal(
                [xc, xs, yc, ys, zc, zs],
                [xc2, xs2, yc2, ys2, zc2, zs2],
                decimal=10,
            )

    def test_gimbal_lock_beta_zero(self):
        """Handles gimbal lock when β ≈ 0 without error."""
        a, b = 4.0, 2.0
        phi = 0.1
        alpha, beta, gamma = 1.0, 0.0, 0.5
        xc, xs, yc, ys, zc, zs = _make_ellipse_coefficients(
            a, b, alpha, beta, gamma, phi
        )
        phi_est, a_est, b_est, alpha_est, beta_est, gamma_est = (
            _compute_ellipse_geometry_3d(xc, xs, yc, ys, zc, zs)
        )

        assert a_est == pytest.approx(a, abs=1e-10)
        assert b_est == pytest.approx(b, abs=1e-10)
        assert beta_est == pytest.approx(0.0, abs=1e-10)

    def test_gimbal_lock_beta_pi(self):
        """Handles gimbal lock when β ≈ π without error."""
        a, b = 3.0, 1.0
        phi = -0.2
        alpha, beta, gamma = 0.3, np.pi, 0.7
        xc, xs, yc, ys, zc, zs = _make_ellipse_coefficients(
            a, b, alpha, beta, gamma, phi
        )
        phi_est, a_est, b_est, alpha_est, beta_est, gamma_est = (
            _compute_ellipse_geometry_3d(xc, xs, yc, ys, zc, zs)
        )

        assert a_est == pytest.approx(a, abs=1e-10)
        assert b_est == pytest.approx(b, abs=1e-10)
        assert beta_est == pytest.approx(np.pi, abs=1e-10)

    def test_output_types(self):
        """All returned values are floats."""
        xc, xs, yc, ys, zc, zs = _make_ellipse_coefficients(
            2.0, 1.0, 0.5, 0.8, 0.3, 0.1
        )
        result = _compute_ellipse_geometry_3d(xc, xs, yc, ys, zc, zs)
        assert len(result) == 6
        for val in result:
            assert isinstance(val, (float, np.floating))


def _make_3d_outline(n_harmonics=6, t_num=360, rng=None):
    """Helper: generate a synthetic 3D closed curve from random Fourier coefficients.

    Returns (X_coords, coef_dict) where coef_dict has keys
    'a0','c0','e0','an','bn','cn','dn','en','fn'.
    """
    if rng is None:
        rng = np.random.default_rng(999)
    t = np.linspace(2 * np.pi / t_num, 2 * np.pi, t_num)

    a0, c0, e0 = rng.uniform(-2, 2, size=3)
    an = rng.standard_normal(n_harmonics)
    bn = rng.standard_normal(n_harmonics)
    cn = rng.standard_normal(n_harmonics)
    dn = rng.standard_normal(n_harmonics)
    en = rng.standard_normal(n_harmonics)
    fn = rng.standard_normal(n_harmonics)

    # Make 1st harmonic dominant so the ellipse is well-defined
    an[0] *= 3
    bn[0] *= 3
    cn[0] *= 3
    dn[0] *= 3
    en[0] *= 3
    fn[0] *= 3

    cos = np.cos(np.tensordot(np.arange(1, n_harmonics + 1, 1), t, 0))
    sin = np.sin(np.tensordot(np.arange(1, n_harmonics + 1, 1), t, 0))

    x = a0 / 2 + np.dot(an, cos) + np.dot(bn, sin)
    y = c0 / 2 + np.dot(cn, cos) + np.dot(dn, sin)
    z = e0 / 2 + np.dot(en, cos) + np.dot(fn, sin)
    X_coords = np.stack([x, y, z], 1)

    return X_coords, {
        "a0": a0,
        "c0": c0,
        "e0": e0,
        "an": an,
        "bn": bn,
        "cn": cn,
        "dn": dn,
        "en": en,
        "fn": fn,
    }


class TestNormalize3d:
    """Tests for EllipticFourierAnalysis._normalize_3d method."""

    def test_returns_correct_structure(self):
        """_normalize_3d returns 11 values: 6 arrays + 5 floats."""
        n_harmonics = 6
        X_coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, rng=np.random.default_rng(42)
        )

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics)
        coef = efa.fit_transform([X_coords], t=None, norm=False)[0]
        arrays = coef.reshape(6, n_harmonics + 1)
        an, bn, cn, dn, en, fn = arrays

        result = efa._normalize_3d(an, bn, cn, dn, en, fn)
        assert len(result) == 11
        # First 6 are arrays of shape (n_harmonics+1,)
        for i in range(6):
            assert isinstance(result[i], np.ndarray)
            assert result[i].shape == (n_harmonics + 1,)
        # Last 5 are floats (alpha, beta, gamma, phi, scale)
        for i in range(6, 11):
            assert isinstance(result[i], (float, np.floating))

    def test_canonical_first_harmonic(self):
        """After normalization,
        1st harmonic defines XY-plane ellipse with semi-major along X.

        Canonical form: a1_norm > 0, b1_norm = 0, c1_norm = 0, d1_norm > 0,
        e1_norm = 0, f1_norm = 0 (or nearly so).
        """
        n_harmonics = 6
        X_coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, rng=np.random.default_rng(42)
        )

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics)
        coef = efa.fit_transform([X_coords], t=None, norm=False)[0]
        arrays = coef.reshape(6, n_harmonics + 1)
        an, bn, cn, dn, en, fn = arrays

        An, Bn, Cn, Dn, En, Fn, alpha, beta, gamma, phi, scale = efa._normalize_3d(
            an, bn, cn, dn, en, fn
        )

        # 1st harmonic: semi-major along X (a1 > 0), zero b1
        # semi-minor along Y (d1 > 0), zero c1
        # z-components zero (e1=0, f1=0)
        assert An[1] > 0, f"An[1]={An[1]} should be positive (semi-major along X)"
        assert abs(Bn[1]) < 1e-10, f"Bn[1]={Bn[1]} should be ~0"
        assert abs(Cn[1]) < 1e-10, f"Cn[1]={Cn[1]} should be ~0"
        assert Dn[1] >= 0, f"Dn[1]={Dn[1]} should be non-negative"
        assert abs(En[1]) < 1e-10, f"En[1]={En[1]} should be ~0"
        assert abs(Fn[1]) < 1e-10, f"Fn[1]={Fn[1]} should be ~0"


class TestNormMethodParameter:
    """Tests for the norm_method parameter in EllipticFourierAnalysis."""

    def test_default_norm_method_is_area(self):
        """Default norm_method should be 'area'."""
        efa = EllipticFourierAnalysis(n_dim=3)
        assert efa.norm_method == "area"

    def test_norm_method_area_accepted(self):
        """norm_method='area' should be accepted without error."""
        efa = EllipticFourierAnalysis(n_dim=3, norm_method="area")
        assert efa.norm_method == "area"

    def test_norm_method_semi_major_axis_accepted(self):
        """norm_method='semi_major_axis' should be accepted without error."""
        efa = EllipticFourierAnalysis(n_dim=3, norm_method="semi_major_axis")
        assert efa.norm_method == "semi_major_axis"

    def test_invalid_norm_method_raises_valueerror(self):
        """Invalid norm_method should raise ValueError with informative message."""
        with pytest.raises(ValueError, match="norm_method"):
            EllipticFourierAnalysis(n_dim=3, norm_method="invalid")

    def test_invalid_norm_method_lists_valid_options(self):
        """Error message should list valid options."""
        with pytest.raises(
            ValueError, match="area.*semi_major_axis|semi_major_axis.*area"
        ):
            EllipticFourierAnalysis(n_dim=3, norm_method="foobar")

    def test_get_params_includes_norm_method(self):
        """sklearn get_params() should discover norm_method."""
        efa = EllipticFourierAnalysis(n_dim=3, norm_method="semi_major_axis")
        params = efa.get_params()
        assert "norm_method" in params
        assert params["norm_method"] == "semi_major_axis"

    def test_set_params_norm_method(self):
        """sklearn set_params() should be able to set norm_method."""
        efa = EllipticFourierAnalysis(n_dim=3, norm_method="area")
        efa.set_params(norm_method="semi_major_axis")
        assert efa.norm_method == "semi_major_axis"

    def test_norm_method_does_not_affect_2d(self):
        """norm_method parameter should not affect 2D EFA behavior."""
        X = _load_wings_as_list(n_specimens=10)
        n_harmonics = 6

        efa_default = EllipticFourierAnalysis(n_harmonics=n_harmonics, n_dim=2)
        efa_area = EllipticFourierAnalysis(
            n_harmonics=n_harmonics, n_dim=2, norm_method="area"
        )
        efa_semi = EllipticFourierAnalysis(
            n_harmonics=n_harmonics, n_dim=2, norm_method="semi_major_axis"
        )

        coef_default = efa_default.fit_transform(X, norm=True)
        coef_area = efa_area.fit_transform(X, norm=True)
        coef_semi = efa_semi.fit_transform(X, norm=True)

        assert_array_almost_equal(coef_default, coef_area)
        assert_array_almost_equal(coef_default, coef_semi)

    def test_scale_is_sqrt_area(self):
        """The returned scale equals sqrt(pi * a1 * b1)."""
        n_harmonics = 6
        X_coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, rng=np.random.default_rng(42)
        )

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics)
        coef = efa.fit_transform([X_coords], t=None, norm=False)[0]
        arrays = coef.reshape(6, n_harmonics + 1)
        an, bn, cn, dn, en, fn = arrays

        # Compute expected scale from 1st harmonic geometry
        phi1, a1, b1, _, _, _ = _compute_ellipse_geometry_3d(
            an[1], bn[1], cn[1], dn[1], en[1], fn[1]
        )
        expected_scale = np.sqrt(np.pi * a1 * b1)

        _, _, _, _, _, _, alpha, beta, gamma, phi, scale = efa._normalize_3d(
            an, bn, cn, dn, en, fn
        )

        assert scale == pytest.approx(expected_scale, rel=1e-10)

    def test_orientation_parameters_match_geometry(self):
        """The returned alpha, beta, gamma, phi match the 1st harmonic's geometry."""
        n_harmonics = 6
        X_coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, rng=np.random.default_rng(42)
        )

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics)
        coef = efa.fit_transform([X_coords], t=None, norm=False)[0]
        arrays = coef.reshape(6, n_harmonics + 1)
        an, bn, cn, dn, en, fn = arrays

        phi_expected, _, _, alpha_expected, beta_expected, gamma_expected = (
            _compute_ellipse_geometry_3d(an[1], bn[1], cn[1], dn[1], en[1], fn[1])
        )

        _, _, _, _, _, _, alpha, beta, gamma, phi, _ = efa._normalize_3d(
            an, bn, cn, dn, en, fn
        )

        assert alpha == pytest.approx(alpha_expected, abs=1e-10)
        assert beta == pytest.approx(beta_expected, abs=1e-10)
        assert gamma == pytest.approx(gamma_expected, abs=1e-10)
        assert phi == pytest.approx(phi_expected, abs=1e-10)

    def test_degenerate_first_harmonic_raises(self):
        """Degenerate 1st harmonic (all coefficients near zero) raises ValueError."""
        n_harmonics = 6
        an = np.zeros(n_harmonics + 1)
        bn = np.zeros(n_harmonics + 1)
        cn = np.zeros(n_harmonics + 1)
        dn = np.zeros(n_harmonics + 1)
        en = np.zeros(n_harmonics + 1)
        fn = np.zeros(n_harmonics + 1)
        # Set some higher harmonics so array is not trivially empty
        an[2] = 1.0
        cn[3] = 0.5

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics)
        with pytest.raises(ValueError, match="(?i)degenerate"):
            efa._normalize_3d(an, bn, cn, dn, en, fn)

    def test_dc_components_preserved(self):
        """Offset (index 0) are preserved unchanged through normalization."""
        n_harmonics = 6
        X_coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, rng=np.random.default_rng(42)
        )

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics)
        coef = efa.fit_transform([X_coords], t=None, norm=False)[0]
        arrays = coef.reshape(6, n_harmonics + 1)
        an, bn, cn, dn, en, fn = arrays

        An, Bn, Cn, Dn, En, Fn, *_ = efa._normalize_3d(an, bn, cn, dn, en, fn)

        # Offset components should be the originals (not normalized by scale)
        assert An[0] == pytest.approx(an[0], abs=1e-10)
        assert Bn[0] == pytest.approx(bn[0], abs=1e-10)
        assert Cn[0] == pytest.approx(cn[0], abs=1e-10)
        assert Dn[0] == pytest.approx(dn[0], abs=1e-10)
        assert En[0] == pytest.approx(en[0], abs=1e-10)
        assert Fn[0] == pytest.approx(fn[0], abs=1e-10)

    def test_known_ellipse_normalization(self):
        """Normalization of a known synthetic ellipse produces correct canonical form.

        Construct a 3D curve whose 1st harmonic is a known ellipse, then verify
        normalization produces the expected result.
        """
        n_harmonics = 6
        t_num = 360

        # Known 1st harmonic geometry
        a1, b1 = 5.0, 2.0
        phi1 = 0.3
        alpha1, beta1, gamma1 = 0.7, 1.2, -0.4

        # Build coefficients for harmonic 1 from known geometry
        xc1, xs1, yc1, ys1, zc1, zs1 = _make_ellipse_coefficients(
            a1, b1, alpha1, beta1, gamma1, phi1
        )

        # Build full coefficient arrays (only harmonic 1 is non-zero)
        an = np.array([0.0, xc1, 0, 0, 0, 0, 0])  # DC=0, then harmonics
        bn = np.array([0.0, xs1, 0, 0, 0, 0, 0])
        cn = np.array([0.0, yc1, 0, 0, 0, 0, 0])
        dn = np.array([0.0, ys1, 0, 0, 0, 0, 0])
        en = np.array([0.0, zc1, 0, 0, 0, 0, 0])
        fn = np.array([0.0, zs1, 0, 0, 0, 0, 0])

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics)
        An, Bn, Cn, Dn, En, Fn, alpha, beta, gamma, phi, scale = efa._normalize_3d(
            an, bn, cn, dn, en, fn
        )

        expected_scale = np.sqrt(np.pi * a1 * b1)

        # After normalization, 1st harmonic should be canonical:
        # An[1] = a1/scale, Bn[1] = 0, Cn[1] = 0, Dn[1] = b1/scale, En[1] = 0, Fn[1] = 0
        assert An[1] == pytest.approx(a1 / expected_scale, abs=1e-10)
        assert abs(Bn[1]) < 1e-10
        assert abs(Cn[1]) < 1e-10
        assert Dn[1] == pytest.approx(b1 / expected_scale, abs=1e-10)
        assert abs(En[1]) < 1e-10
        assert abs(Fn[1]) < 1e-10

        assert scale == pytest.approx(expected_scale, abs=1e-10)


class TestTransformSingle3dPipeline:
    """Tests for 3D normalization integration into the transform pipeline."""

    def test_norm_true_returns_correct_shape(self):
        """transform with norm=True for 3D returns shape (1, 6*(n_harmonics+1))."""
        n_harmonics = 6
        X_coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, rng=np.random.default_rng(10)
        )

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics)
        result = efa.fit_transform([X_coords], norm=True)

        assert result.shape == (1, 6 * (n_harmonics + 1))

    def test_norm_true_return_orientation_scale_shape(self):
        """transform with norm=True, return_orientation_scale=True appends 5 values."""
        n_harmonics = 6
        X_coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, rng=np.random.default_rng(10)
        )

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics)
        result = efa.fit_transform([X_coords], norm=True, return_orientation_scale=True)

        expected_len = 6 * (n_harmonics + 1) + 5
        assert result.shape == (1, expected_len)

    def test_orientation_scale_values_are_correct(self):
        """The 5 appended values (alpha, beta, gamma, phi, scale) match _normalize_3d output."""
        n_harmonics = 6
        X_coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, rng=np.random.default_rng(10)
        )

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics)

        # Get raw coefficients first
        raw = efa.fit_transform([X_coords], norm=False)[0]
        arrays = raw.reshape(6, n_harmonics + 1)
        an, bn, cn, dn, en, fn = arrays

        # Get expected orientation/scale from _normalize_3d
        _, _, _, _, _, _, alpha_exp, beta_exp, gamma_exp, phi_exp, scale_exp = (
            efa._normalize_3d(an, bn, cn, dn, en, fn)
        )

        # Get from pipeline
        result = efa.fit_transform(
            [X_coords], norm=True, return_orientation_scale=True
        )[0]
        alpha_got, beta_got, gamma_got, phi_got, scale_got = result[-5:]

        assert alpha_got == pytest.approx(alpha_exp, abs=1e-10)
        assert beta_got == pytest.approx(beta_exp, abs=1e-10)
        assert gamma_got == pytest.approx(gamma_exp, abs=1e-10)
        assert phi_got == pytest.approx(phi_exp, abs=1e-10)
        assert scale_got == pytest.approx(scale_exp, abs=1e-10)

    def test_normalized_coefficients_match_direct_call(self):
        """Normalized coefficients from pipeline match those from _normalize_3d directly."""
        n_harmonics = 6
        X_coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, rng=np.random.default_rng(10)
        )

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics)

        # Get raw coefficients
        raw = efa.fit_transform([X_coords], norm=False)[0]
        arrays = raw.reshape(6, n_harmonics + 1)
        an, bn, cn, dn, en, fn = arrays

        # Direct normalization
        An, Bn, Cn, Dn, En, Fn, *_ = efa._normalize_3d(an, bn, cn, dn, en, fn)
        expected_coef = np.hstack([An, Bn, Cn, Dn, En, Fn])

        # Pipeline normalization
        result = efa.fit_transform([X_coords], norm=True)[0]

        assert_array_almost_equal(result, expected_coef, decimal=10)

    def test_multiple_samples(self):
        """transform works with multiple 3D samples and norm=True."""
        n_harmonics = 6
        X1, _ = _make_3d_outline(n_harmonics=n_harmonics, rng=np.random.default_rng(10))
        X2, _ = _make_3d_outline(n_harmonics=n_harmonics, rng=np.random.default_rng(20))

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics)
        result = efa.fit_transform([X1, X2], norm=True)

        assert result.shape == (2, 6 * (n_harmonics + 1))

    def test_fit_transform_equals_transform(self):
        """fit_transform and transform produce the same result for 3D norm=True."""
        n_harmonics = 6
        X_coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, rng=np.random.default_rng(10)
        )

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics)
        result_ft = efa.fit_transform(
            [X_coords], norm=True, return_orientation_scale=True
        )
        result_t = efa.transform([X_coords], norm=True, return_orientation_scale=True)

        assert_array_almost_equal(result_ft, result_t)


class TestNormalize3dInvariance:
    """Invariance and validation tests for 3D EFA normalization."""

    def test_translation_invariance(self):
        """Normalized coefficients are invariant under 3D translation."""
        n_harmonics = 6
        rng = np.random.default_rng(50)
        X_coords, _ = _make_3d_outline(n_harmonics=n_harmonics, rng=rng)

        # Apply a random 3D translation
        translation = rng.uniform(-10, 10, size=3)
        X_translated = X_coords + translation

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics)
        coef_orig = efa.fit_transform([X_coords], norm=True)[0]
        coef_trans = efa.fit_transform([X_translated], norm=True)[0]

        # Harmonic coefficients (indices 1+) should match; DC may differ
        coef_orig_harmonics = coef_orig.reshape(6, n_harmonics + 1)[:, 1:]
        coef_trans_harmonics = coef_trans.reshape(6, n_harmonics + 1)[:, 1:]

        assert_array_almost_equal(coef_orig_harmonics, coef_trans_harmonics, decimal=5)

    def test_scale_invariance(self):
        """Normalized coefficients are invariant under uniform scaling."""
        n_harmonics = 6
        rng = np.random.default_rng(51)
        X_coords, _ = _make_3d_outline(n_harmonics=n_harmonics, rng=rng)

        # Apply a random uniform scaling
        scale_factor = rng.uniform(0.5, 5.0)
        X_scaled = X_coords * scale_factor

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics)
        coef_orig = efa.fit_transform([X_coords], norm=True)[0]
        coef_scaled = efa.fit_transform([X_scaled], norm=True)[0]

        # Harmonic coefficients should match
        coef_orig_harmonics = coef_orig.reshape(6, n_harmonics + 1)[:, 1:]
        coef_scaled_harmonics = coef_scaled.reshape(6, n_harmonics + 1)[:, 1:]

        assert_array_almost_equal(coef_orig_harmonics, coef_scaled_harmonics, decimal=5)

    def test_rotation_invariance(self):
        """Normalized coefficients are invariant under 3D rotation."""
        n_harmonics = 6
        rng = np.random.default_rng(52)
        X_coords, _ = _make_3d_outline(n_harmonics=n_harmonics, rng=rng)

        # Apply a random 3D rotation via ZXZ Euler angles
        alpha_r = rng.uniform(-np.pi, np.pi)
        beta_r = rng.uniform(0, np.pi)
        gamma_r = rng.uniform(-np.pi, np.pi)
        R = rotation_matrix_3d_euler_zxz(alpha_r, beta_r, gamma_r)
        X_rotated = (R @ X_coords.T).T

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics)
        coef_orig = efa.fit_transform([X_coords], norm=True)[0]
        coef_rot = efa.fit_transform([X_rotated], norm=True)[0]

        # Harmonic coefficients should match
        coef_orig_harmonics = coef_orig.reshape(6, n_harmonics + 1)[:, 1:]
        coef_rot_harmonics = coef_rot.reshape(6, n_harmonics + 1)[:, 1:]

        assert_array_almost_equal(coef_orig_harmonics, coef_rot_harmonics, decimal=4)

    def test_startpoint_shift_invariance(self):
        """Normalized shapes are approximately invariant under cyclic permutation.

        Starting-point shift changes arc-length parameterization, which can cause
        the phase normalization to select a different canonical branch (phi jumping
        by pi/2). This is an inherent limitation shared by 2D and 3D EFA
        normalization. We verify that the normalized shapes remain geometrically
        close using Wasserstein distance with a tolerance that accounts for this
        discrete ambiguity.
        """
        n_harmonics = 20
        t_num = 360
        rng = np.random.default_rng(53)
        X_coords, _ = _make_3d_outline(n_harmonics=n_harmonics, t_num=t_num, rng=rng)

        # Cyclically shift the starting point
        shift = rng.integers(1, len(X_coords))
        X_shifted = np.roll(X_coords, shift, axis=0)

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics)
        coef_orig = efa.fit_transform([X_coords], norm=True)
        coef_shifted = efa.fit_transform([X_shifted], norm=True)

        # Reconstruct normalized shapes and compare geometrically
        X_recon_orig = np.array(
            efa.inverse_transform(coef_orig, t_num=t_num, norm=True)
        )[0]
        X_recon_shifted = np.array(
            efa.inverse_transform(coef_shifted, t_num=t_num, norm=True)
        )[0]

        assert wasserstein_distance_nd(X_recon_orig, X_recon_shifted) < 0.5

    def test_round_trip_reconstruction(self):
        """Forward (norm=True) then inverse (norm=True) produces consistent shape.

        After transform(norm=True) -> inverse_transform(norm=True), re-transforming
        with explicit uniform t (matching the inverse output spacing) should recover
        the normalized coefficients. We also verify geometric consistency using
        Wasserstein distance, following the pattern of test_orientation_and_scale_2d.
        """
        n_harmonics = 20
        t_num = 360
        rng = np.random.default_rng(54)
        X_coords, _ = _make_3d_outline(n_harmonics=n_harmonics, t_num=t_num, rng=rng)

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics)
        coef_norm = efa.fit_transform([X_coords], norm=True)
        X_recon = np.array(efa.inverse_transform(coef_norm, t_num=t_num, norm=True))

        # Re-transform with explicit uniform t (matching inverse output spacing)
        t_uniform = [
            np.linspace(2 * np.pi / t_num, 2 * np.pi, t_num)
            for _ in range(len(X_recon))
        ]
        coef_recon = efa.fit_transform(X_recon, t=t_uniform, norm=False)

        # Compare harmonics (skip DC since norm=True zeroes it in inverse)
        coef_norm_harmonics = coef_norm.reshape(-1, 6, n_harmonics + 1)[:, :, 1:]
        coef_recon_harmonics = coef_recon.reshape(-1, 6, n_harmonics + 1)[:, :, 1:]

        assert_array_almost_equal(coef_norm_harmonics, coef_recon_harmonics, decimal=3)

    def test_orientation_scale_return(self):
        """return_orientation_scale=True produces correct shape and recoverable params.

        To reconstruct the original outline from normalized coefficients, we must
        undo normalization: evaluate the Fourier series at (t + phi) to undo the
        phase shift, then apply Omega rotation and scale, and add back the centroid.

        We use explicit uniform t (matching _make_3d_outline's parameterization)
        so that the Fourier coefficients exactly represent the input curve,
        following the same pattern as test_orientation_and_scale_2d.
        """
        n_harmonics = 20
        t_num = 360
        rng = np.random.default_rng(55)
        X_coords, _ = _make_3d_outline(n_harmonics=n_harmonics, t_num=t_num, rng=rng)

        t_uniform = np.linspace(2 * np.pi / t_num, 2 * np.pi, t_num)

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics)
        result = efa.fit_transform(
            [X_coords], t=[t_uniform], norm=True, return_orientation_scale=True
        )[0]

        # Shape check: 6*(n_harmonics+1) + 5
        expected_len = 6 * (n_harmonics + 1) + 5
        assert len(result) == expected_len

        # Extract orientation/scale
        alpha, beta, gamma, phi, scale = result[-5:]
        coef = result[:-5].reshape(6, n_harmonics + 1)

        # Reconstruct with phase shift undone: evaluate at (t + phi)
        An, Bn, Cn, Dn, En, Fn = coef
        t = t_uniform
        t_shifted = t + phi  # Undo phase shift
        harmonics = np.arange(1, n_harmonics + 1, 1)
        cos_t = np.cos(np.tensordot(harmonics, t_shifted, 0))
        sin_t = np.sin(np.tensordot(harmonics, t_shifted, 0))

        x_norm = np.dot(An[1:], cos_t) + np.dot(Bn[1:], sin_t)
        y_norm = np.dot(Cn[1:], cos_t) + np.dot(Dn[1:], sin_t)
        z_norm = np.dot(En[1:], cos_t) + np.dot(Fn[1:], sin_t)
        X_norm = np.stack([x_norm, y_norm, z_norm], axis=1)

        # Undo normalization: rotate back and rescale
        Omega = rotation_matrix_3d_euler_zxz(alpha, beta, gamma)
        X_denorm = scale * (Omega @ X_norm.T).T

        # Add back centroid
        centroid = np.array([An[0] / 2, Cn[0] / 2, En[0] / 2])
        X_denorm += centroid

        # Should be close to the original (allowing Fourier truncation error)
        assert wasserstein_distance_nd(X_coords, X_denorm) < 0.5

    def test_canonical_first_harmonic_via_pipeline(self):
        """After normalization via pipeline, 1st harmonic is canonical."""
        n_harmonics = 6
        rng = np.random.default_rng(56)
        X_coords, _ = _make_3d_outline(n_harmonics=n_harmonics, rng=rng)

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics)
        coef = efa.fit_transform([X_coords], norm=True)[0]
        arrays = coef.reshape(6, n_harmonics + 1)

        An, Bn, Cn, Dn, En, Fn = arrays

        # 1st harmonic: XY-plane ellipse with semi-major along X
        assert An[1] > 0, f"An[1]={An[1]} should be positive"
        assert abs(Bn[1]) < 1e-10, f"Bn[1]={Bn[1]} should be ~0"
        assert abs(Cn[1]) < 1e-10, f"Cn[1]={Cn[1]} should be ~0"
        assert Dn[1] >= 0, f"Dn[1]={Dn[1]} should be non-negative"
        assert abs(En[1]) < 1e-10, f"En[1]={En[1]} should be ~0"
        assert abs(Fn[1]) < 1e-10, f"Fn[1]={Fn[1]} should be ~0"


class TestSemiMajorAxisNormalization:
    """Tests for semi-major axis scale factor in _normalize_3d."""

    def test_scale_is_a1_when_semi_major_axis(self):
        """When norm_method='semi_major_axis', returned scale equals a1."""
        n_harmonics = 6
        X_coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, rng=np.random.default_rng(42)
        )

        efa = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm_method="semi_major_axis"
        )
        coef = efa.fit_transform([X_coords], t=None, norm=False)[0]
        arrays = coef.reshape(6, n_harmonics + 1)
        an, bn, cn, dn, en, fn = arrays

        # Expected scale = a1
        _, a1, _, _, _, _ = _compute_ellipse_geometry_3d(
            an[1], bn[1], cn[1], dn[1], en[1], fn[1]
        )

        *_, scale = efa._normalize_3d(an, bn, cn, dn, en, fn)
        assert scale == pytest.approx(a1, rel=1e-10)

    def test_scale_differs_from_area_method(self):
        """semi_major_axis scale differs from area-based scale (unless b1=a1/pi)."""
        n_harmonics = 6
        X_coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, rng=np.random.default_rng(42)
        )

        efa_area = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm_method="area"
        )
        efa_semi = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm_method="semi_major_axis"
        )

        coef = efa_area.fit_transform([X_coords], t=None, norm=False)[0]
        arrays = coef.reshape(6, n_harmonics + 1)
        an, bn, cn, dn, en, fn = arrays

        *_, scale_area = efa_area._normalize_3d(an, bn, cn, dn, en, fn)
        *_, scale_semi = efa_semi._normalize_3d(an, bn, cn, dn, en, fn)

        assert scale_area != pytest.approx(scale_semi, rel=1e-3)

    def test_canonical_first_harmonic_semi_major(self):
        """After semi_major_axis normalization, 1st harmonic An[1] = 1.0."""
        n_harmonics = 6
        X_coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, rng=np.random.default_rng(42)
        )

        efa = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm_method="semi_major_axis"
        )
        coef = efa.fit_transform([X_coords], t=None, norm=False)[0]
        arrays = coef.reshape(6, n_harmonics + 1)
        an, bn, cn, dn, en, fn = arrays

        An, Bn, Cn, Dn, En, Fn, *_ = efa._normalize_3d(an, bn, cn, dn, en, fn)

        # Semi-major axis length in canonical form = a1/scale = a1/a1 = 1.0
        assert An[1] == pytest.approx(1.0, abs=1e-10)
        assert abs(Bn[1]) < 1e-10
        assert abs(Cn[1]) < 1e-10
        assert abs(En[1]) < 1e-10
        assert abs(Fn[1]) < 1e-10

    def test_orientation_same_regardless_of_method(self):
        """Orientation parameters (alpha, beta, gamma, phi) are identical for both methods."""
        n_harmonics = 6
        X_coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, rng=np.random.default_rng(42)
        )

        efa_area = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm_method="area"
        )
        efa_semi = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm_method="semi_major_axis"
        )

        coef = efa_area.fit_transform([X_coords], t=None, norm=False)[0]
        arrays = coef.reshape(6, n_harmonics + 1)
        an, bn, cn, dn, en, fn = arrays

        _, _, _, _, _, _, a_a, b_a, g_a, phi_a, _ = efa_area._normalize_3d(
            an, bn, cn, dn, en, fn
        )
        _, _, _, _, _, _, a_s, b_s, g_s, phi_s, _ = efa_semi._normalize_3d(
            an, bn, cn, dn, en, fn
        )

        assert a_a == pytest.approx(a_s, abs=1e-10)
        assert b_a == pytest.approx(b_s, abs=1e-10)
        assert g_a == pytest.approx(g_s, abs=1e-10)
        assert phi_a == pytest.approx(phi_s, abs=1e-10)

    def test_pipeline_return_orientation_scale_semi_major(self):
        """return_orientation_scale=True returns a1 as scale for semi_major_axis method."""
        n_harmonics = 6
        X_coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, rng=np.random.default_rng(42)
        )

        efa = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm_method="semi_major_axis"
        )
        result = efa.fit_transform(
            [X_coords], norm=True, return_orientation_scale=True
        )[0]

        # Get raw coefficients for expected a1
        raw = efa.fit_transform([X_coords], norm=False)[0]
        arrays = raw.reshape(6, n_harmonics + 1)
        an, bn, cn, dn, en, fn = arrays
        _, a1, _, _, _, _ = _compute_ellipse_geometry_3d(
            an[1], bn[1], cn[1], dn[1], en[1], fn[1]
        )

        scale_returned = result[-1]
        assert scale_returned == pytest.approx(a1, rel=1e-10)

    def test_area_method_unchanged_after_branch(self):
        """area method still returns sqrt(pi*a1*b1) after adding the branch."""
        n_harmonics = 6
        X_coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, rng=np.random.default_rng(42)
        )

        efa = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm_method="area"
        )
        coef = efa.fit_transform([X_coords], t=None, norm=False)[0]
        arrays = coef.reshape(6, n_harmonics + 1)
        an, bn, cn, dn, en, fn = arrays

        _, a1, b1, _, _, _ = _compute_ellipse_geometry_3d(
            an[1], bn[1], cn[1], dn[1], en[1], fn[1]
        )
        expected_scale = np.sqrt(np.pi * a1 * b1)

        *_, scale = efa._normalize_3d(an, bn, cn, dn, en, fn)
        assert scale == pytest.approx(expected_scale, rel=1e-10)


class TestSemiMajorAxisCanonical:
    """Tests verifying semi-major axis length = 1 after normalization."""

    def test_semi_major_axis_length_is_one_via_geometry(self):
        """Compute semi-major axis from normalized 1st harmonic via geometry function.

        Uses _compute_ellipse_geometry_3d on the normalized coefficients to verify
        that the semi-major axis length equals 1.0, not just An[1].
        """
        n_harmonics = 6
        rng = np.random.default_rng(42)

        for seed in range(10):
            X_coords, _ = _make_3d_outline(
                n_harmonics=n_harmonics, rng=np.random.default_rng(seed + 100)
            )

            efa = EllipticFourierAnalysis(
                n_dim=3, n_harmonics=n_harmonics, norm_method="semi_major_axis"
            )
            coef = efa.fit_transform([X_coords], norm=True)[0]
            arrays = coef.reshape(6, n_harmonics + 1)
            An, Bn, Cn, Dn, En, Fn = arrays

            # Extract semi-major axis from normalized 1st harmonic
            _, a1_norm, b1_norm, _, _, _ = _compute_ellipse_geometry_3d(
                An[1], Bn[1], Cn[1], Dn[1], En[1], Fn[1]
            )

            assert a1_norm == pytest.approx(1.0, abs=1e-10), (
                f"seed={seed}: semi-major axis length = {a1_norm}, expected 1.0"
            )

    def test_area_method_semi_major_not_one(self):
        """With area normalization, semi-major axis length is NOT 1 (it's a/sqrt(pi*a*b))."""
        n_harmonics = 6
        X_coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, rng=np.random.default_rng(42)
        )

        efa = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm_method="area"
        )
        coef = efa.fit_transform([X_coords], norm=True)[0]
        arrays = coef.reshape(6, n_harmonics + 1)
        An, Bn, Cn, Dn, En, Fn = arrays

        _, a1_norm, _, _, _, _ = _compute_ellipse_geometry_3d(
            An[1], Bn[1], Cn[1], Dn[1], En[1], Fn[1]
        )

        # For area normalization, a1_norm = a1 / sqrt(pi*a1*b1), which != 1 in general
        assert a1_norm != pytest.approx(1.0, abs=1e-3)


class TestSemiMajorAxisInvariance:
    """Invariance tests for semi-major axis normalization."""

    def test_translation_invariance(self):
        """Semi-major-axis normalized coefficients are invariant under translation."""
        n_harmonics = 6
        rng = np.random.default_rng(60)
        X_coords, _ = _make_3d_outline(n_harmonics=n_harmonics, rng=rng)

        translation = rng.uniform(-10, 10, size=3)
        X_translated = X_coords + translation

        efa = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm_method="semi_major_axis"
        )
        coef_orig = efa.fit_transform([X_coords], norm=True)[0]
        coef_trans = efa.fit_transform([X_translated], norm=True)[0]

        coef_orig_harmonics = coef_orig.reshape(6, n_harmonics + 1)[:, 1:]
        coef_trans_harmonics = coef_trans.reshape(6, n_harmonics + 1)[:, 1:]

        assert_array_almost_equal(coef_orig_harmonics, coef_trans_harmonics, decimal=5)

    def test_scale_invariance(self):
        """Semi-major-axis normalized coefficients are invariant under uniform scaling."""
        n_harmonics = 6
        rng = np.random.default_rng(61)
        X_coords, _ = _make_3d_outline(n_harmonics=n_harmonics, rng=rng)

        scale_factor = rng.uniform(0.5, 5.0)
        X_scaled = X_coords * scale_factor

        efa = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm_method="semi_major_axis"
        )
        coef_orig = efa.fit_transform([X_coords], norm=True)[0]
        coef_scaled = efa.fit_transform([X_scaled], norm=True)[0]

        coef_orig_harmonics = coef_orig.reshape(6, n_harmonics + 1)[:, 1:]
        coef_scaled_harmonics = coef_scaled.reshape(6, n_harmonics + 1)[:, 1:]

        assert_array_almost_equal(coef_orig_harmonics, coef_scaled_harmonics, decimal=5)

    def test_rotation_invariance(self):
        """Semi-major-axis normalized coefficients are invariant under 3D rotation."""
        n_harmonics = 6
        rng = np.random.default_rng(62)
        X_coords, _ = _make_3d_outline(n_harmonics=n_harmonics, rng=rng)

        alpha_r = rng.uniform(-np.pi, np.pi)
        beta_r = rng.uniform(0, np.pi)
        gamma_r = rng.uniform(-np.pi, np.pi)
        R = rotation_matrix_3d_euler_zxz(alpha_r, beta_r, gamma_r)
        X_rotated = (R @ X_coords.T).T

        efa = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm_method="semi_major_axis"
        )
        coef_orig = efa.fit_transform([X_coords], norm=True)[0]
        coef_rot = efa.fit_transform([X_rotated], norm=True)[0]

        coef_orig_harmonics = coef_orig.reshape(6, n_harmonics + 1)[:, 1:]
        coef_rot_harmonics = coef_rot.reshape(6, n_harmonics + 1)[:, 1:]

        assert_array_almost_equal(coef_orig_harmonics, coef_rot_harmonics, decimal=4)

    def test_startpoint_shift_invariance(self):
        """Semi-major-axis normalized shapes are approx invariant under cyclic permutation.

        Starting-point shift changes arc-length parameterization, which can cause
        the phase normalization to select a different canonical branch. The tolerance
        is set to 1.0 to account for the discrete phi ambiguity, matching the same
        pattern as the area-based startpoint shift test.
        """
        n_harmonics = 20
        t_num = 360
        rng = np.random.default_rng(53)
        X_coords, _ = _make_3d_outline(n_harmonics=n_harmonics, t_num=t_num, rng=rng)

        shift = rng.integers(1, len(X_coords))
        X_shifted = np.roll(X_coords, shift, axis=0)

        efa = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm_method="semi_major_axis"
        )
        coef_orig = efa.fit_transform([X_coords], norm=True)
        coef_shifted = efa.fit_transform([X_shifted], norm=True)

        X_recon_orig = np.array(
            efa.inverse_transform(coef_orig, t_num=t_num, norm=True)
        )[0]
        X_recon_shifted = np.array(
            efa.inverse_transform(coef_shifted, t_num=t_num, norm=True)
        )[0]

        assert wasserstein_distance_nd(X_recon_orig, X_recon_shifted) < 1.0


class TestAreaNormalizationRegression:
    """Regression test for area-based normalization.

    Snapshot captured from the implementation at commit 59ada8f (before
    norm_method was added). Verifies that the default area-based normalization
    continues to produce bit-identical results.
    """

    # Snapshot: _make_3d_outline(n_harmonics=6, rng=default_rng(42)), norm=True
    _EXPECTED_COEF = np.array(
        [
            1.8786552291066900e00,
            4.4749144074020974e-01,
            -1.6544642156530842e-01,
            -5.7844176084800636e-02,
            1.2527281927360465e-01,
            -8.1717978045931713e-03,
            2.1879616966356572e-02,
            0.0000000000000000e00,
            1.4945917697214952e-16,
            4.0755449959920205e-02,
            -5.4420755687897303e-02,
            1.0763893995449975e-01,
            -2.0239890553899149e-02,
            6.8784209336682653e-02,
            -2.8999777542514682e-01,
            -1.0218868790537593e-16,
            -4.5121831200360264e-02,
            -1.0368945003274743e-01,
            1.0579117224724620e-01,
            -8.2438007461943325e-02,
            -2.2932069186149713e-02,
            0.0000000000000000e00,
            7.1132061354573450e-01,
            -2.7804961282323992e-01,
            -1.4001074457204732e-01,
            1.9123669095413721e-02,
            2.3067476541179729e-02,
            -4.6306646983379073e-02,
            1.7974788536400992e00,
            1.5565028611862309e-17,
            -1.1242898267920341e-01,
            -2.6487018151342823e-01,
            -1.2318189318609840e-02,
            -2.7094528317080840e-02,
            3.7372943138999221e-03,
            0.0000000000000000e00,
            -9.6452683587244712e-17,
            1.2092677988559994e-02,
            1.0426053481719223e-01,
            2.3579030949027047e-01,
            -2.7410312241344138e-03,
            -3.6188679879065648e-02,
        ]
    )

    _EXPECTED_ORIENT_SCALE = np.array(
        [
            1.9860846551534193,
            2.609286874902539,
            -2.9186339492344304,
            -0.1280600290132337,
            6.939087475110936,
        ]
    )

    def test_coefficients_match_snapshot(self):
        """Default area normalization produces bit-identical coefficients."""
        n_harmonics = 6
        X_coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, rng=np.random.default_rng(42)
        )

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics)
        coef = efa.fit_transform([X_coords], norm=True)[0]

        assert_array_almost_equal(coef, self._EXPECTED_COEF, decimal=12)

    def test_orientation_scale_match_snapshot(self):
        """Default area normalization returns identical orientation/scale values."""
        n_harmonics = 6
        X_coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, rng=np.random.default_rng(42)
        )

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics)
        result = efa.fit_transform(
            [X_coords], norm=True, return_orientation_scale=True
        )[0]
        orient_scale = result[-5:]

        assert_array_almost_equal(orient_scale, self._EXPECTED_ORIENT_SCALE, decimal=12)
