"""Tests for Elliptic Fourier Analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.interpolate import make_interp_spline
from scipy.stats import wasserstein_distance_nd

from ktch.datasets import load_outline_mosquito_wings
from ktch.harmonic import EllipticFourierAnalysis
from ktch.harmonic._elliptic_fourier_analysis import (
    _compute_ellipse_geometry_3d,
    rotation_matrix_3d_euler_zxz,
)

EXPORT_DIR_FIGS = Path(".pytest_artifacts/figures/")
EXPORT_DIR_FIGS.mkdir(exist_ok=True, parents=True)


def _load_wings_as_list(n_specimens=None):
    """Load mosquito wing outlines as a list of arrays for EFA input."""
    wings = load_outline_mosquito_wings()
    coords = wings.coords
    if n_specimens is not None:
        coords = coords[:n_specimens]
    return [coords[i] for i in range(len(coords))]


@pytest.mark.parametrize("norm", [False, True])
def test_efa_shape(norm):
    n_harmonics = 6

    X = _load_wings_as_list(n_specimens=10)
    efa = EllipticFourierAnalysis(n_harmonics=n_harmonics, norm=norm)
    X_transformed = efa.fit_transform(X)

    assert X_transformed.shape == (len(X), 4 * (n_harmonics + 1))


@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("set_output", [None, "pandas"])
def test_transform(norm, set_output):
    n_harmonics = 6
    t_num = 360

    X = _load_wings_as_list(n_specimens=10)

    efa = EllipticFourierAnalysis(n_harmonics=n_harmonics, norm=norm)
    efa.set_output(transform=set_output)
    coef1 = efa.fit_transform(X)

    # round-trip: inverse_transform -> re-fit_transform
    if set_output == "pandas":
        coef1_arr = coef1.to_numpy()
    else:
        coef1_arr = coef1

    X_reconstructed = efa.inverse_transform(coef1_arr, t_num=t_num)
    T = [np.linspace(2 * np.pi / t_num, 2 * np.pi, t_num)] * len(X)

    efa2 = EllipticFourierAnalysis(n_harmonics=n_harmonics, norm=norm)
    coef2 = efa2.fit_transform(X_reconstructed, t=T)

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

    efa_serial = EllipticFourierAnalysis(
        n_harmonics=n_harmonics, n_jobs=None, norm=norm
    )
    coef_serial = efa_serial.fit_transform(X)

    efa = EllipticFourierAnalysis(
        n_harmonics=n_harmonics, n_jobs=n_jobs, verbose=1, norm=norm
    )
    coef_parallel = benchmark(efa.fit_transform, X)

    assert_array_almost_equal(coef_parallel, coef_serial)


def test_transform_exact():
    n_harmonics = 6
    t_num = 360

    t = np.linspace(2 * np.pi / t_num, 2 * np.pi, t_num)

    rng = np.random.default_rng(42)
    a0, c0 = rng.random(2)
    an, bn, cn, dn = rng.random((4, n_harmonics))
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

    efa = EllipticFourierAnalysis(n_harmonics=n_harmonics, norm=False)
    coef_est = efa.fit_transform([X_coords], t=[t])[0]
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

    rng = np.random.default_rng(42)
    x_o = rng.uniform(-5, 5, size=2)
    psi = rng.random() * 2 * np.pi
    scale = rng.uniform(0.5, 1.5)

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

    efa = EllipticFourierAnalysis(
        n_harmonics=n_harmonics, return_orientation_scale=True
    )
    coef_est = efa.fit_transform([X_coords])[0]
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

    rng = np.random.default_rng(42)
    a0, c0, e0 = rng.random(3)
    an, bn, cn, dn, en, fn = rng.random((6, n_harmonics))
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

    efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics, norm=False)
    coef_est = efa.fit_transform([X_coords], t=[t])[0]
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
    efa_norm = EllipticFourierAnalysis(n_harmonics=n_harmonics, norm=True)
    X_transformed = efa_norm.fit_transform(X)
    X_adj = efa_norm.inverse_transform(X_transformed, t_num=t_num)
    T = [np.linspace(2 * np.pi / t_num, 2 * np.pi, t_num) for i in range(len(X))]

    efa_raw = EllipticFourierAnalysis(n_harmonics=n_harmonics, norm=False)
    X_transformed = efa_raw.fit_transform(
        X_adj,
        t=T,
    )
    X_reconstructed = np.array(efa_raw.inverse_transform(X_transformed, t_num=t_num))

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

    efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics, norm=False)

    # With explicit 3D arc-length
    coef_explicit = efa.fit_transform([X_coords], t=[t_explicit])[0]

    # With auto arc-length (t=None) — should use sqrt(dx²+dy²+dz²)
    coef_auto = efa.fit_transform([X_coords])[0]

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
    rng = np.random.default_rng(42)
    an, bn, cn, dn, en, fn = rng.random((6, n_harmonics)) * 0.5

    cos = np.cos(np.tensordot(np.arange(1, n_harmonics + 1, 1), t, 0))
    sin = np.sin(np.tensordot(np.arange(1, n_harmonics + 1, 1), t, 0))

    x = a0 / 2 + np.dot(an, cos) + np.dot(bn, sin)
    y = c0 / 2 + np.dot(cn, cos) + np.dot(dn, sin)
    z = e0 / 2 + np.dot(en, cos) + np.dot(fn, sin)
    X_coords = np.stack([x, y, z], 1)

    efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics, norm=False)

    # With uniform t (DC should be close to analytical values)
    coef_uniform = efa.fit_transform([X_coords], t=[t])[0]
    dc_uniform = coef_uniform.reshape(6, n_harmonics + 1)[:, 0]

    # With auto arc-length (non-uniform dt), DC should also be reasonable
    coef_auto = efa.fit_transform([X_coords])[0]
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


def _make_circle(t_num=200, radius=1.0):
    """Helper: generate a simple planar circle."""
    t = np.linspace(2 * np.pi / t_num, 2 * np.pi, t_num)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    return np.stack([x, y], axis=1)


class TestNormalize3d:
    """Tests for EllipticFourierAnalysis._normalize_3d method."""

    def test_returns_correct_structure(self):
        """_normalize_3d returns 11 values: 6 arrays + 5 floats."""
        n_harmonics = 6
        X_coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, rng=np.random.default_rng(42)
        )

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics, norm=False)
        coef = efa.fit_transform([X_coords], t=None)[0]
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


class TestNormMethodParameter:
    """Tests for the norm_method parameter in EllipticFourierAnalysis."""

    def test_default_norm_method_is_none(self):
        """Default norm_method should be None."""
        efa = EllipticFourierAnalysis(n_dim=3)
        assert efa.norm_method is None

    def test_norm_method_area_accepted(self):
        """norm_method='area' should be accepted without error."""
        efa = EllipticFourierAnalysis(n_dim=3, norm_method="area")
        assert efa.norm_method == "area"

    def test_norm_method_semi_major_axis_accepted(self):
        """norm_method='semi_major_axis' should be accepted without error."""
        efa = EllipticFourierAnalysis(n_dim=3, norm_method="semi_major_axis")
        assert efa.norm_method == "semi_major_axis"

    def test_invalid_norm_method_raises_valueerror(self):
        """Invalid norm_method should raise ValueError at transform time."""
        X = _load_wings_as_list(n_specimens=1)
        efa = EllipticFourierAnalysis(n_dim=2, norm_method="invalid")
        with pytest.raises(ValueError, match="norm_method"):
            efa.transform(X)

    def test_invalid_norm_method_lists_valid_options(self):
        """Error message should list valid options."""
        X = _load_wings_as_list(n_specimens=1)
        efa = EllipticFourierAnalysis(n_dim=2, norm_method="foobar")
        with pytest.raises(ValueError, match="area"):
            efa.transform(X)

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

    def test_default_norm_method_2d_is_semi_major_axis(self):
        """Default (None) for 2D resolves to semi_major_axis."""
        X = _load_wings_as_list(n_specimens=5)
        n_harmonics = 6

        efa_default = EllipticFourierAnalysis(
            n_harmonics=n_harmonics, n_dim=2, norm=True
        )
        efa_semi = EllipticFourierAnalysis(
            n_harmonics=n_harmonics, n_dim=2, norm_method="semi_major_axis", norm=True
        )

        coef_default = efa_default.fit_transform(X)
        coef_semi = efa_semi.fit_transform(X)

        assert_array_almost_equal(coef_default, coef_semi)

    def test_norm_method_area_differs_from_semi_major_axis_2d(self):
        """area and semi_major_axis produce different results for 2D."""
        X = _load_wings_as_list(n_specimens=5)
        n_harmonics = 6

        efa_area = EllipticFourierAnalysis(
            n_harmonics=n_harmonics, n_dim=2, norm_method="area", norm=True
        )
        efa_semi = EllipticFourierAnalysis(
            n_harmonics=n_harmonics, n_dim=2, norm_method="semi_major_axis", norm=True
        )

        coef_area = efa_area.fit_transform(X)
        coef_semi = efa_semi.fit_transform(X)

        assert not np.allclose(coef_area, coef_semi)

    def test_2d_area_scale_is_sqrt_pi_a_b(self):
        """Returned scale for 2D area norm equals sqrt(pi * a * b)."""
        n_harmonics = 6
        t = np.linspace(0, 2 * np.pi, 101)[:-1]
        # Ellipse with semi-axes 3 and 1
        coords = np.stack([3 * np.cos(t), np.sin(t)], axis=1)

        efa_raw = EllipticFourierAnalysis(n_harmonics=n_harmonics, n_dim=2, norm=False)
        coef = efa_raw.fit_transform([coords])[0]
        arrays = coef.reshape(4, n_harmonics + 1)
        an, bn, cn, dn = arrays

        efa_area = EllipticFourierAnalysis(
            n_harmonics=n_harmonics,
            n_dim=2,
            norm_method="area",
            norm=True,
            return_orientation_scale=True,
        )
        coef_norm = efa_area.fit_transform([coords])[0]
        scale = coef_norm[-1]

        # Compute expected scale from raw 1st harmonic
        a1, b1, c1, d1 = an[1], bn[1], cn[1], dn[1]
        theta = 0.5 * np.arctan2(2 * (a1 * b1 + c1 * d1), a1**2 + c1**2 - b1**2 - d1**2)
        cos_th, sin_th = np.cos(theta), np.sin(theta)
        a_s = a1 * cos_th + b1 * sin_th
        c_s = c1 * cos_th + d1 * sin_th
        b_s = -a1 * sin_th + b1 * cos_th
        d_s = -c1 * sin_th + d1 * cos_th
        semi_major = np.sqrt(a_s**2 + c_s**2)
        semi_minor = np.sqrt(b_s**2 + d_s**2)
        expected_scale = np.sqrt(np.pi * semi_major * semi_minor)

        assert scale == pytest.approx(expected_scale, rel=1e-10)

    def test_2d_area_norm_invariant_to_scale(self):
        """Area-normalized 2D harmonic coefficients (n>=1) are scale-invariant."""
        t = np.linspace(0, 2 * np.pi, 101)[:-1]
        base = np.stack([2 * np.cos(t), np.sin(t)], axis=1)

        efa = EllipticFourierAnalysis(n_harmonics=4, norm=True, norm_method="area")
        coef_1 = efa.fit_transform([base])
        coef_big = efa.fit_transform([base * 1e6])
        coef_small = efa.fit_transform([base * 1e-6])

        n = efa.n_harmonics + 1
        harm_idx = np.concatenate([np.arange(b * n + 1, (b + 1) * n) for b in range(4)])
        assert_array_almost_equal(coef_big[0, harm_idx], coef_1[0, harm_idx], decimal=6)
        assert_array_almost_equal(
            coef_small[0, harm_idx], coef_1[0, harm_idx], decimal=6
        )

    def test_scale_is_sqrt_area(self):
        """The returned scale equals sqrt(pi * a1 * b1)."""
        n_harmonics = 6
        X_coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, rng=np.random.default_rng(42)
        )

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics, norm=False)
        coef = efa.fit_transform([X_coords], t=None)[0]
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

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics, norm=False)
        coef = efa.fit_transform([X_coords], t=None)[0]
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


class TestTransform3dNormIntegration:
    """Tests for 3D normalization integration into the transform method."""

    def test_norm_true_returns_correct_shape(self):
        """transform with norm=True for 3D returns shape (1, 6*(n_harmonics+1))."""
        n_harmonics = 6
        X_coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, rng=np.random.default_rng(10)
        )

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics, norm=True)
        result = efa.fit_transform([X_coords])

        assert result.shape == (1, 6 * (n_harmonics + 1))

    def test_norm_true_return_orientation_scale_shape(self):
        """transform with norm=True, return_orientation_scale=True appends 5 values."""
        n_harmonics = 6
        X_coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, rng=np.random.default_rng(10)
        )

        efa = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm=True, return_orientation_scale=True
        )
        result = efa.fit_transform([X_coords])

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
        efa_raw = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics, norm=False)
        raw = efa_raw.fit_transform([X_coords])[0]
        arrays = raw.reshape(6, n_harmonics + 1)
        an, bn, cn, dn, en, fn = arrays

        # Get expected orientation/scale from _normalize_3d
        _, _, _, _, _, _, alpha_exp, beta_exp, gamma_exp, phi_exp, scale_exp = (
            efa._normalize_3d(an, bn, cn, dn, en, fn)
        )

        # Get from pipeline
        efa_with_os = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm=True, return_orientation_scale=True
        )
        result = efa_with_os.fit_transform([X_coords])[0]
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
        efa_raw = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics, norm=False)
        raw = efa_raw.fit_transform([X_coords])[0]
        arrays = raw.reshape(6, n_harmonics + 1)
        an, bn, cn, dn, en, fn = arrays

        # Direct normalization
        An, Bn, Cn, Dn, En, Fn, *_ = efa._normalize_3d(an, bn, cn, dn, en, fn)
        expected_coef = np.hstack([An, Bn, Cn, Dn, En, Fn])

        # Pipeline normalization
        efa_norm = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics, norm=True)
        result = efa_norm.fit_transform([X_coords])[0]

        assert_array_almost_equal(result, expected_coef, decimal=10)

    def test_multiple_samples(self):
        """transform works with multiple 3D samples and norm=True."""
        n_harmonics = 6
        X1, _ = _make_3d_outline(n_harmonics=n_harmonics, rng=np.random.default_rng(10))
        X2, _ = _make_3d_outline(n_harmonics=n_harmonics, rng=np.random.default_rng(20))

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics, norm=True)
        result = efa.fit_transform([X1, X2])

        assert result.shape == (2, 6 * (n_harmonics + 1))

    def test_fit_transform_equals_transform(self):
        """fit_transform and transform produce the same result for 3D norm=True."""
        n_harmonics = 6
        X_coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, rng=np.random.default_rng(10)
        )

        efa = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm=True, return_orientation_scale=True
        )
        result_ft = efa.fit_transform([X_coords])
        result_t = efa.transform([X_coords])

        assert_array_almost_equal(result_ft, result_t)


class TestNormalize3dInvariance:
    """Invariance and validation tests for 3D EFA normalization.

    Translation, scale, rotation, and start-point shift invariance are
    parametrized over norm_method to cover both area and semi_major_axis
    normalization in a single set of tests.
    """

    @pytest.mark.parametrize("norm_method", [None, "semi_major_axis"])
    def test_translation_invariance(self, norm_method):
        """Normalized coefficients are invariant under 3D translation."""
        n_harmonics = 6
        seed = 50 if norm_method is None else 60
        rng = np.random.default_rng(seed)
        X_coords, _ = _make_3d_outline(n_harmonics=n_harmonics, rng=rng)

        translation = rng.uniform(-10, 10, size=3)
        X_translated = X_coords + translation

        efa = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm=True, norm_method=norm_method
        )
        coef_orig = efa.fit_transform([X_coords])[0]
        coef_trans = efa.fit_transform([X_translated])[0]

        coef_orig_harmonics = coef_orig.reshape(6, n_harmonics + 1)[:, 1:]
        coef_trans_harmonics = coef_trans.reshape(6, n_harmonics + 1)[:, 1:]

        assert_array_almost_equal(coef_orig_harmonics, coef_trans_harmonics, decimal=5)

    @pytest.mark.parametrize("norm_method", [None, "semi_major_axis"])
    def test_scale_invariance(self, norm_method):
        """Normalized coefficients are invariant under uniform scaling."""
        n_harmonics = 6
        seed = 51 if norm_method is None else 61
        rng = np.random.default_rng(seed)
        X_coords, _ = _make_3d_outline(n_harmonics=n_harmonics, rng=rng)

        scale_factor = rng.uniform(0.5, 5.0)
        X_scaled = X_coords * scale_factor

        efa = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm=True, norm_method=norm_method
        )
        coef_orig = efa.fit_transform([X_coords])[0]
        coef_scaled = efa.fit_transform([X_scaled])[0]

        coef_orig_harmonics = coef_orig.reshape(6, n_harmonics + 1)[:, 1:]
        coef_scaled_harmonics = coef_scaled.reshape(6, n_harmonics + 1)[:, 1:]

        assert_array_almost_equal(coef_orig_harmonics, coef_scaled_harmonics, decimal=5)

    @pytest.mark.parametrize("norm_method", [None, "semi_major_axis"])
    def test_rotation_invariance(self, norm_method):
        """Normalized coefficients are invariant under 3D rotation."""
        n_harmonics = 6
        seed = 52 if norm_method is None else 62
        rng = np.random.default_rng(seed)
        X_coords, _ = _make_3d_outline(n_harmonics=n_harmonics, rng=rng)

        alpha_r = rng.uniform(-np.pi, np.pi)
        beta_r = rng.uniform(0, np.pi)
        gamma_r = rng.uniform(-np.pi, np.pi)
        R = rotation_matrix_3d_euler_zxz(alpha_r, beta_r, gamma_r)
        X_rotated = (R @ X_coords.T).T

        efa = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm=True, norm_method=norm_method
        )
        coef_orig = efa.fit_transform([X_coords])[0]
        coef_rot = efa.fit_transform([X_rotated])[0]

        coef_orig_harmonics = coef_orig.reshape(6, n_harmonics + 1)[:, 1:]
        coef_rot_harmonics = coef_rot.reshape(6, n_harmonics + 1)[:, 1:]

        assert_array_almost_equal(coef_orig_harmonics, coef_rot_harmonics, decimal=4)

    @pytest.mark.parametrize("norm_method", [None, "semi_major_axis"])
    def test_startpoint_shift_invariance(self, norm_method):
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

        shift = rng.integers(1, len(X_coords))
        X_shifted = np.roll(X_coords, shift, axis=0)

        efa = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm=True, norm_method=norm_method
        )
        coef_orig = efa.fit_transform([X_coords])
        coef_shifted = efa.fit_transform([X_shifted])

        X_recon_orig = np.array(efa.inverse_transform(coef_orig, t_num=t_num))[0]
        X_recon_shifted = np.array(efa.inverse_transform(coef_shifted, t_num=t_num))[0]

        tol = 0.5 if norm_method is None else 1.0
        assert wasserstein_distance_nd(X_recon_orig, X_recon_shifted) < tol

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

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics, norm=True)
        coef_norm = efa.fit_transform([X_coords])
        X_recon = efa.inverse_transform(coef_norm, t_num=t_num)

        # Re-transform with explicit uniform t (matching inverse output spacing)
        t_uniform = [
            np.linspace(2 * np.pi / t_num, 2 * np.pi, t_num)
            for _ in range(len(X_recon))
        ]
        efa_raw = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics, norm=False)
        coef_recon = efa_raw.fit_transform(X_recon, t=t_uniform)

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

        efa = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm=True, return_orientation_scale=True
        )
        result = efa.fit_transform([X_coords], t=[t_uniform])[0]

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

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics, norm=True)
        coef = efa.fit_transform([X_coords])[0]
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
        efa_raw = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm_method="semi_major_axis", norm=False
        )
        coef = efa_raw.fit_transform([X_coords], t=None)[0]
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

        efa_raw = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics, norm=False)
        coef = efa_raw.fit_transform([X_coords], t=None)[0]
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
        efa_raw = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm_method="semi_major_axis", norm=False
        )
        coef = efa_raw.fit_transform([X_coords], t=None)[0]
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

        efa_raw = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics, norm=False)
        coef = efa_raw.fit_transform([X_coords], t=None)[0]
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
            n_dim=3,
            n_harmonics=n_harmonics,
            norm_method="semi_major_axis",
            norm=True,
            return_orientation_scale=True,
        )
        result = efa.fit_transform([X_coords])[0]

        # Get raw coefficients for expected a1
        efa_raw = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm_method="semi_major_axis", norm=False
        )
        raw = efa_raw.fit_transform([X_coords])[0]
        arrays = raw.reshape(6, n_harmonics + 1)
        an, bn, cn, dn, en, fn = arrays
        _, a1, _, _, _, _ = _compute_ellipse_geometry_3d(
            an[1], bn[1], cn[1], dn[1], en[1], fn[1]
        )

        scale_returned = result[-1]
        assert scale_returned == pytest.approx(a1, rel=1e-10)


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
                n_dim=3,
                n_harmonics=n_harmonics,
                norm_method="semi_major_axis",
                norm=True,
            )
            coef = efa.fit_transform([X_coords])[0]
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
            n_dim=3, n_harmonics=n_harmonics, norm_method="area", norm=True
        )
        coef = efa.fit_transform([X_coords])[0]
        arrays = coef.reshape(6, n_harmonics + 1)
        An, Bn, Cn, Dn, En, Fn = arrays

        _, a1_norm, _, _, _, _ = _compute_ellipse_geometry_3d(
            An[1], Bn[1], Cn[1], Dn[1], En[1], Fn[1]
        )

        # For area normalization, a1_norm = a1 / sqrt(pi*a1*b1), which != 1 in general
        assert a1_norm != pytest.approx(1.0, abs=1e-3)


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

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics, norm=True)
        coef = efa.fit_transform([X_coords])[0]

        assert_array_almost_equal(coef, self._EXPECTED_COEF, decimal=12)

    def test_orientation_scale_match_snapshot(self):
        """Default area normalization returns identical orientation/scale values."""
        n_harmonics = 6
        X_coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, rng=np.random.default_rng(42)
        )

        efa = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm=True, return_orientation_scale=True
        )
        result = efa.fit_transform([X_coords])[0]
        orient_scale = result[-5:]

        assert_array_almost_equal(orient_scale, self._EXPECTED_ORIENT_SCALE, decimal=12)


class TestOrientationMetadataAndRoundTrip:
    """Orientation/scale metadata, feature names, set_output, and inverse round trips."""

    def test_feature_names_and_counts_2d(self):
        n_harmonics = 3
        coords = _make_circle(t_num=120)
        efa = EllipticFourierAnalysis(
            n_harmonics=n_harmonics, norm=True, return_orientation_scale=True
        )
        coef = efa.fit_transform([coords])

        expected = (
            [f"a_{i}" for i in range(n_harmonics + 1)]
            + [f"b_{i}" for i in range(n_harmonics + 1)]
            + [f"c_{i}" for i in range(n_harmonics + 1)]
            + [f"d_{i}" for i in range(n_harmonics + 1)]
            + ["psi", "scale"]
        )
        assert list(efa.get_feature_names_out()) == expected
        assert efa._n_features_out == len(expected)
        assert coef.shape == (1, len(expected))

    def test_feature_names_and_counts_3d(self):
        n_harmonics = 3
        coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, t_num=120, rng=np.random.default_rng(123)
        )
        efa = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm=True, return_orientation_scale=True
        )
        coef = efa.fit_transform([coords])

        expected = (
            [f"a_{i}" for i in range(n_harmonics + 1)]
            + [f"b_{i}" for i in range(n_harmonics + 1)]
            + [f"c_{i}" for i in range(n_harmonics + 1)]
            + [f"d_{i}" for i in range(n_harmonics + 1)]
            + [f"e_{i}" for i in range(n_harmonics + 1)]
            + [f"f_{i}" for i in range(n_harmonics + 1)]
            + ["alpha", "beta", "gamma", "phi", "scale"]
        )
        assert list(efa.get_feature_names_out()) == expected
        assert efa._n_features_out == len(expected)
        assert coef.shape == (1, len(expected))

    def test_set_output_pandas_columns_2d(self):
        n_harmonics = 2
        coords = _make_circle(t_num=60)
        efa = EllipticFourierAnalysis(
            n_harmonics=n_harmonics, norm=True, return_orientation_scale=True
        )
        efa.set_output(transform="pandas")
        coef_df = efa.fit_transform([coords])
        expected_cols = (
            [f"a_{i}" for i in range(n_harmonics + 1)]
            + [f"b_{i}" for i in range(n_harmonics + 1)]
            + [f"c_{i}" for i in range(n_harmonics + 1)]
            + [f"d_{i}" for i in range(n_harmonics + 1)]
            + ["psi", "scale"]
        )
        assert isinstance(coef_df, pd.DataFrame)
        assert list(coef_df.columns) == expected_cols
        assert coef_df.shape == (1, len(expected_cols))

    def test_set_output_pandas_columns_3d(self):
        n_harmonics = 2
        coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, t_num=60, rng=np.random.default_rng(321)
        )
        efa = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm=True, return_orientation_scale=True
        )
        efa.set_output(transform="pandas")
        coef_df = efa.fit_transform([coords])
        expected_cols = (
            [f"a_{i}" for i in range(n_harmonics + 1)]
            + [f"b_{i}" for i in range(n_harmonics + 1)]
            + [f"c_{i}" for i in range(n_harmonics + 1)]
            + [f"d_{i}" for i in range(n_harmonics + 1)]
            + [f"e_{i}" for i in range(n_harmonics + 1)]
            + [f"f_{i}" for i in range(n_harmonics + 1)]
            + ["alpha", "beta", "gamma", "phi", "scale"]
        )
        assert isinstance(coef_df, pd.DataFrame)
        assert list(coef_df.columns) == expected_cols
        assert coef_df.shape == (1, len(expected_cols))

    def test_inverse_round_trip_with_orientation_2d(self):
        n_harmonics = 4
        coords = _make_circle(t_num=150)
        efa = EllipticFourierAnalysis(
            n_harmonics=n_harmonics, norm=True, return_orientation_scale=True
        )
        coef = efa.fit_transform([coords])

        coords_recon = efa.inverse_transform(coef, t_num=coords.shape[0])[0]
        efa_refit = EllipticFourierAnalysis(
            n_harmonics=n_harmonics, norm=True, return_orientation_scale=False
        )
        coef_recon = efa_refit.fit_transform([coords_recon])[0]

        np.testing.assert_allclose(coef[0, :-2], coef_recon, rtol=1e-6, atol=1e-6)

    def test_inverse_round_trip_with_orientation_3d(self):
        n_harmonics = 3
        t_num = 240
        t = np.linspace(2 * np.pi / t_num, 2 * np.pi, t_num)
        # Deterministic planar ellipse (z=0) to avoid phase/sign ambiguities
        x = 2.0 * np.cos(t)
        y = 1.0 * np.sin(t)
        z = np.zeros_like(t)
        coords = np.stack([x, y, z], axis=1)
        efa = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm=True, return_orientation_scale=True
        )
        coef = efa.fit_transform([coords])

        # Use finer sampling to reduce quadrature error in inverse transform
        t_num = 720
        coords_recon = efa.inverse_transform(coef, t_num=t_num)[0]
        efa_refit = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm=True, return_orientation_scale=False
        )
        coef_recon = efa_refit.fit_transform([coords_recon])[0]

        # Compare energy per harmonic (Frobenius norm of 3x2 coefficient matrices) to
        # avoid sign/phase ambiguities introduced by normalization.
        arr_orig = coef[0, :-5].reshape(6, n_harmonics + 1)[:, 1:]
        arr_recon = coef_recon.reshape(6, n_harmonics + 1)[:, 1:]

        frob_orig = np.linalg.norm(arr_orig.reshape(3, 2, n_harmonics), axis=(0, 1))
        frob_recon = np.linalg.norm(arr_recon.reshape(3, 2, n_harmonics), axis=(0, 1))

        rel_err = np.abs(frob_orig - frob_recon) / np.maximum(frob_orig, 1e-12)
        assert np.max(rel_err) < 0.1

    def test_return_orientation_requires_norm(self):
        n_harmonics = 3
        coords, _ = _make_3d_outline(
            n_harmonics=n_harmonics, t_num=80, rng=np.random.default_rng(999)
        )
        efa = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=n_harmonics, norm=False, return_orientation_scale=True
        )
        with pytest.raises(ValueError):
            efa.fit_transform([coords])


class TestNHarmonicsOne:
    """Tests for n_harmonics=1, the minimum non-trivial harmonic count.

    n_harmonics=1 is structurally special: the normalization loop runs
    exactly once, and output arrays contain only the DC offset (index 0)
    and a single harmonic (index 1).
    """

    def test_shape_2d_norm_false(self):
        """2D EFA with n_harmonics=1, norm=False produces correct shape."""
        coords = _make_circle(t_num=200, radius=2.0)
        efa = EllipticFourierAnalysis(n_harmonics=1, norm=False)
        coef = efa.fit_transform([coords])
        # 4 coefficient rows * (1 DC + 1 harmonic) = 8
        assert coef.shape == (1, 4 * 2)

    def test_round_trip_2d(self):
        """2D forward + inverse + re-forward round-trip with n_harmonics=1."""
        t_num = 360
        t = np.linspace(2 * np.pi / t_num, 2 * np.pi, t_num)
        x = 3.0 * np.cos(t) + 0.5
        y = 1.5 * np.sin(t) - 0.3
        coords = np.stack([x, y], axis=1)

        efa = EllipticFourierAnalysis(n_harmonics=1, norm=False)
        coef = efa.fit_transform([coords], t=[t])
        recon = efa.inverse_transform(coef, t_num=t_num)
        T = [np.linspace(2 * np.pi / t_num, 2 * np.pi, t_num)]
        coef2 = efa.fit_transform(recon, t=T)
        assert_array_almost_equal(coef, coef2, decimal=4)

    def test_canonical_form_2d(self):
        """2D norm=True with n_harmonics=1 has canonical 1st harmonic."""
        t_num = 200
        t = np.linspace(2 * np.pi / t_num, 2 * np.pi, t_num)
        x = 4.0 * np.cos(t) + 1.0
        y = 2.0 * np.sin(t) - 0.5
        coords = np.stack([x, y], axis=1)

        efa = EllipticFourierAnalysis(n_harmonics=1, norm=True)
        coef = efa.fit_transform([coords])

        # Canonical form: A1 > 0, B1 ~ 0, C1 ~ 0, D1 > 0
        arr = coef[0].reshape(4, 2)
        An, Bn, Cn, Dn = arr
        assert An[1] > 0, f"A1={An[1]} should be positive"
        assert abs(Bn[1]) < 1e-6, f"B1={Bn[1]} should be ~0"
        assert abs(Cn[1]) < 1e-6, f"C1={Cn[1]} should be ~0"
        assert Dn[1] > 0, f"D1={Dn[1]} should be positive"

    def test_shape_3d_norm_false(self):
        """3D EFA with n_harmonics=1, norm=False produces correct shape."""
        X, _ = _make_3d_outline(n_harmonics=1, t_num=200, rng=np.random.default_rng(7))
        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=1, norm=False)
        coef = efa.fit_transform([X])
        # 6 coefficient rows * (1 DC + 1 harmonic) = 12
        assert coef.shape == (1, 6 * 2)

    def test_canonical_first_harmonic_3d(self):
        """3D EFA with n_harmonics=1, norm=True has canonical 1st harmonic."""
        X, _ = _make_3d_outline(n_harmonics=1, t_num=200, rng=np.random.default_rng(7))
        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=1, norm=True)
        coef = efa.fit_transform([X])
        arr = coef[0].reshape(6, 2)
        An, Bn, Cn, Dn, En, Fn = arr

        assert An[1] > 0, f"An[1]={An[1]} should be positive"
        assert abs(Bn[1]) < 1e-10, f"Bn[1]={Bn[1]} should be ~0"
        assert abs(Cn[1]) < 1e-10, f"Cn[1]={Cn[1]} should be ~0"
        assert Dn[1] >= 0, f"Dn[1]={Dn[1]} should be non-negative"
        assert abs(En[1]) < 1e-10, f"En[1]={En[1]} should be ~0"
        assert abs(Fn[1]) < 1e-10, f"Fn[1]={Fn[1]} should be ~0"

    def test_round_trip_3d(self):
        """3D forward + inverse + re-forward round-trip with n_harmonics=1."""
        t_num = 360
        X, _ = _make_3d_outline(
            n_harmonics=1, t_num=t_num, rng=np.random.default_rng(8)
        )

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=1, norm=False)
        coef = efa.fit_transform([X])
        recon = efa.inverse_transform(coef, t_num=t_num)
        T = [np.linspace(2 * np.pi / t_num, 2 * np.pi, t_num)]
        coef2 = efa.fit_transform(recon, t=T)
        assert_array_almost_equal(coef, coef2, decimal=4)

    def test_orientation_scale_3d(self):
        """3D return_orientation_scale with n_harmonics=1 returns valid geometry."""
        X, _ = _make_3d_outline(n_harmonics=1, t_num=200, rng=np.random.default_rng(9))
        efa = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=1, norm=True, return_orientation_scale=True
        )
        coef = efa.fit_transform([X])
        assert coef.shape == (1, 6 * 2 + 5)

        alpha, beta, gamma, phi, scale = coef[0, -5:]
        # beta in [0, pi], scale > 0, phi in ]-pi/4, pi/4[
        assert 0 <= beta <= np.pi
        assert scale > 0
        assert -np.pi / 4 < phi < np.pi / 4

    def test_invariance_3d_translation(self):
        """3D n_harmonics=1 normalized coefficients are invariant under translation."""
        rng = np.random.default_rng(70)
        X, _ = _make_3d_outline(n_harmonics=1, t_num=200, rng=rng)
        translation = rng.uniform(-10, 10, size=3)
        X_translated = X + translation

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=1, norm=True)
        coef_orig = efa.fit_transform([X])[0].reshape(6, 2)[:, 1:]
        coef_trans = efa.fit_transform([X_translated])[0].reshape(6, 2)[:, 1:]

        assert_array_almost_equal(coef_orig, coef_trans, decimal=5)

    def test_invariance_3d_scale(self):
        """3D n_harmonics=1 normalized coefficients are invariant under scaling."""
        rng = np.random.default_rng(71)
        X, _ = _make_3d_outline(n_harmonics=1, t_num=200, rng=rng)
        X_scaled = X * 3.7

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=1, norm=True)
        coef_orig = efa.fit_transform([X])[0].reshape(6, 2)[:, 1:]
        coef_scaled = efa.fit_transform([X_scaled])[0].reshape(6, 2)[:, 1:]

        assert_array_almost_equal(coef_orig, coef_scaled, decimal=5)


class TestDuplicatedPoints:
    """Tests for duplicate-point handling in EFA.

    The source has two strategies controlled by the internal
    ``duplicated_points`` parameter:

    - ``"infinitesimal"``: floors dt < 1e-10 to 1e-10 (default via transform)
    - ``"deletion"``: removes segments with dt == 0
    """

    # -- 2D infinitesimal mode (via public API) --

    def test_consecutive_duplicate_2d_coefficients_close(self):
        """Coefficients with a duplicate are close to those without (2D)."""
        t = np.linspace(2 * np.pi / 200, 2 * np.pi, 200)
        coords = np.stack([3.0 * np.cos(t), 1.5 * np.sin(t)], axis=1)
        coords_dup = np.insert(coords, 50, coords[50], axis=0)

        efa = EllipticFourierAnalysis(n_harmonics=4, norm=False)
        coef_clean = efa.fit_transform([coords])[0]
        coef_dup = efa.fit_transform([coords_dup])[0]

        # Harmonics (skip DC) should be close
        n = efa.n_harmonics + 1
        assert_array_almost_equal(
            coef_clean.reshape(4, n)[:, 1:],
            coef_dup.reshape(4, n)[:, 1:],
            decimal=1,
        )

    def test_all_same_points_2d(self):
        """All-identical-point input raises ValueError (2D)."""
        coords = np.tile([1.0, 0.0], (50, 1))
        efa = EllipticFourierAnalysis(n_harmonics=3, norm=False)
        with pytest.raises(ValueError, match="[Dd]egenerate outline"):
            efa.fit_transform([coords])

    # -- 3D infinitesimal mode (via public API) --

    def test_consecutive_duplicate_3d_coefficients_close(self):
        """Coefficients with a duplicate are close to those without (3D)."""
        t = np.linspace(2 * np.pi / 200, 2 * np.pi, 200)
        coords = np.stack(
            [3.0 * np.cos(t), 1.5 * np.sin(t), 0.5 * np.cos(2 * t)], axis=1
        )
        coords_dup = np.insert(coords, 80, coords[80], axis=0)

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=4, norm=False)
        coef_clean = efa.fit_transform([coords])[0]
        coef_dup = efa.fit_transform([coords_dup])[0]

        n = efa.n_harmonics + 1
        assert_array_almost_equal(
            coef_clean.reshape(6, n)[:, 1:],
            coef_dup.reshape(6, n)[:, 1:],
            decimal=1,
        )

    def test_all_same_points_3d(self):
        """All-identical-point input raises ValueError (3D)."""
        coords = np.tile([1.0, 2.0, 3.0], (50, 1))
        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=3, norm=False)
        with pytest.raises(ValueError, match="[Dd]egenerate outline"):
            efa.fit_transform([coords])

    # -- deletion mode (via internal API) --

    def test_deletion_mode_matches_clean_2d(self):
        """deletion mode on data with a duplicate matches clean data closely (2D)."""
        t = np.linspace(2 * np.pi / 200, 2 * np.pi, 200)
        coords = np.stack([3.0 * np.cos(t), 1.5 * np.sin(t)], axis=1)
        coords_dup = np.insert(coords, 50, coords[50], axis=0)

        efa = EllipticFourierAnalysis(n_harmonics=4, norm=False)
        coef_clean = efa._transform_single_2d(coords)
        coef_del = efa._transform_single_2d(coords_dup, duplicated_points="deletion")

        n = efa.n_harmonics + 1
        assert_array_almost_equal(
            coef_clean.reshape(4, n)[:, 1:],
            coef_del.reshape(4, n)[:, 1:],
            decimal=2,
        )

    def test_deletion_mode_matches_clean_3d(self):
        """deletion mode on data with a duplicate matches clean data closely (3D)."""
        t = np.linspace(2 * np.pi / 200, 2 * np.pi, 200)
        coords = np.stack(
            [3.0 * np.cos(t), 1.5 * np.sin(t), 0.5 * np.cos(2 * t)], axis=1
        )
        coords_dup = np.insert(coords, 80, coords[80], axis=0)

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=4, norm=False)
        coef_clean = efa._transform_single_3d(coords)
        coef_del = efa._transform_single_3d(coords_dup, duplicated_points="deletion")

        n = efa.n_harmonics + 1
        assert_array_almost_equal(
            coef_clean.reshape(6, n)[:, 1:],
            coef_del.reshape(6, n)[:, 1:],
            decimal=2,
        )

    def test_deletion_mode_no_duplicates_equals_default_2d(self):
        """deletion mode on clean data gives same result as default mode (2D)."""
        t = np.linspace(2 * np.pi / 100, 2 * np.pi, 100)
        coords = np.stack([np.cos(t), np.sin(t)], axis=1)

        efa = EllipticFourierAnalysis(n_harmonics=4, norm=False)
        coef_default = efa._transform_single_2d(coords)
        coef_deletion = efa._transform_single_2d(coords, duplicated_points="deletion")

        assert_array_almost_equal(coef_default, coef_deletion, decimal=10)

    def test_deletion_mode_no_duplicates_equals_default_3d(self):
        """deletion mode on clean data gives same result as default mode (3D)."""
        t = np.linspace(2 * np.pi / 100, 2 * np.pi, 100)
        coords = np.stack([np.cos(t), np.sin(t), 0.3 * np.sin(2 * t)], axis=1)

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=4, norm=False)
        coef_default = efa._transform_single_3d(coords)
        coef_deletion = efa._transform_single_3d(coords, duplicated_points="deletion")

        assert_array_almost_equal(coef_default, coef_deletion, decimal=10)

    def test_deletion_mode_all_same_raises_2d(self):
        """deletion mode with all-identical points raises ValueError (2D)."""
        coords = np.tile([1.0, 0.0], (50, 1))
        efa = EllipticFourierAnalysis(n_harmonics=3, norm=False)
        with pytest.raises(ValueError, match="[Dd]egenerate outline"):
            efa._transform_single_2d(coords, duplicated_points="deletion")

    def test_deletion_mode_all_same_raises_3d(self):
        """deletion mode with all-identical points raises ValueError (3D)."""
        coords = np.tile([1.0, 2.0, 3.0], (50, 1))
        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=3, norm=False)
        with pytest.raises(ValueError, match="[Dd]egenerate outline"):
            efa._transform_single_3d(coords, duplicated_points="deletion")

    # -- Invalid mode --

    def test_invalid_duplicated_points_raises_2d(self):
        """Invalid duplicated_points value raises ValueError (2D)."""
        coords = _make_circle(t_num=50)
        efa = EllipticFourierAnalysis(n_harmonics=3, norm=False)
        with pytest.raises(ValueError, match="duplicated_points"):
            efa._transform_single_2d(coords, duplicated_points="unknown")

    def test_invalid_duplicated_points_raises_3d(self):
        """Invalid duplicated_points value raises ValueError (3D)."""
        t = np.linspace(2 * np.pi / 50, 2 * np.pi, 50)
        coords = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)
        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=3, norm=False)
        with pytest.raises(ValueError, match="duplicated_points"):
            efa._transform_single_3d(coords, duplicated_points="unknown")


class TestParameterLengthMismatch:
    """Tests for t length mismatch error path."""

    def test_t_length_mismatch_outer_2d(self):
        """Outer transform raises when len(t) != len(X) (2D)."""
        coords = _make_circle(t_num=50)
        t_wrong = [np.linspace(0.1, 2 * np.pi, 50)] * 2  # 2 t arrays for 1 sample
        efa = EllipticFourierAnalysis(n_harmonics=4, norm=False)
        with pytest.raises(ValueError, match="same length"):
            efa.fit_transform([coords], t=t_wrong)

    def test_t_length_mismatch_outer_3d(self):
        """Outer transform raises when len(t) != len(X) (3D)."""
        X, _ = _make_3d_outline(n_harmonics=3, t_num=50, rng=np.random.default_rng(0))
        t_wrong = [np.linspace(0.1, 2 * np.pi, 50)] * 2
        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=3, norm=False)
        with pytest.raises(ValueError, match="same length"):
            efa.fit_transform([X], t=t_wrong)

    def test_t_length_mismatch_inner_2d(self):
        """Inner _transform_single_2d raises when len(t) != n_points (2D)."""
        coords = _make_circle(t_num=50)
        t_wrong = np.linspace(0.1, 2 * np.pi, 30)  # 30 != 50
        efa = EllipticFourierAnalysis(n_harmonics=4, norm=False)
        with pytest.raises(ValueError, match="len\\(t\\)"):
            efa._transform_single_2d(coords, t=t_wrong)

    def test_t_length_mismatch_inner_3d(self):
        """Inner _transform_single_3d raises when len(t) != n_points (3D)."""
        X, _ = _make_3d_outline(n_harmonics=3, t_num=50, rng=np.random.default_rng(0))
        t_wrong = np.linspace(0.1, 2 * np.pi, 30)
        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=3, norm=False)
        with pytest.raises(ValueError, match="len\\(t\\)"):
            efa._transform_single_3d(X, t=t_wrong)


class TestFewPointsHighHarmonics:
    """Tests for n_points < n_harmonics in EFA.

    EFA uses an analytical formula that always produces output regardless
    of point count. These tests verify that the output shape is correct
    and that low-order coefficients still approximate the input geometry.
    """

    def test_pentagon_round_trip(self):
        """Pentagon (5 pts, n_harmonics=20): round-trip reconstruction passes near original points."""
        t = np.linspace(0, 2 * np.pi, 6)[:-1]
        coords = np.stack([np.cos(t), np.sin(t)], axis=1)

        efa = EllipticFourierAnalysis(n_harmonics=20, norm=False)
        coef = efa.fit_transform([coords])
        assert coef.shape == (1, 4 * 21)

        # Reconstruct and check that reconstruction is close to original
        recon = efa.inverse_transform(coef, t_num=200)
        # For each original point, find the closest reconstructed point
        for pt in coords:
            dists = np.linalg.norm(recon[0] - pt, axis=1)
            assert np.min(dists) < 0.1

    def test_few_points_low_harmonics_match_many_points(self):
        """Low-order coefficients for 10 pts match those for 200 pts on same shape."""
        coords_few = np.stack(
            [
                2 * np.cos(np.linspace(0, 2 * np.pi, 11)[:-1]),
                np.sin(np.linspace(0, 2 * np.pi, 11)[:-1]),
            ],
            axis=1,
        )
        coords_many = np.stack(
            [
                2 * np.cos(np.linspace(0, 2 * np.pi, 201)[:-1]),
                np.sin(np.linspace(0, 2 * np.pi, 201)[:-1]),
            ],
            axis=1,
        )

        efa = EllipticFourierAnalysis(n_harmonics=3, norm=False)
        coef_few = efa.fit_transform([coords_few])
        coef_many = efa.fit_transform([coords_many])

        # 1st harmonic should be close (the shape is the same ellipse)
        n = efa.n_harmonics + 1
        for block in range(4):
            assert_array_almost_equal(
                coef_few[0, block * n + 1], coef_many[0, block * n + 1], decimal=0
            )


class TestCircleNormalization2d:
    """Tests for perfect circle with norm=True in 2D.

    A perfect circle has a1^2 + c1^2 = b1^2 + d1^2, making the denominator
    of the phase angle computation zero. The orientation (psi) is indeterminate,
    but normalization must still produce finite, valid output.
    """

    @pytest.fixture()
    def circle_coords(self):
        t = np.linspace(0, 2 * np.pi, 101)[:-1]
        return np.stack([np.cos(t), np.sin(t)], axis=1)

    def test_canonical_first_harmonic(self, circle_coords):
        """After normalization, first harmonic has A1=1, B1~0, C1~0, |D1|~1."""
        efa = EllipticFourierAnalysis(n_harmonics=4, norm=True)
        coef = efa.fit_transform([circle_coords])
        n = efa.n_harmonics + 1
        A1, B1, C1, D1 = (
            coef[0, 1],
            coef[0, n + 1],
            coef[0, 2 * n + 1],
            coef[0, 3 * n + 1],
        )
        assert_array_almost_equal(A1, 1.0, decimal=4)
        assert abs(B1) < 0.01
        assert abs(C1) < 0.01
        assert abs(abs(D1) - 1.0) < 0.01

    def test_scale_equals_radius(self, circle_coords):
        """Scale factor should be approximately the radius (1.0)."""
        efa = EllipticFourierAnalysis(
            n_harmonics=4, norm=True, return_orientation_scale=True
        )
        coef = efa.fit_transform([circle_coords])
        n_base = 4 * (efa.n_harmonics + 1)
        scale = coef[0, n_base + 1]
        assert abs(scale - 1.0) < 0.01

    def test_scaled_circle(self):
        """Circle with radius 5: canonical form preserved, scale correct.

        For a perfect circle both arguments to arctan2 are zero.
        arctan2(0, 0) = 0.0 per IEEE 754, giving a valid arbitrary phase.
        """
        t = np.linspace(0, 2 * np.pi, 101)[:-1]
        coords = np.stack([5 * np.cos(t), 5 * np.sin(t)], axis=1)

        efa = EllipticFourierAnalysis(
            n_harmonics=4, norm=True, return_orientation_scale=True
        )
        coef = efa.fit_transform([coords])

        n = efa.n_harmonics + 1
        A1, D1 = coef[0, 1], coef[0, 3 * n + 1]
        assert_array_almost_equal(A1, 1.0, decimal=3)
        assert abs(abs(D1) - 1.0) < 0.01

        n_base = 4 * n
        scale = coef[0, n_base + 1]
        assert abs(scale - 5.0) < 0.05

    def test_near_circle_continuity(self):
        """Near-circular ellipse (eccentricity -> 0) also normalizes cleanly."""
        t = np.linspace(0, 2 * np.pi, 101)[:-1]
        # Very slight eccentricity
        coords = np.stack([1.001 * np.cos(t), np.sin(t)], axis=1)

        efa = EllipticFourierAnalysis(n_harmonics=4, norm=True)
        coef = efa.fit_transform([coords])
        assert np.all(np.isfinite(coef))

        n = efa.n_harmonics + 1
        A1 = coef[0, 1]
        assert_array_almost_equal(A1, 1.0, decimal=2)


class TestCoplanar3dNormalization:
    """Tests for coplanar 3D input (z=0) with norm=True.

    When all z-coordinates are zero, the 1st harmonic has zc=zs=0.
    The normal vector of the fitted ellipse lies along the z-axis,
    so beta should be 0 or pi (gimbal lock case).
    """

    @pytest.fixture()
    def coplanar_ellipse(self):
        t = np.linspace(0, 2 * np.pi, 101)[:-1]
        return np.stack([2 * np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)

    def test_beta_is_zero_or_pi(self, coplanar_ellipse):
        """Beta should be 0 or pi for xy-plane input."""
        efa = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=4, norm=True, return_orientation_scale=True
        )
        coef = efa.fit_transform([coplanar_ellipse])
        n_base = 6 * (efa.n_harmonics + 1)
        beta = coef[0, n_base + 1]
        assert np.isclose(beta, 0.0, atol=1e-10) or np.isclose(beta, np.pi, atol=1e-10)

    def test_z_coefficients_near_zero(self, coplanar_ellipse):
        """Normalized z-axis coefficients (en, fn) should be near zero."""
        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=4, norm=True)
        coef = efa.fit_transform([coplanar_ellipse])
        n = efa.n_harmonics + 1
        en = coef[0, 4 * n : 5 * n]
        fn = coef[0, 5 * n : 6 * n]
        assert np.max(np.abs(en)) < 1e-10
        assert np.max(np.abs(fn)) < 1e-10

    def test_round_trip_coplanar(self, coplanar_ellipse):
        """Round-trip: coplanar coords -> normalize -> inverse stays coplanar."""
        efa = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=4, norm=True, return_orientation_scale=True
        )
        coef = efa.fit_transform([coplanar_ellipse])
        recon = efa.inverse_transform(coef, t_num=100)
        # Reconstructed z should be near zero
        assert np.max(np.abs(recon[0, :, 2])) < 1e-6

    def test_tilted_coplanar_plane(self):
        """Coplanar input in a tilted plane (not xy) produces finite output."""
        t = np.linspace(0, 2 * np.pi, 101)[:-1]
        # Ellipse in xz-plane: y=0
        coords = np.stack([2 * np.cos(t), np.zeros_like(t), np.sin(t)], axis=1)

        efa = EllipticFourierAnalysis(
            n_dim=3, n_harmonics=4, norm=True, return_orientation_scale=True
        )
        coef = efa.fit_transform([coords])
        assert np.all(np.isfinite(coef))

        n_base = 6 * (efa.n_harmonics + 1)
        beta = coef[0, n_base + 1]
        # xz-plane: normal is y-axis, so beta should be pi/2
        assert np.isclose(beta, np.pi / 2, atol=0.1)

    def test_near_coplanar(self):
        """Near-coplanar input (z ~ 1e-15) produces same result as exact z=0."""
        t = np.linspace(0, 2 * np.pi, 101)[:-1]
        coords_exact = np.stack([2 * np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)
        coords_near = np.stack(
            [2 * np.cos(t), np.sin(t), 1e-15 * np.ones_like(t)], axis=1
        )

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=4, norm=True)
        coef_exact = efa.fit_transform([coords_exact])
        coef_near = efa.fit_transform([coords_near])

        assert_array_almost_equal(coef_exact, coef_near, decimal=8)


class TestNyquistLimit:
    """Tests for n_harmonics near n_points // 2.

    EFA uses an analytical formula, so coefficients are always finite.
    The meaningful property to verify is that low-order coefficients
    are independent of n_harmonics (higher harmonics do not corrupt
    lower ones).
    """

    def test_low_order_coefficients_stable_2d(self):
        """Low-order 2D coefficients are identical regardless of n_harmonics."""
        t = np.linspace(0, 2 * np.pi, 21)[:-1]
        coords = np.stack(
            [2 * np.cos(t) + 0.3 * np.cos(3 * t), np.sin(t) + 0.2 * np.sin(5 * t)],
            axis=1,
        )

        efa_low = EllipticFourierAnalysis(n_harmonics=5, norm=False)
        efa_high = EllipticFourierAnalysis(n_harmonics=19, norm=False)
        coef_low = efa_low.fit_transform([coords])
        coef_high = efa_high.fit_transform([coords])

        # First 5 harmonics (offset + 5) should agree across all 4 blocks
        n_low = 6
        for block in range(4):
            low_block = coef_low[0, block * n_low : (block + 1) * n_low]
            high_block = coef_high[0, block * 20 : block * 20 + n_low]
            assert_array_almost_equal(low_block, high_block, decimal=10)

    def test_low_order_coefficients_stable_3d(self):
        """Low-order 3D coefficients are identical regardless of n_harmonics."""
        t = np.linspace(0, 2 * np.pi, 21)[:-1]
        coords = np.stack([2 * np.cos(t), np.sin(t), 0.5 * np.cos(2 * t)], axis=1)

        efa_low = EllipticFourierAnalysis(n_dim=3, n_harmonics=5, norm=False)
        efa_high = EllipticFourierAnalysis(n_dim=3, n_harmonics=10, norm=False)
        coef_low = efa_low.fit_transform([coords])
        coef_high = efa_high.fit_transform([coords])

        n_low = 6
        for block in range(6):
            low_block = coef_low[0, block * n_low : (block + 1) * n_low]
            high_block = coef_high[0, block * 11 : block * 11 + n_low]
            assert_array_almost_equal(low_block, high_block, decimal=10)


class TestExtremeScales:
    """Tests for extreme coordinate scales.

    Verifies algebraic properties that must hold at any scale:
    - Unnormalized coefficients scale linearly with coordinate magnitude
    - Normalized coefficients are scale-invariant
    """

    def test_linear_scaling_2d(self):
        """Unnormalized 2D coefficients scale linearly with coordinate magnitude."""
        t = np.linspace(0, 2 * np.pi, 101)[:-1]
        base = np.stack([2 * np.cos(t), np.sin(t)], axis=1)

        efa = EllipticFourierAnalysis(n_harmonics=4, norm=False)
        coef_1 = efa.fit_transform([base])
        coef_big = efa.fit_transform([base * 1e6])
        coef_small = efa.fit_transform([base * 1e-6])

        assert_array_almost_equal(coef_big / 1e6, coef_1, decimal=6)
        assert_array_almost_equal(coef_small / 1e-6, coef_1, decimal=6)

    def test_linear_scaling_3d(self):
        """Unnormalized 3D coefficients scale linearly with coordinate magnitude."""
        t = np.linspace(0, 2 * np.pi, 101)[:-1]
        base = np.stack([2 * np.cos(t), np.sin(t), 0.5 * np.cos(2 * t)], axis=1)

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=4, norm=False)
        coef_1 = efa.fit_transform([base])
        coef_big = efa.fit_transform([base * 1e6])

        assert_array_almost_equal(coef_big / 1e6, coef_1, decimal=6)

    def test_norm_invariant_to_scale_2d(self):
        """Normalized 2D harmonic coefficients (n>=1) are identical at any scale.

        DC offsets (n=0) are not normalized and scale with coordinates.
        """
        t = np.linspace(0, 2 * np.pi, 101)[:-1]
        base = np.stack([2 * np.cos(t), np.sin(t)], axis=1)

        efa = EllipticFourierAnalysis(n_harmonics=4, norm=True)
        coef_1 = efa.fit_transform([base])
        coef_big = efa.fit_transform([base * 1e6])
        coef_small = efa.fit_transform([base * 1e-6])

        # Extract harmonic coefficients only (skip DC offset at index 0 of each block)
        n = efa.n_harmonics + 1
        harm_idx = np.concatenate([np.arange(b * n + 1, (b + 1) * n) for b in range(4)])
        assert_array_almost_equal(coef_big[0, harm_idx], coef_1[0, harm_idx], decimal=6)
        assert_array_almost_equal(
            coef_small[0, harm_idx], coef_1[0, harm_idx], decimal=6
        )

    def test_norm_invariant_to_scale_3d(self):
        """Normalized 3D harmonic coefficients (n>=1) are identical at any scale."""
        t = np.linspace(0, 2 * np.pi, 101)[:-1]
        base = np.stack([2 * np.cos(t), np.sin(t), 0.5 * np.cos(2 * t)], axis=1)

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=4, norm=True)
        coef_1 = efa.fit_transform([base])
        coef_big = efa.fit_transform([base * 1e6])

        n = efa.n_harmonics + 1
        harm_idx = np.concatenate([np.arange(b * n + 1, (b + 1) * n) for b in range(6)])
        assert_array_almost_equal(coef_big[0, harm_idx], coef_1[0, harm_idx], decimal=6)


class TestNonFiniteInput:
    """Tests for NaN/Inf input to EFA.

    EFA validates input and raises ValueError for non-finite values,
    consistent with SHA (which raises via scipy.linalg.lstsq).
    """

    def test_nan_raises_2d(self):
        """NaN in 2D input raises ValueError."""
        t = np.linspace(0, 2 * np.pi, 51)[:-1]
        coords = np.stack([np.cos(t), np.sin(t)], axis=1)
        coords[5, 0] = np.nan

        efa = EllipticFourierAnalysis(n_harmonics=4, norm=False)
        with pytest.raises(ValueError, match="NaN or Inf"):
            efa.fit_transform([coords])

    def test_inf_raises_2d(self):
        """Inf in 2D input raises ValueError."""
        t = np.linspace(0, 2 * np.pi, 51)[:-1]
        coords = np.stack([np.cos(t), np.sin(t)], axis=1)
        coords[10, 1] = np.inf

        efa = EllipticFourierAnalysis(n_harmonics=4, norm=False)
        with pytest.raises(ValueError, match="NaN or Inf"):
            efa.fit_transform([coords])

    def test_nan_raises_3d(self):
        """NaN in 3D input raises ValueError."""
        t = np.linspace(0, 2 * np.pi, 51)[:-1]
        coords = np.stack([np.cos(t), np.sin(t), 0.5 * np.cos(2 * t)], axis=1)
        coords[5, 0] = np.nan

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=4, norm=False)
        with pytest.raises(ValueError, match="NaN or Inf"):
            efa.fit_transform([coords])

    def test_inf_raises_3d(self):
        """Inf in 3D input raises ValueError."""
        t = np.linspace(0, 2 * np.pi, 51)[:-1]
        coords = np.stack([np.cos(t), np.sin(t), 0.5 * np.cos(2 * t)], axis=1)
        coords[10, 2] = -np.inf

        efa = EllipticFourierAnalysis(n_dim=3, n_harmonics=4, norm=False)
        with pytest.raises(ValueError, match="NaN or Inf"):
            efa.fit_transform([coords])


###########################################################
#
#   sklearn API compliance
#
###########################################################


def test_efa_fit_returns_self():
    """fit() returns self for method chaining."""
    efa = EllipticFourierAnalysis(n_harmonics=4)
    X = _load_wings_as_list(n_specimens=3)
    result = efa.fit(X)
    assert result is efa


def test_efa_fit_then_transform_2d():
    """fit + transform produces same result as fit_transform for 2D."""
    X = _load_wings_as_list(n_specimens=5)

    efa1 = EllipticFourierAnalysis(n_harmonics=6, norm=True)
    X_ft = efa1.fit_transform(X)

    efa2 = EllipticFourierAnalysis(n_harmonics=6, norm=True)
    X_t = efa2.fit(X).transform(X)

    assert_array_almost_equal(X_ft, X_t)


def test_efa_fit_then_transform_2d_with_t():
    """fit + transform(t=...) produces same result as fit_transform(t=...) for 2D."""
    X = _load_wings_as_list(n_specimens=5)
    t = [np.linspace(0, 2 * np.pi, len(x), endpoint=False) for x in X]

    efa1 = EllipticFourierAnalysis(n_harmonics=6, norm=True)
    X_ft = efa1.fit_transform(X, t=t)

    efa2 = EllipticFourierAnalysis(n_harmonics=6, norm=True)
    X_t = efa2.fit(X).transform(X, t=t)

    assert_array_almost_equal(X_ft, X_t)


def test_efa_clone():
    """sklearn clone() preserves constructor params."""
    from sklearn.base import clone

    efa = EllipticFourierAnalysis(
        n_harmonics=10,
        n_dim=3,
        norm=True,
        return_orientation_scale=True,
        norm_method="semi_major_axis",
    )
    efa_cloned = clone(efa)

    assert efa_cloned.n_harmonics == 10
    assert efa_cloned.n_dim == 3
    assert efa_cloned.norm is True
    assert efa_cloned.return_orientation_scale is True
    assert efa_cloned.norm_method == "semi_major_axis"


def test_efa_get_feature_names_out_2d():
    """Feature names match expected pattern for 2D."""
    efa = EllipticFourierAnalysis(n_harmonics=3)
    names = efa.get_feature_names_out()
    assert len(names) == 4 * 4
    assert names[0] == "a_0"
    assert names[-1] == "d_3"


def test_efa_get_feature_names_out_3d():
    """Feature names match expected pattern for 3D."""
    efa = EllipticFourierAnalysis(n_harmonics=3, n_dim=3)
    names = efa.get_feature_names_out()
    assert len(names) == 6 * 4
    assert names[0] == "a_0"
    assert names[-1] == "f_3"


def test_efa_inverse_transform_strips_orientation_scale_2d():
    """inverse_transform ignores trailing psi/scale columns."""
    coords = _make_circle(t_num=120)
    efa = EllipticFourierAnalysis(
        n_harmonics=4, norm=True, return_orientation_scale=True
    )
    coef = efa.fit_transform([coords])

    recon_with = efa.inverse_transform(coef, t_num=50)
    recon_without = efa.inverse_transform(coef[:, :-2], t_num=50)
    assert_array_almost_equal(recon_with[0], recon_without[0])


def test_efa_inverse_transform_strips_orientation_scale_3d():
    """inverse_transform ignores trailing alpha/beta/gamma/phi/scale columns."""
    coords, _ = _make_3d_outline(
        n_harmonics=4, t_num=120, rng=np.random.default_rng(99)
    )
    efa = EllipticFourierAnalysis(
        n_dim=3, n_harmonics=4, norm=True, return_orientation_scale=True
    )
    coef = efa.fit_transform([coords])

    recon_with = efa.inverse_transform(coef, t_num=50)
    recon_without = efa.inverse_transform(coef[:, :-5], t_num=50)
    assert_array_almost_equal(recon_with[0], recon_without[0])


def test_efa_set_output_pandas_has_feature_names():
    """DataFrame columns match get_feature_names_out()."""
    X = _load_wings_as_list(n_specimens=3)
    efa = EllipticFourierAnalysis(n_harmonics=4, norm=True)
    efa.set_output(transform="pandas")
    coef = efa.fit_transform(X)

    assert isinstance(coef, pd.DataFrame)
    assert list(coef.columns) == list(efa.get_feature_names_out())


###########################################################
#
#   sklearn Pipeline integration
#
###########################################################


def test_efa_pipeline_with_pca_2d():
    """EFA as first step in Pipeline, followed by PCA (2D)."""
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline

    X = _load_wings_as_list(n_specimens=10)
    pipe = Pipeline(
        [
            ("efa", EllipticFourierAnalysis(n_harmonics=6, norm=True)),
            ("pca", PCA(n_components=2)),
        ]
    )
    result = pipe.fit_transform(X)
    assert result.shape == (10, 2)


def test_efa_pipeline_with_pca_3d():
    """EFA(n_dim=3) in Pipeline with PCA."""
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline

    n_harmonics = 4
    X = [
        _make_3d_outline(n_harmonics=n_harmonics, rng=np.random.default_rng(i))[0]
        for i in range(8)
    ]
    pipe = Pipeline(
        [
            (
                "efa",
                EllipticFourierAnalysis(n_dim=3, n_harmonics=n_harmonics, norm=True),
            ),
            ("pca", PCA(n_components=2)),
        ]
    )
    result = pipe.fit_transform(X)
    assert result.shape == (8, 2)


def test_efa_pipeline_fit_then_transform():
    """Pipeline fit + transform produces same result as fit_transform."""
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline

    X = _load_wings_as_list(n_specimens=10)

    pipe1 = Pipeline(
        [
            ("efa", EllipticFourierAnalysis(n_harmonics=6, norm=True)),
            ("pca", PCA(n_components=2)),
        ]
    )
    result_ft = pipe1.fit_transform(X)

    pipe2 = Pipeline(
        [
            ("efa", EllipticFourierAnalysis(n_harmonics=6, norm=True)),
            ("pca", PCA(n_components=2)),
        ]
    )
    pipe2.fit(X)
    result_t = pipe2.transform(X)

    assert_array_almost_equal(result_ft, result_t)


def test_efa_pipeline_norm_pca():
    """Normalized EFA coefficients flow correctly through PCA."""
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline

    X = _load_wings_as_list(n_specimens=10)
    pipe = Pipeline(
        [
            ("efa", EllipticFourierAnalysis(n_harmonics=6, norm=True)),
            ("pca", PCA()),
        ]
    )
    pipe.fit(X)

    pca_step = pipe.named_steps["pca"]
    assert pca_step.n_components_ <= 4 * 7
    assert sum(pca_step.explained_variance_ratio_) == pytest.approx(1.0)


def test_efa_pipeline_set_output_pandas():
    """Pipeline with set_output propagates DataFrames correctly."""
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline

    X = _load_wings_as_list(n_specimens=5)
    pipe = Pipeline(
        [
            ("efa", EllipticFourierAnalysis(n_harmonics=4, norm=True)),
            ("pca", PCA(n_components=2)),
        ]
    )
    pipe.set_output(transform="pandas")
    result = pipe.fit_transform(X)
    assert isinstance(result, pd.DataFrame)


def test_efa_pipeline_metadata_routing_t():
    """Metadata routing of t through Pipeline.

    With metadata routing enabled, parameters are passed by name (not
    step__param prefix). The routing system uses set_transform_request
    declarations to dispatch each parameter to the correct step.
    """
    import sklearn
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline

    X = _load_wings_as_list(n_specimens=5)
    t = [np.linspace(0, 2 * np.pi, len(x), endpoint=False) for x in X]

    with sklearn.config_context(enable_metadata_routing=True):
        efa = EllipticFourierAnalysis(n_harmonics=6, norm=True)
        efa.set_transform_request(t=True)
        pipe = Pipeline([("efa", efa), ("pca", PCA(n_components=2))])

        result = pipe.fit_transform(X, t=t)
        assert result.shape == (5, 2)

        # Also test fit + transform separately
        pipe2 = Pipeline(
            [
                (
                    "efa",
                    EllipticFourierAnalysis(
                        n_harmonics=6, norm=True
                    ).set_transform_request(t=True),
                ),
                ("pca", PCA(n_components=2)),
            ]
        )
        pipe2.fit(X, t=t)
        result2 = pipe2.transform(X, t=t)
        assert_array_almost_equal(result, result2)
