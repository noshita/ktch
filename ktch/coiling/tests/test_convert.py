"""Tests for Raup <-> growing-tube conversion and surface-level consistency."""

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from ktch.coiling import l_g, l_r, raup
from ktch.coiling._convert import (
    growing_tube_aperture_to_raup,
    growing_tube_to_raup,
    raup_aperture_to_growing_tube,
    raup_to_growing_tube,
)
from ktch.coiling._generating_curve import _assemble_surface
from ktch.coiling._growing_tube import _growing_tube_trajectory_ode
from ktch.coiling._raup import _rotation_x, _rotation_z


def test_import_smoke():
    import ktch

    assert hasattr(ktch, "coiling")
    from ktch.coiling import GrowingTubeModel, RaupModel, growing_tube  # noqa: F401


# --- locus conversion -------------------------------------------------------


def test_planispiral_has_zero_torsion():
    e_g, c_g, t_g = raup_to_growing_tube(1.6, 0.0, 0.5)
    assert t_g == 0.0
    assert e_g > 0.0
    assert c_g > 0.0


@pytest.mark.parametrize(
    "w_r, t_r, d_r",
    [
        (1.5, 2.6, 0.6),
        (1.8, 1.0, 0.4),
        (1.3, 0.5, 0.2),
        (2.0, 3.0, 0.7),
        (1.6, 1.0, -0.3),
    ],
)
def test_locus_round_trip(w_r, t_r, d_r):
    e_g, c_g, t_g = raup_to_growing_tube(w_r, t_r, d_r)
    w2, t2, d2 = growing_tube_to_raup(e_g, c_g, t_g)
    np.testing.assert_allclose([w2, t2, d2], [w_r, t_r, d_r], rtol=1e-4, atol=1e-6)


def test_growing_tube_to_raup_straight_raises():
    with pytest.raises(ValueError):
        growing_tube_to_raup(0.1, 0.0, 0.05)


def _raup_locus_standardized(w_r, t_r, d_r, r0=1.0, n_whorls=2.0, n=4000):
    theta = np.linspace(0.0, 2.0 * np.pi * n_whorls, n)
    dtheta = theta[1] - theta[0]
    vx = 2.0 * d_r / (1.0 - d_r) + 1.0
    vz = 2.0 * t_r * (d_r / (1.0 - d_r) + 1.0)
    scale = r0 * w_r ** (theta / (2.0 * np.pi))
    p = np.column_stack(
        [scale * vx * np.cos(theta), scale * vx * np.sin(theta), scale * vz]
    )
    radius = scale
    p1 = np.gradient(p, dtheta, axis=0)
    p2 = np.gradient(p1, dtheta, axis=0)
    p3 = np.gradient(p2, dtheta, axis=0)
    cross12 = np.cross(p1, p2)
    norm_cross = np.linalg.norm(cross12, axis=1)
    speed = np.linalg.norm(p1, axis=1)
    kappa = norm_cross / speed**3
    tau = np.sum(cross12 * p3, axis=1) / norm_cross**2
    arclength = cumulative_trapezoid(speed, theta, initial=0.0)
    sl = slice(int(0.2 * n), int(0.8 * n))
    e_g = np.polyfit(arclength[sl], radius[sl], 1)[0]
    return e_g, np.median((radius * kappa)[sl]), np.median((radius * tau)[sl])


@pytest.mark.parametrize(
    "w_r, t_r, d_r", [(1.5, 2.6, 0.6), (1.8, 1.0, 0.4), (1.6, 0.8, 0.3)]
)
def test_locus_consistency_geometry(w_r, t_r, d_r):
    e_g, c_g, t_g = raup_to_growing_tube(w_r, t_r, d_r)
    e_num, c_num, t_num = _raup_locus_standardized(w_r, t_r, d_r)
    np.testing.assert_allclose(e_g, e_num, rtol=2e-2)
    np.testing.assert_allclose(c_g, c_num, rtol=5e-2)
    np.testing.assert_allclose(t_g, t_num, rtol=5e-2, atol=1e-3)


# --- aperture-orientation conversion ----------------------------------------


def _raup_frenet0(w_r, t_r, d_r):
    th = np.linspace(-0.02, 0.02, 401)
    dth = th[1] - th[0]
    vx = 2 * d_r / (1 - d_r) + 1
    vz = 2 * t_r * (d_r / (1 - d_r) + 1)
    sc = w_r ** (th / (2 * np.pi))
    p = np.column_stack([sc * vx * np.cos(th), sc * vx * np.sin(th), sc * vz])
    d1 = np.gradient(p, dth, axis=0)
    xi1 = d1 / np.linalg.norm(d1, axis=1, keepdims=True)
    d2 = np.gradient(xi1, dth, axis=0)
    xi2 = d2 / np.linalg.norm(d2, axis=1, keepdims=True)
    xi3 = np.cross(xi1, xi2)
    i = 200
    return np.array([xi1[i], xi2[i], xi3[i]]), p[i]


@pytest.mark.parametrize(
    "w_r, t_r, d_r, delta_r, gamma_r",
    [
        (1.5, 2.6, 0.6, 0.0, 0.0),
        (1.8, 1.0, 0.4, 0.3, -0.2),
        (1.6, 0.8, 0.3, -0.25, 0.15),
    ],
)
def test_aperture_surface_consistency(w_r, t_r, d_r, delta_r, gamma_r):
    # The Raup surface (with orientation) and the growing-tube surface (with the
    # converted orientation, aligned start) have coincident aperture circles:
    # same locus, same radius, coplanar.
    e_g, c_g, t_g = raup_to_growing_tube(w_r, t_r, d_r)
    delta_g, gamma_g = raup_aperture_to_growing_tube(w_r, t_r, d_r, delta_r, gamma_r)
    frame0, p0 = _raup_frenet0(w_r, t_r, d_r)
    theta = np.linspace(0, 3 * np.pi, 800)
    phi = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    s = theta * np.log(w_r) / (2 * np.pi * e_g)

    raup(
        w_r,
        t_r,
        d_r,
        delta_r=delta_r,
        gamma_r=gamma_r,
        theta_range=theta,
        phi_range=phi,
    )  # smoke
    ax, frames, rad = _growing_tube_trajectory_ode(e_g, c_g, t_g, 1.0, s, p0, frame0)
    Xg = _assemble_surface(
        ax, frames, rad, orientation=(gamma_g, delta_g), phi_range=phi
    )

    # locus coincides
    vx = 2 * d_r / (1 - d_r) + 1
    vz = 2 * t_r * (d_r / (1 - d_r) + 1)
    sc = w_r ** (theta / (2 * np.pi))
    pR = np.column_stack([sc * vx * np.cos(theta), sc * vx * np.sin(theta), sc * vz])
    sl = slice(80, 720)
    assert np.max(np.linalg.norm(ax - pR, axis=1)[sl]) < 1e-5

    # Raup aperture plane normal, and GTM circle coplanarity + radius
    base = _rotation_z(gamma_r) @ _rotation_x(delta_r) @ np.array([0.0, 1.0, 0.0])
    nrm = np.column_stack(
        [
            np.cos(theta) * base[0] - np.sin(theta) * base[1],
            np.sin(theta) * base[0] + np.cos(theta) * base[1],
            np.full_like(theta, base[2]),
        ]
    )
    rel = Xg - ax[:, None, :]
    plane_dist = np.abs(np.einsum("ijk,ik->ij", rel, nrm)) / rad[:, None]
    radii = np.linalg.norm(rel, axis=2) / rad[:, None]
    assert plane_dist[sl].max() < 1e-6
    np.testing.assert_allclose(radii[sl], 1.0, atol=1e-6)


@pytest.mark.parametrize(
    "w_r, t_r, d_r, delta_r, gamma_r",
    [(1.5, 2.6, 0.6, 0.2, -0.1), (1.8, 1.0, 0.4, -0.3, 0.25)],
)
def test_aperture_round_trip(w_r, t_r, d_r, delta_r, gamma_r):
    delta_g, gamma_g = raup_aperture_to_growing_tube(w_r, t_r, d_r, delta_r, gamma_r)
    d2, g2 = growing_tube_aperture_to_raup(w_r, t_r, d_r, gamma_g, delta_g)
    np.testing.assert_allclose([d2, g2], [delta_r, gamma_r], rtol=1e-4, atol=1e-6)


# --- arc-length consistency between models -----------------------------------


@pytest.mark.parametrize("w_r, t_r, d_r", [(1.5, 2.6, 0.6), (2.0, 0.5, 0.05)])
def test_raup_growing_tube_arc_length_consistency(w_r, t_r, d_r):
    # The arc length l is one geometric quantity: l_r(theta) and l_g(s) agree at
    # the same physical point (theta = D_G * s).
    e_g, c_g, t_g = raup_to_growing_tube(w_r, t_r, d_r)
    d_g = np.hypot(c_g, t_g)
    theta = np.linspace(0.0, 2.0 * np.pi * 3.0, 40)
    s = theta / d_g
    np.testing.assert_allclose(l_r(theta, w_r, t_r, d_r), l_g(s, e_g), rtol=1e-7)
