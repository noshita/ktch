"""Tests for shared generating curve assembly and sampling grids."""

import numpy as np
import pytest

from ktch.coiling._generating_curve import (
    _aperture_plane_basis,
    _assemble_surface,
    whorl_s_range,
    whorl_theta_range,
)


def test_assemble_surface_cylinder():
    # Straight axis along x with identity frames and constant radius -> cylinder.
    n, n_phi, r = 10, 24, 2.0
    axis = np.column_stack([np.linspace(0, 5, n), np.zeros(n), np.zeros(n)])
    frames = np.repeat(np.eye(3)[None, :, :], n, axis=0)
    radius = np.full(n, r)
    phi = np.linspace(0, 2 * np.pi, n_phi)

    X = _assemble_surface(axis, frames, radius, phi_range=phi)

    assert X.shape == (n, n_phi, 3)
    np.testing.assert_allclose(
        X[:, :, 0], np.broadcast_to(axis[:, 0][:, None], X[:, :, 0].shape)
    )
    radial = np.hypot(X[:, :, 1], X[:, :, 2])
    np.testing.assert_allclose(radial, r)


def test_aperture_plane_basis_perpendicular_default():
    # (0, 0) -> circle in the (xi2, xi3) plane: u = xi2, v = xi3.
    u, v = _aperture_plane_basis(0.0, 0.0)
    np.testing.assert_allclose(u, [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(v, [0.0, 0.0, 1.0], atol=1e-12)


def test_assemble_surface_orientation_tilts_out_of_plane():
    # A non-zero orientation introduces a non-zero xi1 (tangent) component.
    n, n_phi = 5, 30
    axis = np.zeros((n, 3))
    frames = np.repeat(np.eye(3)[None, :, :], n, axis=0)
    radius = np.ones(n)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    X_perp = _assemble_surface(
        axis, frames, radius, orientation=(0.0, 0.0), phi_range=phi
    )
    X_tilt = _assemble_surface(
        axis, frames, radius, orientation=(0.3, -0.2), phi_range=phi
    )

    # frames = identity, so the xi1 component is just the x coordinate.
    assert np.allclose(X_perp[:, :, 0], 0.0)
    assert np.abs(X_tilt[:, :, 0]).max() > 0.1


def test_assemble_surface_rejects_aperture():
    axis = np.zeros((3, 3))
    frames = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
    radius = np.ones(3)
    with pytest.raises(NotImplementedError):
        _assemble_surface(axis, frames, radius, aperture=np.ones((4, 3)))


def test_whorl_theta_range_spans_revolutions():
    theta = whorl_theta_range(2.0, n_points_per_whorl=50)
    assert theta[0] == 0.0
    np.testing.assert_allclose(theta[-1], 2 * 2 * np.pi)
    assert len(theta) == 100


def test_whorl_s_range_uses_total_curvature():
    s = whorl_s_range(1.0, 0.3, 0.4, n_points_per_whorl=50)  # D_G = 0.5
    np.testing.assert_allclose(s[-1], 2 * np.pi / 0.5)


def test_whorl_s_range_straight_raises():
    with pytest.raises(ValueError):
        whorl_s_range(1.0, 0.0, 0.0)
