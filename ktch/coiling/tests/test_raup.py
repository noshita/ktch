"""Tests for Raup's model."""

import numpy as np
import pandas as pd
import pytest

from ktch.coiling import RaupModel, l_r, raup, theta_r


def test_raup_shape_and_finite():
    theta = np.linspace(0, 2 * np.pi * 2, 200)
    phi = np.linspace(0, 2 * np.pi, 40)
    X = raup(1.5, 2.6, 0.6, theta_range=theta, phi_range=phi)
    assert X.shape == (200, 40, 3)
    assert np.all(np.isfinite(X))


def test_raup_orientation_changes_surface():
    theta = np.linspace(0, 2 * np.pi * 2, 200)
    phi = np.linspace(0, 2 * np.pi, 40)
    X_radial = raup(1.5, 2.6, 0.6, theta_range=theta, phi_range=phi)
    X_tilt = raup(
        1.5, 2.6, 0.6, delta_r=0.3, gamma_r=-0.2, theta_range=theta, phi_range=phi
    )
    assert np.all(np.isfinite(X_tilt))
    assert not np.allclose(X_radial, X_tilt)


@pytest.mark.parametrize(
    "w_r, t_r, d_r",
    [(1.0, 1.0, 0.5), (0.9, 1.0, 0.5), (1.5, 1.0, 1.0), (1.5, 1.0, -1.0)],
)
def test_raup_invalid_params(w_r, t_r, d_r):
    with pytest.raises(ValueError):
        raup(w_r, t_r, d_r)


@pytest.mark.parametrize("d_r", [-0.3, -0.9])
def test_raup_negative_d_in_range(d_r):
    theta = np.linspace(0, 2 * np.pi * 2, 100)
    phi = np.linspace(0, 2 * np.pi, 20)
    X = raup(1.5, 1.0, d_r, theta_range=theta, phi_range=phi)
    assert np.all(np.isfinite(X))


def test_raup_output_reserved():
    with pytest.raises(NotImplementedError):
        raup(1.5, 2.6, 0.6, output="centerline")


def test_raup_model_inverse_transform_full_and_locus_only():
    model = RaupModel()
    theta = np.linspace(0, 2 * np.pi, 50)
    phi = np.linspace(0, 2 * np.pi, 20)
    # 5-column (full) and 3-column (locus only, orientation -> 0) agree when
    # the orientation columns are zero.
    full = np.array([[1.5, 2.6, 0.6, 0.0, 0.0]])
    locus = np.array([[1.5, 2.6, 0.6]])
    X5 = model.inverse_transform(full, theta_range=theta, phi_range=phi)
    X3 = model.inverse_transform(locus, theta_range=theta, phi_range=phi)
    assert X5.shape == (1, 50, 20, 3)
    np.testing.assert_allclose(X5, X3)


def test_raup_model_inverse_transform_batch_and_single():
    model = RaupModel()
    theta = np.linspace(0, 2 * np.pi, 40)
    phi = np.linspace(0, 2 * np.pi, 16)
    params = np.array([[1.5, 2.6, 0.6, 0.1, 0.0], [1.7, 2.0, 0.4, 0.0, 0.2]])
    X = model.inverse_transform(params, theta_range=theta, phi_range=phi)
    assert X.shape == (2, 40, 16, 3)
    Xs = model.inverse_transform(
        np.array([1.5, 2.6, 0.6, 0.1, 0.0]), theta_range=theta, phi_range=phi
    )
    assert Xs.shape == (40, 16, 3)


def test_raup_model_inverse_transform_as_frame():
    model = RaupModel()
    theta = np.linspace(0, 2 * np.pi, 10)
    phi = np.linspace(0, 2 * np.pi, 6)
    df = model.inverse_transform(
        np.array([[1.5, 2.6, 0.6, 0.0, 0.0]]),
        theta_range=theta,
        phi_range=phi,
        as_frame=True,
    )
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["x", "y", "z"]
    assert df.index.names == ["specimen_id", "trajectory_id", "phi_id"]
    assert len(df) == 10 * 6


def test_raup_model_feature_names():
    model = RaupModel()
    assert list(model.get_feature_names_out()) == [
        "w_r",
        "t_r",
        "d_r",
        "delta_r",
        "gamma_r",
    ]


def test_raup_model_fit_returns_self():
    model = RaupModel()
    assert model.fit(None) is model


def test_raup_model_transform_not_implemented():
    with pytest.raises(NotImplementedError):
        RaupModel().transform([np.zeros((10, 2))])


# --- arc-length conversion (l_r, theta_r) -----------------------------------

ARC_CASES = [(1.5, 2.6, 0.6), (1.3, 1.5, 0.2), (2.0, 0.5, 0.05), (1.1, 0.0, 0.4)]


def _raup_reference_trajectory(theta, w_r, t_r, d_r, r0=1.0):
    """Reference-point trajectory (circle centre), matching _raup_surface."""
    vx = (1.0 + d_r) / (1.0 - d_r)
    vz = 2.0 * t_r / (1.0 - d_r)
    scale = r0 * w_r ** (theta / (2.0 * np.pi))
    return np.column_stack(
        [scale * vx * np.cos(theta), scale * vx * np.sin(theta), scale * vz]
    )


@pytest.mark.parametrize("w_r, t_r, d_r", ARC_CASES)
def test_l_r_matches_numerical_arc_length(w_r, t_r, d_r):
    theta = np.linspace(0.0, 2.0 * np.pi * 4.0, 400001)
    p = _raup_reference_trajectory(theta, w_r, t_r, d_r)
    seg = np.sqrt(np.sum(np.diff(p, axis=0) ** 2, axis=1))
    l_num = np.concatenate([[0.0], np.cumsum(seg)])
    assert np.max(np.abs(l_r(theta, w_r, t_r, d_r) - l_num)) / l_num[-1] < 1e-6


@pytest.mark.parametrize("w_r, t_r, d_r", ARC_CASES)
def test_raup_theta_l_roundtrip(w_r, t_r, d_r):
    theta = np.linspace(0.0, 2.0 * np.pi * 4.0, 50)
    np.testing.assert_allclose(
        theta_r(l_r(theta, w_r, t_r, d_r), w_r, t_r, d_r), theta, atol=1e-9
    )


def test_l_r_r0_scaling():
    theta = np.linspace(0.1, 2.0 * np.pi * 2.0, 30)
    np.testing.assert_allclose(
        l_r(theta, 1.4, 1.0, 0.2, r0=3.0), 3.0 * l_r(theta, 1.4, 1.0, 0.2)
    )


def test_l_r_scalar_and_array():
    assert isinstance(l_r(1.0, 1.4, 1.0, 0.2), float)
    assert isinstance(theta_r(0.5, 1.4, 1.0, 0.2), float)
    x = np.linspace(0.1, 1.0, 7)
    assert l_r(x, 1.4, 1.0, 0.2).shape == x.shape


@pytest.mark.parametrize("bad_r0", [0.0, -1.0])
def test_l_r_invalid_r0(bad_r0):
    with pytest.raises(ValueError):
        l_r(1.0, 1.4, 1.0, 0.2, r0=bad_r0)
