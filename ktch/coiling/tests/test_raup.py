"""Tests for Raup's model."""

import numpy as np
import pandas as pd
import pytest

from ktch.coiling import RaupModel, raup


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
    [(1.0, 1.0, 0.5), (0.9, 1.0, 0.5), (1.5, 1.0, 1.0), (1.5, 1.0, -0.1)],
)
def test_raup_invalid_params(w_r, t_r, d_r):
    with pytest.raises(ValueError):
        raup(w_r, t_r, d_r)


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
