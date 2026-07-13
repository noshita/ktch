"""Tests for the growing tube model."""

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid
from sklearn.utils.validation import check_is_fitted

from ktch.coiling import GrowingTubeModel, growing_tube, l_g, s_g


def test_growing_tube_shape_and_finite():
    s = np.linspace(0, 30, 200)
    phi = np.linspace(0, 2 * np.pi, 40)
    X = growing_tube(0.02, 0.4, 0.06, s_range=s, phi_range=phi)
    assert X.shape == (200, 40, 3)
    assert np.all(np.isfinite(X))


def test_growing_tube_straight_tube():
    s = np.linspace(0, 5, 100)
    phi = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    X = growing_tube(0.0, 0.0, 0.0, r0=1.0, s_range=s, phi_range=phi)
    assert X.shape == (100, 20, 3)
    centroid = X.mean(axis=1)
    np.testing.assert_allclose(centroid[:, 1], 0.0, atol=1e-9)
    np.testing.assert_allclose(centroid[:, 2], 0.0, atol=1e-9)
    assert centroid[-1, 0] > centroid[0, 0]


def test_growing_tube_ode_matches_closed():
    s = np.linspace(0, 25, 300)
    phi = np.linspace(0, 2 * np.pi, 30)
    X_ode = growing_tube(0.05, 0.4, 0.1, s_range=s, phi_range=phi, method="ode")
    X_closed = growing_tube(0.05, 0.4, 0.1, s_range=s, phi_range=phi, method="closed")
    np.testing.assert_allclose(X_ode, X_closed, atol=1e-4, rtol=1e-4)


def test_growing_tube_orientation_changes_surface():
    s = np.linspace(0, 25, 200)
    phi = np.linspace(0, 2 * np.pi, 30)
    X0 = growing_tube(0.05, 0.4, 0.1, s_range=s, phi_range=phi)
    X1 = growing_tube(
        0.05, 0.4, 0.1, delta_g=0.2, gamma_g=0.1, s_range=s, phi_range=phi
    )
    assert np.all(np.isfinite(X1))
    assert not np.allclose(X0, X1)


def test_growing_tube_radius_law():
    s = np.linspace(0, 10, 100)
    phi = np.linspace(0, 2 * np.pi, 40)
    e_g, r0 = 0.1, 1.0
    X = growing_tube(e_g, 0.3, 0.05, r0=r0, s_range=s, phi_range=phi)
    centroid = X.mean(axis=1, keepdims=True)
    radial = np.linalg.norm(X - centroid, axis=2).mean(axis=1)
    np.testing.assert_allclose(radial, r0 * np.exp(e_g * s), rtol=1e-2)


@pytest.mark.parametrize(
    "c_g, r0, method",
    [(-0.1, 1.0, "ode"), (0.3, 0.0, "ode"), (0.3, 1.0, "bogus")],
)
def test_growing_tube_invalid_params(c_g, r0, method):
    with pytest.raises(ValueError):
        growing_tube(0.02, c_g, 0.06, r0=r0, method=method)


def test_growing_tube_output_reserved():
    with pytest.raises(NotImplementedError):
        growing_tube(0.02, 0.4, 0.06, output="landmarks")


def test_growing_tube_model_batch_and_feature_names():
    model = GrowingTubeModel()
    s = np.linspace(0, 20, 50)
    phi = np.linspace(0, 2 * np.pi, 20)
    params = np.array([[0.02, 0.4, 0.06, 0.1, 0.0], [0.05, 0.2, 0.3, 0.0, 0.2]])
    X = model.inverse_transform(params, s_range=s, phi_range=phi)
    assert X.shape == (2, 50, 20, 3)
    assert list(model.get_feature_names_out()) == [
        "e_g",
        "c_g",
        "t_g",
        "delta_g",
        "gamma_g",
    ]


def test_growing_tube_model_accepts_locus_only():
    model = GrowingTubeModel()
    s = np.linspace(0, 20, 40)
    phi = np.linspace(0, 2 * np.pi, 16)
    X5 = model.inverse_transform(
        np.array([[0.02, 0.4, 0.06, 0.0, 0.0]]), s_range=s, phi_range=phi
    )
    X3 = model.inverse_transform(
        np.array([[0.02, 0.4, 0.06]]), s_range=s, phi_range=phi
    )
    np.testing.assert_allclose(X5, X3)


def test_growing_tube_model_is_fitted():
    model = GrowingTubeModel()
    assert model.__sklearn_is_fitted__() is True
    check_is_fitted(model)


def test_growing_tube_model_nls3d_round_trip():
    # Synthesize the centroid locus + thickness from known params, then recover.
    e_true, c_true, t_true = 0.05, 0.4, 0.06
    s = np.linspace(0, 20, 200)
    phi = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    surf = growing_tube(
        e_true, c_true, t_true, s_range=s, phi_range=phi, method="closed"
    )
    centroids = surf.mean(axis=1)  # circle centroid == trajectory point
    r = np.exp(e_true * s)  # r0 = 1 default
    X = [np.column_stack([centroids, r])]
    out = GrowingTubeModel().transform(X)
    assert out.shape == (1, 5)
    np.testing.assert_allclose(
        out[0, :3], [e_true, c_true, t_true], rtol=2e-2, atol=1e-3
    )
    np.testing.assert_array_equal(out[0, 3:], [0.0, 0.0])


def test_growing_tube_model_nls3d_uses_domain_coords_arc_length():
    # Supplying the exact arc length via domain_coords removes the tentative
    # chord-length error, so recovery is tighter than the chord-length default.
    e_true, c_true, t_true = 0.05, 0.4, 0.06
    s = np.linspace(0, 20, 200)
    phi = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    surf = growing_tube(
        e_true, c_true, t_true, s_range=s, phi_range=phi, method="closed"
    )
    centroids = surf.mean(axis=1)
    r = np.exp(e_true * s)
    l_exact = np.expm1(e_true * s) / e_true  # r0 = 1
    X = [np.column_stack([centroids, r])]
    out = GrowingTubeModel().transform(X, domain_coords=[l_exact.reshape(-1, 1)])
    np.testing.assert_allclose(
        out[0, :3], [e_true, c_true, t_true], rtol=1e-3, atol=1e-4
    )


def test_growing_tube_model_transform_validates_channels():
    # 3 columns (locus only, no thickness) fails the required 4-channel contract.
    with pytest.raises(ValueError, match="n_channels"):
        GrowingTubeModel().transform([np.zeros((10, 3))])


def test_growing_tube_model_transform_rejects_aperture():
    with pytest.raises(NotImplementedError, match="aperture"):
        GrowingTubeModel().transform([np.zeros((10, 4))], aperture=object())


def test_growing_tube_model_fit_transform_routes_domain_coords():
    # A length-mismatched domain_coords surfaces only if fit_transform forwards
    # it to transform's _check_panel.
    with pytest.raises(ValueError, match="same length"):
        GrowingTubeModel().fit_transform(
            [np.zeros((10, 4)), np.zeros((8, 4))],
            domain_coords=[np.zeros(10)],
        )


# --- surface estimator (structured-surface fit) -----------------------------


def test_growing_tube_surface_round_trip_orientation():
    # transform(inverse_transform(params)) recovers all five parameters,
    # including the aperture orientation a centerline fit cannot see.
    model = GrowingTubeModel(estimator="surface", method="closed")
    params = np.array(
        [[0.30, 1.20, 0.25, 0.40, 0.30], [0.05, 0.80, -0.30, -0.20, 0.10]]
    )
    s = np.linspace(0, 4.5, 200)
    surf = model.inverse_transform(params, s_range=s)
    assert surf.shape[0] == 2 and surf.shape[-1] == 3
    est = model.transform(surf)
    assert est.shape == (2, 5)
    np.testing.assert_allclose(est, params, atol=1e-6)


def test_growing_tube_surface_round_trip_ode():
    # An ODE-generated surface still round-trips; the ode/closed gap sets the tol.
    model = GrowingTubeModel(estimator="surface")  # method="ode"
    params = np.array([[0.05, 0.6, 0.2, 0.1, -0.15]])
    s = np.linspace(0, 20, 200)
    est = model.transform(model.inverse_transform(params, s_range=s))
    np.testing.assert_allclose(est, params, atol=1e-4)


def test_growing_tube_surface_recovers_arbitrary_scale():
    # r0 is a free nuisance, so a surface at a non-unit scale round-trips.
    model = GrowingTubeModel(estimator="surface", method="closed", r0=2.0)
    params = np.array([[0.2, 1.0, 0.3, 0.15, -0.2]])
    s = np.linspace(0, 4.5, 200)
    est = model.transform(model.inverse_transform(params, s_range=s))
    np.testing.assert_allclose(est, params, atol=1e-6)


def test_growing_tube_surface_recovers_under_rigid_pose():
    # The estimator fits its own pose, so an arbitrarily rotated/translated
    # surface yields the same (pose-invariant) shape parameters.
    from scipy.spatial.transform import Rotation

    model = GrowingTubeModel(estimator="surface", method="closed")
    params = np.array([0.20, 1.0, 0.3, 0.25, -0.2])
    s = np.linspace(0, 4.5, 200)
    surf = model.inverse_transform(params, s_range=s)  # canonical pose
    R = Rotation.from_euler("xyz", [0.5, -0.8, 1.2]).as_matrix()
    posed = surf @ R.T + np.array([3.0, -2.0, 5.0])
    est = model.transform(posed)
    np.testing.assert_allclose(est[0], params, atol=1e-6)


@pytest.mark.parametrize("seed", range(6))
def test_growing_tube_surface_battery(seed):
    rng = np.random.default_rng(seed)
    model = GrowingTubeModel(estimator="surface", method="closed")
    params = np.array(
        [
            rng.uniform(0.05, 0.5),
            rng.uniform(0.4, 1.8),
            rng.uniform(-0.8, 0.8),
            rng.uniform(-0.4, 0.4),
            rng.uniform(-0.4, 0.4),
        ]
    )
    s = np.linspace(0, rng.uniform(3.5, 5.5), 180)
    est = model.transform(model.inverse_transform(params[None, :], s_range=s))
    np.testing.assert_allclose(est[0], params, atol=1e-5)


def test_growing_tube_surface_input_encodings():
    # 4D ndarray, list of 3D, single 3D, and the long DataFrame all agree.
    model = GrowingTubeModel(estimator="surface", method="closed")
    params = np.array([[0.2, 1.0, 0.3, 0.1, 0.2]])
    s = np.linspace(0, 4.0, 150)
    surf4d = model.inverse_transform(params, s_range=s)
    ref = model.transform(surf4d)
    np.testing.assert_allclose(model.transform([surf4d[0]]), ref, atol=1e-9)
    np.testing.assert_allclose(model.transform(surf4d[0]), ref, atol=1e-9)
    df = model.inverse_transform(params, s_range=s, as_frame=True)
    np.testing.assert_allclose(model.transform(df), ref, atol=1e-9)


def test_growing_tube_surface_empty_panel():
    assert GrowingTubeModel(estimator="surface").transform([]).shape == (0, 5)


def test_growing_tube_invalid_estimator():
    with pytest.raises(ValueError, match="estimator"):
        GrowingTubeModel(estimator="bogus").transform(np.zeros((1, 5, 5, 3)))


def test_growing_tube_surface_rejects_aperture():
    with pytest.raises(NotImplementedError, match="aperture"):
        GrowingTubeModel(estimator="surface").transform(
            np.zeros((1, 5, 5, 3)), aperture=object()
        )


def test_growing_tube_surface_rejects_non_finite():
    # A partial aperture (missing points as NaN) is not supported yet; the
    # estimator fails fast rather than propagating NaN into the fit.
    surf = np.zeros((1, 5, 5, 3))
    surf[0, 2, 1, :] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        GrowingTubeModel(estimator="surface").transform(surf)


def test_growing_tube_surface_nonconvergence_warns():
    # A heavily perturbed surface cannot be fit well; the estimator warns.
    model = GrowingTubeModel(estimator="surface", method="closed")
    params = np.array([[0.2, 1.0, 0.3, 0.0, 0.0]])
    s = np.linspace(0, 4.0, 120)
    surf = model.inverse_transform(params, s_range=s)[0]
    rng = np.random.default_rng(0)
    surf = surf + rng.normal(scale=0.5, size=surf.shape)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        model.transform([surf])


# --- heteromorph (varying parameters) ---------------------------------------


def test_growing_tube_constant_callable_matches_scalar():
    s = np.linspace(0, 25, 300)
    phi = np.linspace(0, 2 * np.pi, 30)
    X_scalar = growing_tube(0.05, 0.4, 0.1, s_range=s, phi_range=phi)
    X_callable = growing_tube(
        lambda u: 0.05, lambda u: 0.4, lambda u: 0.1, s_range=s, phi_range=phi
    )
    X_array = growing_tube(
        np.full_like(s, 0.05),
        np.full_like(s, 0.4),
        np.full_like(s, 0.1),
        s_range=s,
        phi_range=phi,
    )
    np.testing.assert_allclose(X_callable, X_scalar, atol=1e-7)
    np.testing.assert_allclose(X_array, X_scalar, atol=1e-7)


def test_growing_tube_heteromorph_shape_and_finite():
    s = np.linspace(0, 50, 400)
    phi = np.linspace(0, 2 * np.pi, 30)
    X = growing_tube(0.02, lambda u: 0.1 + 0.01 * u, 0.05, s_range=s, phi_range=phi)
    assert X.shape == (400, 30, 3)
    assert np.all(np.isfinite(X))


def test_growing_tube_heteromorph_radius_integrates_expansion():
    # Varying e_g(s): tube radius follows r0 * exp(integral_0^s e_g).
    s = np.linspace(0, 20, 400)
    phi = np.linspace(0, 2 * np.pi, 40)
    r0 = 1.0

    def e_g(u):
        return 0.02 + 0.004 * u

    X = growing_tube(e_g, 0.3, 0.05, r0=r0, s_range=s, phi_range=phi)
    centroid = X.mean(axis=1, keepdims=True)
    radial = np.linalg.norm(X - centroid, axis=2).mean(axis=1)
    expected = r0 * np.exp(cumulative_trapezoid(e_g(s), s, initial=0.0))
    np.testing.assert_allclose(radial, expected, rtol=1e-2)


def test_growing_tube_varying_requires_ode():
    s = np.linspace(0, 20, 200)
    with pytest.raises(ValueError):
        growing_tube(0.02, lambda u: 0.1 + 0.01 * u, 0.05, s_range=s, method="closed")


def test_growing_tube_varying_requires_s_range():
    with pytest.raises(ValueError):
        growing_tube(0.02, lambda u: 0.1 + 0.01 * u, 0.05)


# --- arc-length conversion (l_g, s_g) ---------------------------------------


@pytest.mark.parametrize("e_g", [0.02, 0.1, -0.05])
def test_l_g_matches_numerical_arc_length(e_g):
    s = np.linspace(0.0, 80.0, 200001)
    r = np.exp(e_g * s)  # r0 = 1
    l_num = cumulative_trapezoid(r, s, initial=0.0)
    assert np.max(np.abs(l_g(s, e_g) - l_num)) / l_num[-1] < 1e-6


@pytest.mark.parametrize("e_g", [0.02, 0.1, -0.05])
def test_growing_tube_s_l_roundtrip(e_g):
    s = np.linspace(0.0, 80.0, 50)
    np.testing.assert_allclose(s_g(l_g(s, e_g), e_g), s, atol=1e-9)


def test_l_g_e_g_zero_limit():
    s = np.linspace(0.0, 10.0, 20)
    np.testing.assert_allclose(l_g(s, 0.0, r0=2.0), 2.0 * s)
    np.testing.assert_allclose(s_g(2.0 * s, 0.0, r0=2.0), s)
    np.testing.assert_allclose(l_g(s, 1e-9), l_g(s, 0.0), atol=1e-6)


def test_l_g_r0_scaling():
    s = np.linspace(0.1, 20.0, 30)
    np.testing.assert_allclose(l_g(s, 0.05, r0=3.0), 3.0 * l_g(s, 0.05))


def test_l_g_scalar_and_array():
    assert isinstance(l_g(1.0, 0.05), float)
    assert isinstance(s_g(0.5, 0.05), float)
    x = np.linspace(0.1, 1.0, 7)
    assert l_g(x, 0.05).shape == x.shape


@pytest.mark.parametrize("bad_r0", [0.0, -1.0])
def test_l_g_invalid_r0(bad_r0):
    with pytest.raises(ValueError):
        l_g(1.0, 0.05, r0=bad_r0)
