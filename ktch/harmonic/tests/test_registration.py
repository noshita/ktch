"""Tests for shared harmonic registration utilities."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ktch.harmonic._registration import (
    moment_frame,
    moment_register,
    validate_registration,
)


def _random_rotation(n_dim, rng):
    """Random proper rotation (det = +1)."""
    A = rng.standard_normal((n_dim, n_dim))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


class TestMomentFrame:
    """Tests for the moment_frame principal-axis estimator."""

    def test_orthonormal_and_proper(self):
        rng = np.random.default_rng(1)
        vectors = rng.standard_normal((60, 3))
        Q, size, w = moment_frame(vectors)

        assert_allclose(Q.T @ Q, np.eye(3), atol=1e-10)
        assert np.linalg.det(Q) == pytest.approx(1.0, abs=1e-10)
        # descending eigenvalues
        assert np.all(np.diff(w) <= 1e-9)
        # size = sqrt(trace(M)) = sqrt(sum of squared coefficients)
        assert size == pytest.approx(np.sqrt(np.sum(vectors**2)), rel=1e-10)

    def test_reflect_allows_improper(self):
        # This seed yields a det=-1 (improper) frame, exercising the branch
        # where reflect=True suppresses the proper-rotation flip.
        rng = np.random.default_rng(2)
        vectors = rng.standard_normal((60, 3))
        Q, _, _ = moment_frame(vectors, reflect=True)
        assert_allclose(Q.T @ Q, np.eye(3), atol=1e-10)
        assert np.linalg.det(Q) == pytest.approx(-1.0, abs=1e-10)

    def test_reflect_invariance_under_mirror(self):
        # reflect=True maps a shape and its mirror image to the same registered
        # coefficients; reflect=False (chirality-preserving) does not.
        rng = np.random.default_rng(2)
        mat = rng.standard_normal((3, 8))  # (n_dim=3, n_modes); col 0 = constant
        mirror = np.diag([-1.0, 1.0, 1.0])
        mat_m = mirror @ mat

        reg = moment_register(mat.ravel(), 3, scale=True, reflect=True)
        reg_m = moment_register(mat_m.ravel(), 3, scale=True, reflect=True)
        assert_allclose(reg, reg_m, atol=1e-9)

        reg_nf = moment_register(mat.ravel(), 3, scale=True, reflect=False)
        reg_m_nf = moment_register(mat_m.ravel(), 3, scale=True, reflect=False)
        assert np.max(np.abs(reg_nf - reg_m_nf)) > 1e-3


class TestMomentRegisterInvariance:
    """Tests that moment_register is invariant to similarity transforms."""

    @pytest.mark.parametrize("n_dim", [2, 3, 4])
    def test_translation_rotation_scale_invariance(self, n_dim):
        rng = np.random.default_rng(10 + n_dim)
        n_modes = 16
        mat = rng.standard_normal((n_dim, n_modes))
        flat1 = mat.ravel()

        R = _random_rotation(n_dim, rng)
        s = 2.5
        t = rng.standard_normal(n_dim)
        mat2 = s * (R @ mat)
        mat2[:, 0] = mat2[:, 0] + t  # translation enters the constant mode
        flat2 = mat2.ravel()

        r1 = moment_register(flat1, n_dim, scale=True, reflect=False)
        r2 = moment_register(flat2, n_dim, scale=True, reflect=False)
        assert_allclose(r1, r2, atol=1e-8)

    def test_translation_removed(self):
        n_dim, n_modes = 3, 9
        rng = np.random.default_rng(7)
        mat = rng.standard_normal((n_dim, n_modes))
        out = moment_register(mat.ravel(), n_dim, scale=True)
        # constant mode (column 0 of the reshaped output) is zeroed
        assert_allclose(out.reshape(n_dim, n_modes)[:, 0], 0.0, atol=1e-12)

    def test_scale_false_keeps_size(self):
        n_dim, n_modes = 3, 9
        rng = np.random.default_rng(8)
        mat = rng.standard_normal((n_dim, n_modes))
        flat = mat.ravel()
        with_scale = moment_register(flat, n_dim, scale=True)
        without = moment_register(flat, n_dim, scale=False)
        # form space (scale=False) has larger norm than shape space
        assert np.linalg.norm(without) > np.linalg.norm(with_scale)

    def test_degenerate_size_raises_when_scaling(self):
        n_dim, n_modes = 3, 9
        flat = np.zeros(n_dim * n_modes)
        with pytest.raises(ValueError, match="near-zero centroid size"):
            moment_register(flat, n_dim, scale=True)


class TestValidateRegistration:
    """Tests for the validate_registration argument checks."""

    _SMBR = {
        "first_order": {None, "semi_major_axis", "ellipse_area"},
        "moment": {None, "centroid_size"},
    }

    def test_none_ok(self):
        validate_registration(
            None,
            None,
            self._SMBR,
            n_dim=3,
            return_transform=False,
            allow_first_order=False,
        )

    def test_moment_2d_3d_ok(self):
        for n_dim in (2, 3):
            validate_registration(
                "moment",
                None,
                self._SMBR,
                n_dim=n_dim,
                return_transform=False,
                allow_first_order=False,
            )

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="registration must be one of"):
            validate_registration(
                "bogus",
                None,
                self._SMBR,
                n_dim=3,
                return_transform=False,
                allow_first_order=True,
            )

    def test_reserved_raises(self):
        with pytest.raises(NotImplementedError, match="reserved"):
            validate_registration(
                "landmark",
                None,
                self._SMBR,
                n_dim=3,
                return_transform=False,
                allow_first_order=True,
            )

    def test_first_order_not_allowed_raises(self):
        with pytest.raises(NotImplementedError, match="first_order"):
            validate_registration(
                "first_order",
                None,
                self._SMBR,
                n_dim=3,
                return_transform=False,
                allow_first_order=False,
            )

    @pytest.mark.parametrize("n_dim", [1, 4])
    def test_registration_requires_2d_3d(self, n_dim):
        with pytest.raises(ValueError, match="2D/3D"):
            validate_registration(
                "moment",
                None,
                self._SMBR,
                n_dim=n_dim,
                return_transform=False,
                allow_first_order=False,
            )

    def test_invalid_scale_method_raises(self):
        with pytest.raises(ValueError, match="not valid for"):
            validate_registration(
                "moment",
                "semi_major_axis",
                self._SMBR,
                n_dim=3,
                return_transform=False,
                allow_first_order=False,
            )

    def test_return_transform_with_none_raises(self):
        with pytest.raises(ValueError, match="return_transform requires"):
            validate_registration(
                None,
                None,
                self._SMBR,
                n_dim=3,
                return_transform=True,
                allow_first_order=False,
            )

    def test_return_transform_not_implemented(self):
        with pytest.raises(NotImplementedError, match="return_transform"):
            validate_registration(
                "moment",
                None,
                self._SMBR,
                n_dim=3,
                return_transform=True,
                allow_first_order=False,
            )
