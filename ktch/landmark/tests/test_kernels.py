"""Tests for kernel functions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

from ktch.landmark._kernels import (
    tps_bending_energy,
    tps_bending_energy_matrix,
    tps_coefficients,
    tps_kernel,
    tps_kernel_matrix,
    tps_system_matrix,
    tps_warp,
)

###########################################################
#
#   tps_kernel
#
###########################################################


def test_tps_kernel_2d_zero():
    """Test that TPS kernel is 0 at r=0 for 2D."""
    assert tps_kernel(0.0, n_dim=2) == 0.0


def test_tps_kernel_2d_one():
    """Test TPS kernel at r=1 for 2D (log(1)=0, so U(1)=0)."""
    assert tps_kernel(1.0, n_dim=2) == 0.0


def test_tps_kernel_2d_values():
    """Test TPS kernel values for 2D."""
    r = np.array([0.5, 2.0])
    expected = r**2 * np.log(r)
    result = tps_kernel(r, n_dim=2)
    assert_array_almost_equal(result, expected)


def test_tps_kernel_3d():
    """Test TPS kernel for 3D (U(r) = -r)."""
    r = np.array([0.0, 1.0, 2.0])
    expected = np.array([0.0, -1.0, -2.0])
    result = tps_kernel(r, n_dim=3)
    assert_array_almost_equal(result, expected)


def test_tps_kernel_invalid_dim():
    """Test that invalid n_dim raises error."""
    with pytest.raises(ValueError, match="n_dim must be 2 or 3"):
        tps_kernel(1.0, n_dim=4)


###########################################################
#
#   tps_kernel_matrix
#
###########################################################


def test_tps_kernel_matrix_shape():
    """Test kernel matrix shape."""
    X = np.random.randn(5, 2)
    K = tps_kernel_matrix(X)
    assert K.shape == (5, 5)


def test_tps_kernel_matrix_symmetric():
    """Test that kernel matrix is symmetric."""
    X = np.random.randn(5, 2)
    K = tps_kernel_matrix(X)
    assert_array_almost_equal(K, K.T)


def test_tps_kernel_matrix_diagonal():
    """Test that kernel matrix diagonal is zero."""
    X = np.random.randn(5, 2)
    K = tps_kernel_matrix(X)
    assert_array_almost_equal(np.diag(K), np.zeros(5))


###########################################################
#
#   tps_system_matrix
#
###########################################################


def test_tps_system_matrix_shape_2d():
    """Test system matrix shape for 2D."""
    X = np.random.randn(5, 2)
    L = tps_system_matrix(X)
    # Shape should be (n + n_dim + 1, n + n_dim + 1) = (5 + 2 + 1, 5 + 2 + 1)
    assert L.shape == (8, 8)


def test_tps_system_matrix_shape_3d():
    """Test system matrix shape for 3D."""
    X = np.random.randn(5, 3)
    L = tps_system_matrix(X)
    # Shape should be (5 + 3 + 1, 5 + 3 + 1) = (9, 9)
    assert L.shape == (9, 9)


def test_tps_system_matrix_symmetric():
    """Test that system matrix is symmetric."""
    X = np.random.randn(5, 2)
    L = tps_system_matrix(X)
    assert_array_almost_equal(L, L.T)


###########################################################
#
#   tps_coefficients
#
###########################################################


def test_tps_coefficients_shapes():
    """Test coefficient shapes."""
    source = np.random.randn(5, 2)
    target = np.random.randn(5, 2)
    W, c, A = tps_coefficients(source, target)
    assert W.shape == (5, 2)
    assert c.shape == (2,)
    assert A.shape == (2, 2)


def test_tps_coefficients_identity():
    """Test that identity transformation gives zero non-affine coeffs."""
    source = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    W, c, A = tps_coefficients(source, source)
    # W should be all zeros for identity
    assert_array_almost_equal(W, np.zeros_like(W))
    # c should be zero
    assert_array_almost_equal(c, np.zeros(2))
    # A should be identity
    assert_array_almost_equal(A, np.eye(2))


def test_tps_coefficients_translation():
    """Test pure translation."""
    source = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    target = source + np.array([5, 3])
    W, c, A = tps_coefficients(source, target)
    # W should be zero for pure affine
    assert_array_almost_equal(W, np.zeros_like(W), decimal=10)
    # c should be the translation
    assert_array_almost_equal(c, np.array([5, 3]))


def test_tps_coefficients_shape_mismatch():
    """Test that shape mismatch raises error."""
    source = np.random.randn(5, 2)
    target = np.random.randn(4, 2)
    with pytest.raises(ValueError, match="same shape"):
        tps_coefficients(source, target)


###########################################################
#
#   tps_warp
#
###########################################################


def test_tps_warp_interpolation():
    """Test that TPS warp is consistent with the legacy implementation."""
    from ktch.landmark._kriging import _thin_plate_spline_2d, _tps_2d

    source = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    target = np.array([[0.1, 0.1], [1.2, 0], [0, 1.1], [1, 1]], dtype=float)

    # Get coefficients from both implementations
    W_old, c_old, A_old = _thin_plate_spline_2d(source, target)
    W_new, c_new, A_new = tps_coefficients(source, target)

    # Verify coefficients are the same
    assert_allclose(W_new, W_old)
    assert_allclose(c_new, c_old)
    assert_allclose(A_new, A_old)

    # Verify warp results are the same
    for i, (x, y) in enumerate(source):
        old_result = _tps_2d(x, y, source, W_old, c_old, A_old)
        new_result = tps_warp(np.array([[x, y]]), source, W_new, c_new, A_new)[0]
        assert_allclose(new_result, old_result)


def test_tps_warp_shape():
    """Test warped output shape."""
    source = np.random.randn(4, 2)
    target = np.random.randn(4, 2)
    W, c, A = tps_coefficients(source, target)
    points = np.random.randn(10, 2)
    warped = tps_warp(points, source, W, c, A)
    assert warped.shape == points.shape


###########################################################
#
#   tps_bending_energy
#
###########################################################


def test_tps_bending_energy_identity():
    """Test that identity transformation has zero bending energy."""
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    be = tps_bending_energy(X, X)
    assert_allclose(be, 0.0, atol=1e-10)


def test_tps_bending_energy_translation():
    """Test that pure translation has zero bending energy."""
    source = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    target = source + np.array([5, 3])
    be = tps_bending_energy(source, target)
    assert_allclose(be, 0.0, atol=1e-10)


def test_tps_bending_energy_affine():
    """Test that pure affine transformation has zero bending energy."""
    source = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    # Apply rotation and scaling
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    target = (source @ R.T) * 2 + np.array([1, 2])
    be = tps_bending_energy(source, target)
    assert_allclose(be, 0.0, atol=1e-10)


def test_tps_bending_energy_non_affine():
    """Test that non-affine transformation has positive bending energy."""
    source = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    # Apply non-affine deformation
    target = source.copy()
    target[0] += np.array([0.1, 0.1])  # Move one point
    be = tps_bending_energy(source, target)
    assert be > 0


def test_tps_bending_energy_nonnegative():
    """Test that bending energy is always non-negative."""
    np.random.seed(42)
    for _ in range(10):
        source = np.random.randn(5, 2)
        target = np.random.randn(5, 2)
        be = tps_bending_energy(source, target)
        assert be >= -1e-10  # Allow small numerical errors


###########################################################
#
#   tps_bending_energy_matrix
#
###########################################################


def test_tps_bending_energy_matrix_shape_2d():
    """Test bending energy matrix shape for 2D."""
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    Lk_inv = tps_bending_energy_matrix(X)
    assert Lk_inv.shape == (4, 4)


def test_tps_bending_energy_matrix_shape_3d():
    """Test bending energy matrix shape for 3D."""
    X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    Lk_inv = tps_bending_energy_matrix(X)
    assert Lk_inv.shape == (4, 4)


def test_tps_bending_energy_matrix_symmetric():
    """Test that bending energy matrix is symmetric."""
    X = np.array(
        [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], dtype=float
    )
    Lk_inv = tps_bending_energy_matrix(X)
    assert_allclose(Lk_inv, Lk_inv.T, atol=1e-10)


def test_tps_bending_energy_matrix_equivalence():
    """Test that H^T L_k^{-1} H equals trace(W^T K W) for bending energy."""
    source = np.array(
        [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], dtype=float
    )
    target = source.copy()
    target[2] += np.array([0.1, -0.1])

    # Method 1: trace(W^T K W) (existing function)
    be1 = tps_bending_energy(source, target)

    # Method 2: sum over dims of H_d^T L_k^{-1} H_d
    Lk_inv = tps_bending_energy_matrix(source)
    W, c, A = tps_coefficients(source, target)
    K = tps_kernel_matrix(source)
    # H = K @ W (the target heights projected through kernel)
    # But actually BE = Tr(W^T K W) = sum_d (K @ W[:,d])^T L_k^{-1} ... no.
    # The equivalence is: Tr(W^T K W) = sum_d H_d^T L_k^{-1} H_d
    # where H_d = target[:,d] (the target coordinate values)
    # This holds because W = L_k^{-1,full} @ H and the relationship
    # through the TPS system.
    # Instead, verify via direct computation:
    be2 = 0.0
    for d in range(source.shape[1]):
        H_d = target[:, d]
        be2 += H_d @ Lk_inv @ H_d

    # These should be approximately equal
    assert_allclose(be1, be2, atol=1e-6)
