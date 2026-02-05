import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from ktch.datasets import load_landmark_mosquito_wings
from ktch.landmark import (
    GeneralizedProcrustesAnalysis,
    centroid_size,
    combine_landmarks_and_curves,
    define_curve_sliders,
)

data_landmark_mosquito_wings = load_landmark_mosquito_wings(as_frame=True)
data_landmark_mosquito_wings.coords

X = data_landmark_mosquito_wings.coords.to_numpy().reshape(-1, 18 * 2)


def test_gpa_shape():
    gpa = GeneralizedProcrustesAnalysis()
    gpa.fit_transform(X)
    X_transformed = gpa.fit_transform(X)

    assert X.shape == X_transformed.shape


@pytest.mark.parametrize("n_dim", [2, 3])
def test_centroid_size(n_dim):
    x = np.random.uniform(0, 100, (10, n_dim))
    cs_r = np.sqrt(np.sum((x - x.mean(axis=0)) ** 2))
    cs_t = centroid_size(x)

    assert_array_almost_equal(cs_r, cs_t)


###########################################################
#
#   define_curve_sliders
#
###########################################################


def test_define_curve_sliders_open():
    """Test define_curve_sliders with open curve."""
    result = define_curve_sliders([0, 1, 2, 3, 4])
    expected = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    assert_array_equal(result, expected)


def test_define_curve_sliders_closed_curve_pattern():
    """Test define_curve_sliders for closed curve using repeated anchor index."""
    result = define_curve_sliders([0, 1, 2, 3, 0])
    expected = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 0]])
    assert_array_equal(result, expected)


def test_define_curve_sliders_minimum():
    """Test define_curve_sliders with minimum 3 points."""
    result = define_curve_sliders([5, 6, 7])
    expected = np.array([[5, 6, 7]])
    assert_array_equal(result, expected)


def test_define_curve_sliders_too_few_points():
    """Test define_curve_sliders raises error with less than 3 points."""
    with pytest.raises(ValueError, match="At least 3 points"):
        define_curve_sliders([0, 1])


def test_define_curve_sliders_combine_multiple():
    """Test combining multiple curves."""
    curve1 = define_curve_sliders([0, 1, 2, 3])
    curve2 = define_curve_sliders([3, 4, 5, 6])
    combined = np.vstack([curve1, curve2])
    expected = np.array([[0, 1, 2], [1, 2, 3], [3, 4, 5], [4, 5, 6]])
    assert_array_equal(combined, expected)


###########################################################
#
#   combine_landmarks_and_curves
#
###########################################################


def test_combine_landmarks_and_curves_basic():
    """Test basic combine_landmarks_and_curves functionality."""
    landmarks = np.array(
        [
            [[0, 0], [1, 0], [1, 1]],
            [[0, 0], [1, 0], [1, 1]],
        ]
    )
    curves = [
        [np.array([[0.2, 0.2], [0.4, 0.4], [0.6, 0.6], [0.8, 0.8]])],
        [np.array([[0.2, 0.2], [0.4, 0.4], [0.6, 0.6], [0.8, 0.8]])],
    ]

    combined, slider_matrix, info = combine_landmarks_and_curves(
        landmarks, curves, curve_landmarks=[(0, 2)]
    )

    assert combined.shape == (2, 7, 2)  # 3 landmarks + 4 curve points
    assert slider_matrix.shape == (4, 3)  # all 4 curve points slide
    assert info["n_landmarks"] == 3
    assert info["n_curve_points"] == 4


def test_combine_landmarks_and_curves_slider_indices():
    """Test that slider indices are correct with anchor flags."""
    landmarks = np.array(
        [
            [[0, 0], [5, 0]],
        ]
    )
    curves = [
        [np.array([[1, 1], [2, 2], [3, 3], [4, 4]])],
    ]

    combined, slider_matrix, info = combine_landmarks_and_curves(
        landmarks, curves, anchor_first=True, anchor_last=True
    )

    # With anchor_first=True, anchor_last=True:
    # Points 0,1 are landmarks (indices 0,1)
    # Curve points are at indices 2,3,4,5
    # First (index 2) and last (index 5) are anchors
    # Sliders are indices 3,4

    assert slider_matrix.shape[0] == 2  # 2 sliders
    assert np.all(slider_matrix[:, 1] == [3, 4])  # slider indices


def test_combine_landmarks_and_curves_with_curve_landmarks():
    """Test that curve_landmarks produces correct slider topology."""
    landmarks = np.array(
        [
            [[0, 0], [5, 0], [5, 5]],
        ]
    )
    curves = [
        [np.array([[1, 1], [2, 2], [3, 3]])],
    ]

    combined, slider_matrix, info = combine_landmarks_and_curves(
        landmarks, curves, curve_landmarks=[(0, 2)]
    )

    # 3 landmarks (0,1,2) + 3 curve points (3,4,5)
    # All curve points slide with LM0 and LM2 as anchors
    assert slider_matrix.shape == (3, 3)
    # First semilandmark: before=LM0, slider=3, after=4
    assert np.all(slider_matrix[0] == [0, 3, 4])
    # Middle: before=3, slider=4, after=5
    assert np.all(slider_matrix[1] == [3, 4, 5])
    # Last semilandmark: before=4, slider=5, after=LM2
    assert np.all(slider_matrix[2] == [4, 5, 2])


###########################################################
#
#   GPA with semilandmarks
#
###########################################################


@pytest.fixture
def semilandmark_test_data():
    """Create test data for GPA with semilandmarks."""
    np.random.seed(42)
    base = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [2.0, 1.0],
            [3.0, 0.5],
            [4.0, 0.0],
        ]
    )
    X = np.array(
        [
            base + np.random.randn(5, 2) * 0.1,
            base + np.random.randn(5, 2) * 0.1,
            base + np.random.randn(5, 2) * 0.1,
        ]
    )
    X_flat = X.reshape(3, -1)
    curves = define_curve_sliders([0, 1, 2, 3, 4])
    return X_flat, curves


def test_gpa_with_semilandmarks_procrustes(semilandmark_test_data):
    """Test GPA with semilandmarks using Procrustes distance criterion."""
    X_flat, curves = semilandmark_test_data

    gpa = GeneralizedProcrustesAnalysis(
        n_dim=2,
        curves=curves,
        sliding_criterion="procrustes_distance",
    )
    X_aligned = gpa.fit_transform(X_flat)

    assert X_aligned.shape == X_flat.shape
    assert gpa.mu_ is not None
    assert gpa.mu_.shape == (5, 2)


def test_gpa_with_semilandmarks_bending_energy(semilandmark_test_data):
    """Test GPA with semilandmarks using bending energy criterion."""
    X_flat, curves = semilandmark_test_data

    gpa = GeneralizedProcrustesAnalysis(
        n_dim=2,
        curves=curves,
        sliding_criterion="bending_energy",
    )
    X_aligned = gpa.fit_transform(X_flat)

    assert X_aligned.shape == X_flat.shape
    assert gpa.mu_ is not None


def test_gpa_backward_compatibility(semilandmark_test_data):
    """Test that GPA without curves produces same results as before."""
    X_flat, _ = semilandmark_test_data

    # Without curves parameter
    gpa1 = GeneralizedProcrustesAnalysis(n_dim=2)
    X1 = gpa1.fit_transform(X_flat)

    # With curves=None explicitly
    gpa2 = GeneralizedProcrustesAnalysis(n_dim=2, curves=None)
    X2 = gpa2.fit_transform(X_flat)

    assert_array_almost_equal(X1, X2)


###########################################################
#
#   parameter validation
#
###########################################################


def test_gpa_invalid_sliding_criterion():
    """Test that invalid sliding_criterion raises error."""
    X = np.random.randn(3, 10)
    gpa = GeneralizedProcrustesAnalysis(
        n_dim=2,
        curves=np.array([[0, 1, 2]]),
        sliding_criterion="invalid",
    )
    with pytest.raises(ValueError, match="sliding_criterion must be"):
        gpa.fit(X)


def test_gpa_invalid_curves_shape():
    """Test that invalid curves shape raises error."""
    X = np.random.randn(3, 10)
    gpa = GeneralizedProcrustesAnalysis(
        n_dim=2,
        curves=np.array([0, 1, 2]),  # 1D instead of 2D
    )
    with pytest.raises(ValueError, match="curves must be a 2D array"):
        gpa.fit(X)


def test_gpa_curves_index_out_of_range():
    """Test that out-of-range curve indices raise error."""
    X = np.random.randn(3, 10)  # 5 landmarks * 2 dims
    gpa = GeneralizedProcrustesAnalysis(
        n_dim=2,
        curves=np.array([[0, 1, 100]]),  # index 100 is out of range
    )
    with pytest.raises(ValueError, match="curves indices must be in range"):
        gpa.fit(X)


def test_gpa_surfaces_not_implemented():
    """Test that surfaces parameter raises NotImplementedError."""
    X = np.random.randn(3, 10)
    gpa = GeneralizedProcrustesAnalysis(
        n_dim=2,
        surfaces=np.array([0, 1, 2]),
    )
    with pytest.raises(NotImplementedError, match="Surface semilandmarks"):
        gpa.fit(X)


###########################################################
#
#   closed-form bending energy sliding
#
###########################################################


def test_bending_energy_sliding_reduces_energy():
    """Test that sliding reduces bending energy."""
    from ktch.landmark._kernels import tps_bending_energy

    np.random.seed(42)
    base = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [2.0, 1.0],
            [3.0, 0.5],
            [4.0, 0.0],
        ]
    )
    # Perturb the semilandmarks
    specimen = base.copy()
    specimen[1] += np.array([0.0, 0.2])
    specimen[2] += np.array([0.0, -0.3])
    specimen[3] += np.array([0.0, 0.15])

    curves = define_curve_sliders([0, 1, 2, 3, 4])
    gpa = GeneralizedProcrustesAnalysis(
        n_dim=2,
        curves=curves,
        sliding_criterion="bending_energy",
    )

    # Compute bending energy before sliding
    be_before = tps_bending_energy(base, specimen)

    # Slide
    slid = gpa._slide_bending_energy(specimen, base, curves)

    # Compute bending energy after sliding
    be_after = tps_bending_energy(base, slid)

    assert be_after < be_before


def test_bending_energy_sliding_only_moves_sliders():
    """Test that sliding only moves slider points, not anchors."""
    np.random.seed(42)
    base = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [2.0, 1.0],
            [3.0, 0.5],
            [4.0, 0.0],
        ]
    )
    specimen = base + np.random.randn(5, 2) * 0.1

    curves = define_curve_sliders([0, 1, 2, 3, 4])
    # Sliders are indices 1, 2, 3; anchors are 0, 4
    gpa = GeneralizedProcrustesAnalysis(
        n_dim=2,
        curves=curves,
        sliding_criterion="bending_energy",
    )

    slid = gpa._slide_bending_energy(specimen, base, curves)

    # Anchor points (0 and 4) should not move
    assert_array_almost_equal(slid[0], specimen[0])
    assert_array_almost_equal(slid[4], specimen[4])

    # At least some slider points should have moved
    assert not np.allclose(slid[1:4], specimen[1:4])


###########################################################
#
#   re-projection onto curves
#
###########################################################


def test_reproject_onto_curves_basic():
    """Test that re-projection places points on the line segment."""
    base = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ]
    )
    # Slide point 2 off-curve (perpendicular displacement)
    slid = base.copy()
    slid[2] = [2.0, 0.5]

    curves = define_curve_sliders([0, 1, 2, 3, 4])
    gpa = GeneralizedProcrustesAnalysis(n_dim=2, curves=curves)
    result = gpa._reproject_onto_curves(slid, base, curves)

    # Point 2 should be projected back onto the segment [1,0]->[3,0] at (2,0)
    assert_array_almost_equal(result[2], [2.0, 0.0])
    # Anchors should be unchanged
    assert_array_almost_equal(result[0], base[0])
    assert_array_almost_equal(result[4], base[4])


def test_reproject_onto_curves_preserves_anchors():
    """Test that re-projection does not move anchor points."""
    base = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
        ]
    )
    slid = base.copy()
    slid[1] = [1.5, 1.5]  # move slider off-curve

    curves = define_curve_sliders([0, 1, 2])
    gpa = GeneralizedProcrustesAnalysis(n_dim=2, curves=curves)
    result = gpa._reproject_onto_curves(slid, base, curves)

    assert_array_almost_equal(result[0], base[0])
    assert_array_almost_equal(result[2], base[2])


def test_reproject_onto_curves_3d():
    """Test re-projection in 3D space."""
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ]
    )
    slid = base.copy()
    slid[1] = [1.0, 1.5, 0.5]  # move off the line

    curves = define_curve_sliders([0, 1, 2])
    gpa = GeneralizedProcrustesAnalysis(n_dim=3, curves=curves)
    result = gpa._reproject_onto_curves(slid, base, curves)

    # The segment goes from (0,0,0) to (2,2,2).
    # Projection of (1,1.5,0.5) onto this line: t = dot(P-A,AB)/dot(AB,AB)
    # AB = (2,2,2), P-A = (1,1.5,0.5), dot = 2+3+1 = 6, |AB|^2 = 12
    # t = 0.5, projected = (1,1,1)
    assert_array_almost_equal(result[1], [1.0, 1.0, 1.0])


def test_reproject_clamps_to_segment():
    """Test that re-projection clamps to segment endpoints."""
    base = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ]
    )
    slid = base.copy()
    slid[1] = [-1.0, 0.0]  # beyond point A

    curves = define_curve_sliders([0, 1, 2])
    gpa = GeneralizedProcrustesAnalysis(n_dim=2, curves=curves)
    result = gpa._reproject_onto_curves(slid, base, curves)

    # Should clamp to segment start (point A = [0,0])
    assert_array_almost_equal(result[1], [0.0, 0.0])


def test_reproject_two_segments_selects_nearest():
    """Test that re-projection picks the nearest of the two segments."""
    # L-shaped curve: B=(0,0), S_orig=(1,0), A=(1,1)
    # Segment B->S_orig is horizontal, segment S_orig->A is vertical
    base = np.array(
        [
            [0.0, 0.0],  # B (anchor)
            [1.0, 0.0],  # S_orig (slider)
            [1.0, 1.0],  # A (anchor)
        ]
    )
    curves = define_curve_sliders([0, 1, 2])
    gpa = GeneralizedProcrustesAnalysis(n_dim=2, curves=curves)

    # Point near the horizontal segment B->S_orig
    slid_h = base.copy()
    slid_h[1] = [0.5, 0.1]
    result_h = gpa._reproject_onto_curves(slid_h, base, curves)
    assert_array_almost_equal(result_h[1], [0.5, 0.0])

    # Point near the vertical segment S_orig->A
    slid_v = base.copy()
    slid_v[1] = [1.1, 0.5]
    result_v = gpa._reproject_onto_curves(slid_v, base, curves)
    assert_array_almost_equal(result_v[1], [1.0, 0.5])


def test_reproject_two_segments_3d():
    """Test 2-segment re-projection in 3D with non-collinear curve."""
    # V-shaped curve in 3D: B=(0,0,0), S_orig=(1,0,0), A=(1,1,0)
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    curves = define_curve_sliders([0, 1, 2])
    gpa = GeneralizedProcrustesAnalysis(n_dim=3, curves=curves)

    # Displace slider above the plane, closer to segment S_orig->A
    slid = base.copy()
    slid[1] = [1.05, 0.5, 0.3]
    result = gpa._reproject_onto_curves(slid, base, curves)

    # Projection onto S_orig->A: S_orig=(1,0,0), A=(1,1,0), SA=(0,1,0)
    # P-S=(0.05,0.5,0.3), t = 0.5, proj = (1,0.5,0)
    # Projection onto B->S_orig: B=(0,0,0), S_orig=(1,0,0), BS=(1,0,0)
    # P-B=(1.05,0.5,0.3), t = 1.05 clamped to 1.0, proj = (1,0,0)
    # dist to S_orig->A proj: (0.05)^2 + 0 + 0.09 = 0.0925
    # dist to B->S_orig proj: (0.05)^2 + 0.25 + 0.09 = 0.3425
    # Nearest is S_orig->A projection
    assert_array_almost_equal(result[1], [1.0, 0.5, 0.0])


def test_sliding_with_reprojection_reduces_energy():
    """Test that sliding with re-projection still reduces bending energy."""
    from ktch.landmark._kernels import tps_bending_energy

    np.random.seed(42)
    base = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [2.0, 1.0],
            [3.0, 0.5],
            [4.0, 0.0],
        ]
    )
    specimen = base.copy()
    specimen[1] += np.array([0.0, 0.2])
    specimen[2] += np.array([0.0, -0.3])
    specimen[3] += np.array([0.0, 0.15])

    curves = define_curve_sliders([0, 1, 2, 3, 4])
    gpa = GeneralizedProcrustesAnalysis(
        n_dim=2,
        curves=curves,
        sliding_criterion="bending_energy",
    )

    be_before = tps_bending_energy(base, specimen)

    # _slide_semilandmarks now includes re-projection
    X = np.array([specimen])
    X_curve_geom = X.copy()
    X_slid = gpa._slide_semilandmarks(X, base, curves, X_curve_geom)

    be_after = tps_bending_energy(base, X_slid[0])
    assert be_after < be_before


###########################################################
#
#   3D curve semilandmarks
#
###########################################################


@pytest.fixture
def semilandmark_3d_test_data():
    """Create 3D test data for GPA with curve semilandmarks."""
    np.random.seed(123)
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.5, 0.2],
            [2.0, 1.0, 0.4],
            [3.0, 0.5, 0.2],
            [4.0, 0.0, 0.0],
        ]
    )
    X = np.array(
        [
            base + np.random.randn(5, 3) * 0.05,
            base + np.random.randn(5, 3) * 0.05,
            base + np.random.randn(5, 3) * 0.05,
        ]
    )
    X_flat = X.reshape(3, -1)
    curves = define_curve_sliders([0, 1, 2, 3, 4])
    return X_flat, curves


def test_gpa_with_3d_curve_semilandmarks_bending_energy(semilandmark_3d_test_data):
    """Test GPA with 3D curve semilandmarks using bending energy criterion."""
    X_flat, curves = semilandmark_3d_test_data

    gpa = GeneralizedProcrustesAnalysis(
        n_dim=3,
        curves=curves,
        sliding_criterion="bending_energy",
    )
    X_aligned = gpa.fit_transform(X_flat)

    assert X_aligned.shape == X_flat.shape
    assert gpa.mu_ is not None
    assert gpa.mu_.shape == (5, 3)


def test_gpa_with_3d_curve_semilandmarks_procrustes(semilandmark_3d_test_data):
    """Test GPA with 3D curve semilandmarks using Procrustes distance criterion."""
    X_flat, curves = semilandmark_3d_test_data

    gpa = GeneralizedProcrustesAnalysis(
        n_dim=3,
        curves=curves,
        sliding_criterion="procrustes_distance",
    )
    X_aligned = gpa.fit_transform(X_flat)

    assert X_aligned.shape == X_flat.shape
    assert gpa.mu_ is not None
    assert gpa.mu_.shape == (5, 3)


def test_3d_tangent_computation():
    """Test that tangent vectors in 3D are correctly computed."""
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    curves = define_curve_sliders([0, 1, 2, 3, 4])
    gpa = GeneralizedProcrustesAnalysis(n_dim=3, curves=curves)

    tangents = gpa._compute_tangents(X, curves)

    # All tangents should be unit vectors along x-axis
    for t in tangents:
        assert_array_almost_equal(t, [1.0, 0.0, 0.0])

    # Non-axis-aligned case
    X2 = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ]
    )
    curves2 = define_curve_sliders([0, 1, 2])
    tangents2 = gpa._compute_tangents(X2, curves2)
    expected = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    assert_array_almost_equal(tangents2[0], expected)


def test_bending_energy_sliding_moves_along_tangent():
    """Test that sliding displacement is parallel to the tangent direction."""
    base = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ]
    )
    specimen = base.copy()
    specimen[2] += np.array([0.0, 0.5])  # Move middle point off the line

    curves = define_curve_sliders([0, 1, 2, 3, 4])
    gpa = GeneralizedProcrustesAnalysis(
        n_dim=2,
        curves=curves,
        sliding_criterion="bending_energy",
    )

    # Compute tangents from specimen (same as _slide_bending_energy does)
    tangents = gpa._compute_tangents(specimen, curves)

    slid = gpa._slide_bending_energy(specimen, base, curves)

    slider_idx = curves[:, 1]
    for j, s_idx in enumerate(slider_idx):
        displacement = slid[s_idx] - specimen[s_idx]
        if np.linalg.norm(displacement) < 1e-14:
            continue
        # Displacement should be parallel to tangent:
        # cross product (2D) should be zero
        cross = displacement[0] * tangents[j][1] - displacement[1] * tangents[j][0]
        assert abs(cross) < 1e-10
