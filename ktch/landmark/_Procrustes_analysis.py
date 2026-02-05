"""Procrustes Analysis"""

# Copyright 2020 Koji Noshita
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from abc import ABCMeta
from typing import Optional

import numpy as np
import numpy.typing as npt
import scipy as sp
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin

from ._kernels import tps_bending_energy, tps_bending_energy_matrix, tps_system_matrix


class GeneralizedProcrustesAnalysis(
    OneToOneFeatureMixin,
    TransformerMixin,
    BaseEstimator,
    metaclass=ABCMeta,
):
    r"""
    Generalized Procrustes Analysis (GPA)

    Parameters
    ------------
    tol: float, default=10^-7
        Tolerance for convergence of Procrustes analysis.

    scaling: bool, default=True
        If True, scale configurations to unit centroid size (shape analysis).
        If False, only translate and rotate (size-and-shape analysis).

    n_dim: int, default=2
        Dimensions of the configurations (2 or 3).

    curves: array-like, shape (n_sliders, 3), default=None
        Curve semilandmark topology matrix.
        Each row specifies ``[before_index, slider_index, after_index]``
        defining the sliding direction along the curve tangent.

    surfaces: array-like, shape (n_surface, 3), default=None
        Surface semilandmark topology matrix.
        Each row specifies ``[patch_id, landmark_index, normal_anchor_index]``
        where `patch_id` identifies the surface patch,
        `landmark_index` is the semilandmark to slide,
        and `normal_anchor_index` is a nearby fixed landmark
        used to orient the local tangent plane.

        Currently not implemented; will raise `NotImplementedError` if provided.

    sliding_criterion: str, default='bending_energy'
        - 'bending_energy': Minimize thin-plate spline bending energy
        - 'procrustes_distance': Minimize Procrustes distance to reference

    n_jobs: int, default=None
        The number of parallel jobs to run for Procrustes analysis.

    Attributes
    ------------
    mu_: ndarray, shape (n_landmarks, n_dim)
        The mean shape of the aligned shapes.
    n_dim_: int, 2 or 3
        Dimensions of the configurations.

    Notes
    ------------
    GPA for shape involves translating, rotating, and scaling the configurations
    to each other to minimize the sum of the squared distances with respect to
    positional, rotational, and size parameters, subject to a size constraint
    [Gower_1975]_, [Goodall_1991]_.

    When semilandmarks are specified via the `curves` parameter,
    they are slid along their tangent directions to minimize the chosen
    criterion (bending energy or Procrustes distance) during each GPA iteration
    [Bookstein_1997]_, [Gunz_2013]_.

    References
    ------------
    .. [Gower_1975] Gower, J.C., 1975. Generalized procrustes analysis.
       Psychometrika 40, 33-51.
    .. [Goodall_1991] Goodall, C., 1991. Procrustes Methods in the Statistical
       Analysis of Shape. J Royal Stat Soc B 53, 285-321.
    .. [Bookstein_1997] Bookstein, F.L., 1997. Landmark methods for forms
       without landmarks. Medical Image Analysis 1, 225-243.
    .. [Gunz_2013] Gunz, P., Mitteroecker, P., 2013. Semilandmarks: a method
       for quantifying curves and surfaces. Hystrix 24, 103-109.

    """

    def __init__(
        self,
        tol=10**-7,
        scaling=True,
        n_dim=2,
        curves: Optional[npt.ArrayLike] = None,
        surfaces: Optional[npt.ArrayLike] = None,
        sliding_criterion: str = "bending_energy",
        n_jobs: Optional[int] = None,
        debug=False,
    ):
        self.tol = tol
        self.scaling = scaling
        self.debug = debug
        self.mu_ = None
        self.n_dim_ = n_dim
        self.curves = curves
        self.surfaces = surfaces
        self.sliding_criterion = sliding_criterion
        self.n_jobs = n_jobs

    @property
    def n_dim(self):
        return self.n_dim_

    @n_dim.setter
    def n_dim(self, n_dim):
        if n_dim not in [2, 3]:
            raise ValueError("n_dim must be 2 or 3.")
        self.n_dim_ = n_dim

    def _validate_semilandmark_params(self, n_landmarks):
        """Validate semilandmark parameters.

        Parameters
        ----------
        n_landmarks : int
            Total number of landmarks in the data.

        Raises
        ------
        ValueError
            If parameters are invalid.
        NotImplementedError
            If surfaces are specified (not yet implemented).
        """
        if self.surfaces is not None:
            raise NotImplementedError("Surface semilandmarks are not yet implemented. ")

        if self.sliding_criterion not in ["bending_energy", "procrustes_distance"]:
            raise ValueError(
                f"sliding_criterion must be 'bending_energy' or 'procrustes_distance', "
                f"got '{self.sliding_criterion}'."
            )

        if self.curves is not None:
            curves = np.asarray(self.curves)
            if curves.ndim != 2 or curves.shape[1] != 3:
                raise ValueError(
                    f"curves must be a 2D array with shape (n_sliders, 3), "
                    f"got shape {curves.shape}."
                )
            if curves.min() < 0 or curves.max() >= n_landmarks:
                raise ValueError(
                    f"curves indices must be in range [0, {n_landmarks - 1}], "
                    f"got range [{curves.min()}, {curves.max()}]."
                )

            slider_indices = curves[:, 1]
            if len(slider_indices) != len(np.unique(slider_indices)):
                raise ValueError(
                    "Duplicate slider indices found in curves. "
                    "Each semilandmark can only be a slider in one curve segment."
                )

    def fit(self, X: npt.ArrayLike) -> GeneralizedProcrustesAnalysis:
        """Fit GPA.

        Parameters
        ----------
        X : array-like, shape (n_specimens, n_landmarks, n_dim)
            or DataFrame, shape (n_specimens, n_landmarks * n_dim)
            Configurations to be aligned.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns

        # Validate semilandmark parameters
        n_dim = self.n_dim_
        if self.n_features_in_ % n_dim != 0:
            raise ValueError("X must have n_landmarks * n_dim features.")
        n_landmarks = self.n_features_in_ // n_dim
        self._validate_semilandmark_params(n_landmarks)

        return self

    def transform(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """GPA for shapes/size-and-shapes.

        Parameters
        ----------
        X : array-like, shape (n_specimens, n_landmarks*n_dim)
            Configurations to be aligned.

        Returns
        -------
        X_ : ndarray, shape (n_specimens, n_landmarks, n_dim)
            Shapes/Size-and-Shape
        """

        X_ = np.array(X)
        n_specimen = len(X_)
        n_dim = self.n_dim_
        if len(X_[0]) % n_dim != 0:
            raise ValueError("X must be n_specimens x n_landmarks*n_dim.")
        n_landmarks = int(len(X_[0]) / n_dim)
        X_ = X_.reshape(n_specimen, n_landmarks, n_dim)

        scaling = self.scaling
        if scaling:
            X_ = self._transform_shape(X_)
        else:
            X_ = self._transform_size_and_shape(X_)

        return X_.reshape(n_specimen, n_landmarks * n_dim)

    def _transform_shape(self, X):
        X_ = np.array(X, dtype=np.double, copy=True)
        X_ = self._center(X_)
        X_ = self._scale(X_)
        mu = np.sum(X_, axis=0) / len(X_)
        mu = mu / centroid_size(mu)
        if self.debug:
            print("mu: ", mu, "centroid_size(mu): ", centroid_size(mu))

        curves = None
        if self.curves is not None:
            curves = np.asarray(self.curves)

        # Store original curve geometry for re-projection.
        X_curve_geom = X_.copy() if curves is not None else None

        diff_disp = np.inf
        total_disp_prev = np.inf
        while diff_disp > self.tol:
            if curves is not None:
                # GPA with semilandmarks
                total_disp = 0.0
                for i in range(len(X_)):
                    R, _ = sp.linalg.orthogonal_procrustes(X_[i], mu)
                    X_[i] = X_[i] @ R
                    X_curve_geom[i] = X_curve_geom[i] @ R
                    total_disp += np.sum((X_[i] - mu) ** 2)

                # Slide semilandmarks and re-project onto stored curve geometry
                X_ = self._slide_semilandmarks(X_, mu, curves, X_curve_geom)

                for i in range(len(X_)):
                    center = np.mean(X_[i], axis=0)
                    X_[i] -= center
                    X_curve_geom[i] -= center
                    cs = centroid_size(X_[i])
                    X_[i] /= cs
                    X_curve_geom[i] /= cs
            else:
                # GPA (without semilandmarks)
                results = [sp.spatial.procrustes(mu, x) for x in X_]
                X_ = np.array([x_aligned for _, x_aligned, _ in results])
                total_disp = np.sum(np.array([disp for _, _, disp in results]))

            diff_disp = np.abs(total_disp_prev - total_disp)
            total_disp_prev = total_disp

            mu = np.sum(X_, axis=0) / len(X_)
            mu = mu / centroid_size(mu)

            if self.debug:
                print("total_disp: ", total_disp, "diff_disp: ", diff_disp)

        self.mu_ = mu

        return X_

    def _transform_size_and_shape(self, X):
        X_ = np.array(X, dtype=np.double, copy=True)
        X_ = self._center(X_)
        mu = np.sum(X_, axis=0) / len(X_)

        diff_disp = np.inf
        total_disp_prev = np.inf
        while diff_disp > self.tol:
            total_disp = 0
            X_ = np.array([x @ sp.linalg.orthogonal_procrustes(x, mu)[0] for x in X])
            total_disp = np.sum([(x_ - mu) ** 2 for x_ in X_])

            diff_disp = np.abs(total_disp_prev - total_disp)
            total_disp_prev = total_disp
            mu = np.sum(X_, axis=0) / len(X_)

            if self.debug:
                print("total_disp: ", total_disp, "diff_disp: ", diff_disp)

        self.mu_ = mu

        return X_.reshape(self.n_specimen_, self.n_landmarks_ * self.n_dim_)

    def fit_transform(self, X):
        """GPA for shapes/size-and-shapes.

        Parameters
        ----------
        X : array-like, shape (n_specimens, n_landmarks * n_dim)
            /DataFrame, shape (n_specimens, n_landmarks * n_dim)
            Configurations to be aligned.

        Returns
        -------
        X_ : ndarray, shape (n_specimens, n_landmarks, n_dim)
            Shapes/Size-and-Shape

        """
        self.fit(X)
        X_ = self.transform(X)
        return X_

    def _center(self, X):
        X_centered = np.array([x - np.mean(x, axis=0) for x in X])
        return X_centered

    def _scale(self, X):
        X_scaled = np.array([x / centroid_size(x) for x in X])
        return X_scaled

    def _compute_tangents(self, X, curves):
        """Compute tangent directions for curve semilandmarks.

        Parameters
        ----------
        X : ndarray, shape (n_landmarks, n_dim)
            Single configuration.
        curves : ndarray, shape (n_sliders, 3)
            Topology matrix [before, slider, after].

        Returns
        -------
        tangents : ndarray, shape (n_sliders, n_dim)
            Unit tangent vectors for each slider.
        """
        before_idx = curves[:, 0]
        after_idx = curves[:, 2]

        tangent_vectors = X[after_idx] - X[before_idx]
        norms = np.linalg.norm(tangent_vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # Avoid division by zero
        tangents = tangent_vectors / norms

        return tangents

    def _slide_procrustes(self, X, reference, curves):
        """Slide semilandmarks to minimize Procrustes distance to reference.

        Parameters
        ----------
        X : ndarray, shape (n_landmarks, n_dim)
            Single configuration to slide.
        reference : ndarray, shape (n_landmarks, n_dim)
            Reference configuration (usually mean shape).
        curves : ndarray, shape (n_sliders, 3)
            Topology matrix [before, slider, after].

        Returns
        -------
        X_slid : ndarray, shape (n_landmarks, n_dim)
            Configuration with sliders moved along tangents.
        """
        X_slid = X.copy()
        slider_idx = curves[:, 1]

        tangents = self._compute_tangents(X_slid, curves)

        displacements = reference[slider_idx] - X_slid[slider_idx]

        # Project displacement onto tangent: d_parallel = (d · t) * t
        dot_products = np.sum(displacements * tangents, axis=1, keepdims=True)
        parallel_displacements = dot_products * tangents

        X_slid[slider_idx] += parallel_displacements

        return X_slid

    def _slide_bending_energy(self, X, reference, curves):
        """Slide semilandmarks to minimize TPS bending energy.

        Uses the closed-form weighted least squares solution from
        Bookstein (1997), Eq. 8:

        $$ T = -(U^T L_k^{-1,block} U)^{-1} U^T L_k^{-1,block} Y^0 $$

        where $L_k^{-1}$ is the bending energy matrix (upper-left k x k block
        of $L^{-1}$), U encodes the tangent sliding directions, and $Y^0$ are
        the initial positions.

        Parameters
        ----------
        X : ndarray, shape (n_landmarks, n_dim)
            Single configuration to slide.
        reference : ndarray, shape (n_landmarks, n_dim)
            Reference configuration (usually mean shape).
        curves : ndarray, shape (n_sliders, 3)
            Topology matrix [before, slider, after].

        Returns
        -------
        X_slid : ndarray, shape (n_landmarks, n_dim)
            Configuration with sliders moved along tangents.
        """
        X_slid = X.copy()
        slider_idx = curves[:, 1]
        k = len(X)
        n_dim = X.shape[1]
        n_sliders = len(slider_idx)

        # Compute bending energy matrix from reference
        Lk_inv = tps_bending_energy_matrix(reference)
        Lk_inv_block = sp.linalg.block_diag(*[Lk_inv] * n_dim)

        # Compute tangent directions
        tangents = self._compute_tangents(X_slid, curves)

        # Build constraint matrix U (k * n_dim, n_sliders)
        U = np.zeros((k * n_dim, n_sliders))
        for j, (s_idx, tangent) in enumerate(zip(slider_idx, tangents)):
            for d in range(n_dim):
                U[d * k + s_idx, j] = tangent[d]

        # Solve Eq. 8
        UtL = U.T @ Lk_inv_block
        A_mat = UtL @ U
        Y0 = X_slid.T.ravel()  # shape (k * n_dim,)
        b_vec = UtL @ Y0
        T, _, _, _ = np.linalg.lstsq(A_mat, -b_vec, rcond=None)

        # Update positions Y = Y^0 + U T
        Y = Y0 + U @ T
        X_slid = Y.reshape(n_dim, k).T

        return X_slid

    def _reproject_onto_curves(self, X_slid, X_curve_geom, curves):
        """Re-project slid semilandmarks back onto the original curve geometry.

        After sliding, each semilandmark may have drifted off its original curve.
        For each slider with neighbors B (before) and A (after) and original
        position S_orig, this method projects the slid position onto the two
        piecewise-linear segments B-S_orig and S_orig-A, then selects the
        nearest projection (Gunz et al., 2005).

        Parameters
        ----------
        X_slid : ndarray, shape (n_landmarks, n_dim)
            Configuration after sliding.
        X_curve_geom : ndarray, shape (n_landmarks, n_dim)
            Original curve geometry.
        curves : ndarray, shape (n_sliders, 3)
            Topology matrix [before, slider, after].

        Returns
        -------
        X_out : ndarray, shape (n_landmarks, n_dim)
            Configuration with sliders re-projected onto the original curve.
        """
        X_out = X_slid.copy()
        for s_before, s_idx, s_after in curves:
            P = X_slid[s_idx]
            B = X_curve_geom[s_before]
            S_orig = X_curve_geom[s_idx]
            A = X_curve_geom[s_after]

            # Project P onto B-S_orig
            BS = S_orig - B
            denom_bs = np.dot(BS, BS)
            if denom_bs > 1e-30:
                t_bs = np.clip(np.dot(P - B, BS) / denom_bs, 0.0, 1.0)
                proj_bs = B + t_bs * BS
            else:
                proj_bs = B

            # Project P onto S_orig-A
            SA = A - S_orig
            denom_sa = np.dot(SA, SA)
            if denom_sa > 1e-30:
                t_sa = np.clip(np.dot(P - S_orig, SA) / denom_sa, 0.0, 1.0)
                proj_sa = S_orig + t_sa * SA
            else:
                proj_sa = S_orig

            # Nearest projection
            dist_bs = np.sum((P - proj_bs) ** 2)
            dist_sa = np.sum((P - proj_sa) ** 2)
            X_out[s_idx] = proj_bs if dist_bs <= dist_sa else proj_sa

        return X_out

    def _slide_semilandmarks(self, X, reference, curves, X_curve_geom):
        """Slide semilandmarks according to sliding_criterion.

        Parameters
        ----------
        X : ndarray, shape (n_specimens, n_landmarks, n_dim)
            Configurations to slide.
        reference : ndarray, shape (n_landmarks, n_dim)
            Reference configuration.
        curves : ndarray, shape (n_sliders, 3)
            Topology matrix [before, slider, after].
        X_curve_geom : ndarray, shape (n_specimens, n_landmarks, n_dim)
            Original curve geometry for re-projection.

        Returns
        -------
        X_slid : ndarray, shape (n_specimens, n_landmarks, n_dim)
            Configurations with sliders moved.
        """
        X_slid = np.array(X, copy=True)

        if self.sliding_criterion == "procrustes_distance":
            slide_func = self._slide_procrustes
        else:
            slide_func = self._slide_bending_energy

        for i in range(len(X_slid)):
            X_slid[i] = slide_func(X_slid[i], reference, curves)
            X_slid[i] = self._reproject_onto_curves(X_slid[i], X_curve_geom[i], curves)

        return X_slid

    # @property
    # def _n_features_out(self):
    #     """Number of transformed output features."""
    #     return self.n_landmarks_ * self.n_dim_


class LandmarkImputer(TransformerMixin, BaseEstimator):
    def __init__(self, missing_values=np.nan):
        pass


###########################################################
#
#   utility functions
#
###########################################################


def define_curve_sliders(indices):
    """Generate slider matrix from sequential landmark indices.

    Creates a topology matrix for curve semilandmarks where each row
    specifies [before_index, slider_index, after_index].
    The first and last indices are treated as fixed anchors and do not slide.

    Parameters
    ----------
    indices : array-like
        Sequential landmark indices along the curve.
        The first and last points are fixed anchors and do not slide.
        For closed curves, repeat the anchor index at both ends
        (e.g., ``[0, 1, 2, 3, 0]``).

    Returns
    -------
    slider_matrix : ndarray, shape (n_sliders, 3)
        Topology matrix where each row is [before, slider, after].
        n_sliders = len(indices) - 2.

    See Also
    --------
    GeneralizedProcrustesAnalysis : GPA with semilandmark support.
    combine_landmarks_and_curves : Convert TPS format to GPA format.
    """
    indices = np.asarray(indices)
    n = len(indices)

    if n < 3:
        raise ValueError(f"At least 3 points are required to define a curve, got {n}.")

    sliders = []
    for i in range(1, n - 1):
        before = indices[i - 1]
        slider = indices[i]
        after = indices[i + 1]
        sliders.append([before, slider, after])

    return np.array(sliders, dtype=np.intp)


def combine_landmarks_and_curves(
    landmarks,
    curves,
    curve_landmarks=None,
    anchor_first=False,
    anchor_last=False,
):
    """Combine TPS landmarks and curves into unified GPA format.

    Converts the separate landmarks and curves format from `read_tps()`
    into the combined format required by `GeneralizedProcrustesAnalysis`.

    Parameters
    ----------
    landmarks : ndarray, shape (n_specimens, n_landmarks, n_dim)
        Fixed landmarks from TPS file.
    curves : list of list of ndarray
        Curve semilandmarks from TPS file.
        curves[specimen_idx][curve_idx] = ndarray (n_points, n_dim)
    curve_landmarks : list of tuple of (int, int), optional
        Landmark indices that anchor each curve endpoint.
        Each tuple ``(start_lm, end_lm)`` specifies the landmark
        indices (in the landmarks array) that the first and last
        semilandmarks of that curve connect to. When provided, all
        semilandmarks slide and the anchoring landmarks serve as
        the before/after neighbors for the curve endpoints.
    anchor_first : bool, default=False
        If True, first point of each curve is treated as a fixed anchor
        (does not slide). Ignored when ``curve_landmarks`` is provided.
    anchor_last : bool, default=False
        If True, last point of each curve is treated as a fixed anchor
        (does not slide). Ignored when ``curve_landmarks`` is provided.

    Returns
    -------
    combined : ndarray, shape (n_specimens, n_total_points, n_dim)
        All coordinates concatenated: landmarks followed by curve points.
    slider_matrix : ndarray, shape (n_sliders, 3)
        Topology matrix [before, slider, after] for GPA.
    curve_info : dict
        Information about the combined structure:
        - 'n_landmarks': number of original fixed landmarks
        - 'n_curve_points': total number of curve points added
        - 'curve_offsets': starting index of each curve in combined array
        - 'curve_lengths': number of points in each curve

    """
    landmarks = np.asarray(landmarks)
    n_specimens = len(landmarks)
    n_landmarks = landmarks.shape[1]

    # Validate curves structure
    if len(curves) != n_specimens:
        raise ValueError(
            f"Number of specimens in landmarks ({n_specimens}) and curves "
            f"({len(curves)}) must match."
        )

    # Validate number of curves and points
    n_curves = len(curves[0])
    curve_lengths = [len(c) for c in curves[0]]
    for i, specimen_curves in enumerate(curves):
        if len(specimen_curves) != n_curves:
            raise ValueError(
                f"Specimen {i} has {len(specimen_curves)} curves, "
                f"but specimen 0 has {n_curves}."
            )
        for j, curve in enumerate(specimen_curves):
            if len(curve) != curve_lengths[j]:
                raise ValueError(
                    f"Curve {j} in specimen {i} has {len(curve)} points, "
                    f"but specimen 0 has {curve_lengths[j]}."
                )

    if curve_landmarks is not None and len(curve_landmarks) != n_curves:
        raise ValueError(
            f"curve_landmarks has {len(curve_landmarks)} entries, "
            f"but there are {n_curves} curves."
        )

    # Build combined array and slider matrix
    slider_rows = []
    current_offset = n_landmarks

    curve_offsets = []
    for curve_idx, curve_len in enumerate(curve_lengths):
        curve_offsets.append(current_offset)

        if curve_landmarks is not None:
            # All semilandmarks slide; landmarks serve as anchors
            start_lm, end_lm = curve_landmarks[curve_idx]
            for i in range(curve_len):
                slider = current_offset + i
                if i == 0:
                    before = start_lm
                else:
                    before = current_offset + i - 1
                if i == curve_len - 1:
                    after = end_lm
                else:
                    after = current_offset + i + 1
                slider_rows.append([before, slider, after])
        else:
            # Use anchor_first / anchor_last to determine which points slide
            start_idx = 1 if anchor_first else 0
            end_idx = curve_len - 1 if anchor_last else curve_len

            for i in range(start_idx, end_idx):
                before = current_offset + i - 1
                slider = current_offset + i
                after = current_offset + i + 1

                if i == 0:
                    before = (
                        current_offset + curve_len - 1 if not anchor_first else before
                    )
                if i == curve_len - 1:
                    after = current_offset if not anchor_last else after

                slider_rows.append([before, slider, after])

        current_offset += curve_len

    # Combine landmarks and curve points for each specimen
    combined_list = []
    for spec_idx in range(n_specimens):
        spec_points = [landmarks[spec_idx]]
        for curve in curves[spec_idx]:
            spec_points.append(np.asarray(curve))
        combined_list.append(np.vstack(spec_points))

    combined = np.array(combined_list)
    slider_matrix = np.array(slider_rows, dtype=np.intp)

    total_curve_points = sum(curve_lengths)
    curve_info = {
        "n_landmarks": n_landmarks,
        "n_curve_points": total_curve_points,
        "curve_offsets": curve_offsets,
        "curve_lengths": curve_lengths,
    }

    return combined, slider_matrix, curve_info


def centroid_size(x):
    """Calculate the centroid size.

    Parameters
    ----------
    x : array-like, shape (n_landmarks, n_dim)
        Configuration, pre-shape, shape, etc.

    Returns
    -------
    centroid_size : float
        Centroid size of the input.
    """
    x = np.array(x)
    x_c = x - np.mean(x, axis=0)
    centroid_size = np.sqrt(np.trace(np.dot(x_c, x_c.T)))
    return centroid_size
