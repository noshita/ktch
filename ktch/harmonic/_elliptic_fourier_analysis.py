"""Elliptic Fourier Analysis"""

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

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
)
from sklearn.utils.parallel import Parallel, delayed

# Tolerance for detecting degenerate (near-zero) geometric quantities
# (arc length, semi-axes, phase-angle denominator).
_DEGENERACY_TOL = 1e-15

# Floor value for near-zero arc-length segments when
# duplicated_points="infinitesimal".
_INFINITESIMAL_DT = 1e-10

# Tolerance for gimbal-lock detection in ZXZ Euler angle extraction.
_GIMBAL_TOL = 1e-10


class EllipticFourierAnalysis(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
):
    r"""
    Elliptic Fourier Analysis (EFA)

    Parameters
    ----------
    n_harmonics: int, default=20
        Number of harmonics
    n_dim: int, default=2
        Dimension of the coordinate space.
        Must be 2 (for planar curves) or 3 (for space curves).
    norm : bool, default=True
        Normalize the elliptic Fourier coefficients
        by the major axis of the 1st ellipse.
    return_orientation_scale : bool, default=False
        Return orientation and scale of the outline (requires ``norm=True``).

        - 2D: Appends ``[psi, scale]`` to the end of the coefficient vector.
        - 3D: Appends ``[alpha, beta, gamma, phi, scale]`` to the end.
    n_jobs: int, default=None
        The number of jobs to run in parallel. None means 1 unless in a
        joblib.parallel_backend context. -1 means using all processors.
    verbose: int, default=0
        The verbosity level.
    norm_method : str, default="area"
        Normalization method for 3D EFA coefficients when ``norm=True``.
        Only affects ``n_dim=3``.

        - ``"area"``: Scale by ``sqrt(pi * a1 * b1)`` where ``a1`` and ``b1``
          are the semi-major and semi-minor axis lengths of the 1st harmonic
          ellipse (Godefroy et al. 2012).
        - ``"semi_major_axis"``: Scale by the semi-major axis length ``a1``
          of the 1st harmonic ellipse, consistent with the 2D normalization
          convention (Kuhl & Giardina 1982).

    Notes
    -----
    EFA is widely applied for outline shape analysis
    in two-dimensional space [Kuhl_Giardina_1982]_.

    .. math::
        \begin{align}
            x(l) &=
            \frac{a_0}{2} + \sum_{i=1}^{n}
            \left[ a_i \cos\left(\frac{2\pi i t}{T}\right)
            + b_i \sin\left(\frac{2\pi i t}{T}\right) \right]\\
            y(l) &=
            \frac{c_0}{2} + \sum_{i=1}^{n}
            \left[ c_i \cos\left(\frac{2\pi i t}{T}\right)
            + d_i \sin\left(\frac{2\pi i t}{T}\right) \right]\\
        \end{align}


    EFA is also applied for a closed curve in the three-dimensional space
    (e.g., [Lestrel_1997]_, [Lestrel_et_al_1997]_, and [Godefroy_et_al_2012]_).

    For 3D data (``n_dim=3``), normalization (``norm=True``) follows
    Godefroy et al. (2012) §3.1: rescaling by the 1st harmonic ellipse area,
    reorientation using ZXZ Euler angles, phase shift, and direction correction.

    When ``return_orientation_scale=True`` with 3D normalized data, 5 values
    are appended to the output: ``[alpha, beta, gamma, phi, scale]``, where
    ``(alpha, beta, gamma)`` are ZXZ Euler angles (in radians) of the 1st
    harmonic ellipse orientation, ``phi`` is the phase angle, and ``scale``
    is the normalization factor (``sqrt(pi * a1 * b1)`` for
    ``norm_method="area"``, or ``a1`` for ``norm_method="semi_major_axis"``).

    References
    ----------
    .. [Kuhl_Giardina_1982] Kuhl, F.P., Giardina, C.R. (1982) Elliptic Fourier features of a closed contour. Comput. Graph. Image Process. 18: 236–258. https://doi.org/10.1016/0146-664X(82)90034-X
    .. [Lestrel_1997]  Lestrel, P.E., 1997. Introduction and overview of Fourier descriptors, in: Fourier Descriptors and Their Applications in Biology. Cambridge University Press, pp. 22–44. https://doi.org/10.1017/cbo9780511529870.003
    .. [Lestrel_et_al_1997] Lestrel, P.E., Read, D.W., Wolfe, C., 1997. Size and shape of the rabbit orbit: 3-D Fourier descriptors, in: Lestrel, P.E. (Ed.), Fourier Descriptors and Their Applications in Biology. Cambridge University Press, pp. 359–378. https://doi.org/10.1017/cbo9780511529870.017
    .. [Godefroy_et_al_2012] Godefroy, J.E., Bornert, F., Gros, C.I., Constantinesco, A., 2012. Elliptical Fourier descriptors for contours in three dimensions: A new tool for morphometrical analysis in biology. C. R. Biol. 335, 205–213. https://doi.org/10.1016/j.crvi.2011.12.004

    """

    _VALID_NORM_METHODS = {"area", "semi_major_axis"}

    def __init__(
        self,
        n_harmonics: int = 20,
        n_dim: int = 2,
        norm: bool = True,
        return_orientation_scale: bool = False,
        n_jobs: int | None = None,
        verbose: int = 0,
        norm_method: str = "area",
    ):
        self.n_harmonics = n_harmonics
        self.n_dim = n_dim
        self.norm = norm
        self.return_orientation_scale = return_orientation_scale
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.norm_method = norm_method

    def fit(self, X, y=None):
        """Fit the model (no-op for stateless transformer).

        Parameters
        ----------
        X : ignored
        y : ignored

        Returns
        -------
        self
        """
        return self

    def fit_transform(self, X, y=None, t=None):
        """Fit and transform in a single step.

        Overridden to support metadata routing of ``t``.

        Parameters
        ----------
        X : list of array-like of shape (n_coords_i, n_dim)
            Coordinate values of n_samples.
        y : ignored
        t : list of array-like, optional
            Per-sample parameterization. Passed to ``transform``.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features_out)
        """
        return self.fit(X, y).transform(X, t=t)

    def transform(
        self,
        X: list[npt.ArrayLike] | npt.ArrayLike,
        t: npt.ArrayLike = None,
    ) -> npt.ArrayLike:
        """Elliptic Fourier Analysis.

        Parameters
        ----------
        X : {list of array-like, array-like} of shape (n_samples, n_coords, n_dim)
            Coordinate values of n_samples.
            The i-th array-like of shape (n_coords_i, n_dim) represents
            coordinate values of the i-th sample.

        t : list of array-like of shape (n_coords_i,), optional
            Parameters indicating the position on the outline of n_samples.
            The i-th element corresponds to each coordinate value in the
            i-th element of X. If ``None``, arc-length parameterization
            is computed automatically.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features_out)
            Elliptic Fourier coefficients.

            - 2D (return_orientation_scale=False):
              ``[a_0..a_n, b_0..b_n, c_0..c_n, d_0..d_n]``
              length = ``4 * (n_harmonics + 1)``.
            - 2D (return_orientation_scale=True):
              Same as above with ``[psi, scale]`` appended (length +2).
            - 3D (return_orientation_scale=False):
              ``[a_0..a_n, b_0..b_n, c_0..c_n, d_0..d_n, e_0..e_n, f_0..f_n]``
              length = ``6 * (n_harmonics + 1)``.
            - 3D (return_orientation_scale=True):
              Same as above with ``[alpha, beta, gamma, phi, scale]`` appended (length +5).
        """
        n_dim = self.n_dim
        norm = self.norm
        return_orientation_scale = self.return_orientation_scale

        if n_dim not in (2, 3):
            raise ValueError("n_dim must be 2 or 3")
        if self.norm_method not in self._VALID_NORM_METHODS:
            raise ValueError(
                f"norm_method must be 'area' or 'semi_major_axis', got '{self.norm_method}'"
            )

        if return_orientation_scale and not norm:
            raise ValueError("return_orientation_scale requires norm=True.")

        if t is None:
            t_ = [None] * len(X)
        else:
            t_ = t

        if len(t_) != len(X):
            raise ValueError(
                f"t ({len(t_)}) must have the same length as X ({len(X)})"
            )

        if isinstance(X, pd.DataFrame):
            X_ = [
                row.dropna().to_numpy().reshape(n_dim, -1).T
                for idx, row in X.iterrows()
            ]
        else:
            X_ = X

        if n_dim == 2:
            X_transformed = np.stack(
                Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                    delayed(self._transform_single_2d)(X_[i], t_[i])
                    for i in range(len(X_))
                )
            )
        elif n_dim == 3:
            X_transformed = np.stack(
                Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                    delayed(self._transform_single_3d)(X_[i], t_[i])
                    for i in range(len(X_))
                )
            )

        return X_transformed

    def inverse_transform(self, X_transformed, t_num=100, as_frame=False):
        """Inverse analysis of elliptic Fourier analysis.

        Parameters
        ----------
        X_transformed : array-like of shape (n_samples, n_features)
            Elliptic Fourier coefficients. Accepted lengths per sample:
              - 2D: ``4*(n_harmonics+1)`` or ``4*(n_harmonics+1)+2`` (psi, scale).
              - 3D: ``6*(n_harmonics+1)`` or ``6*(n_harmonics+1)+5`` (alpha, beta, gamma, phi, scale).
            Orientation/scale columns, if present, are ignored for reconstruction.
        t_num : int, default = 100
            Number of coordinate values.
        as_frame : bool, default = False
            If True, return pd.DataFrame.

        Returns
        -------
        X_coords : array-like of shape (n_samples, t_num, n_dim) or pd.DataFrame
            Coordinate values reconstructed from the elliptic Fourier coefficients.

        """
        X_list = []
        sp_num = X_transformed.shape[0]
        col_names = ["x", "y", "z"][: self.n_dim]

        for i in range(sp_num):
            if as_frame:
                coef = X_transformed.loc[i]
                X = self._inverse_transform_single(coef, t_num=t_num)
                df_X = pd.DataFrame(X, columns=col_names)
                df_X["coord_id"] = list(range(len(X)))
                df_X["specimen_id"] = i
                X_list.append(df_X)
            else:
                coef = X_transformed[i]
                X = self._inverse_transform_single(coef, t_num=t_num)
                X_list.append(X)

        if as_frame:
            X_coords = pd.concat(X_list)
            X_coords = X_coords.set_index(["specimen_id", "coord_id"])
        else:
            X_coords = np.stack(X_list)

        return X_coords

    ###########################################################
    #
    #   2D
    #
    ###########################################################

    def _transform_single_2d(
        self,
        X: np.ndarray,
        t: np.ndarray | None = None,
        duplicated_points: str = "infinitesimal",
    ):
        """Fit the model with a single outline.

        Parameters
        ----------
        X: ndarray of shape (n_coords, 2)
            Coordinate values of an 2D outline.

        t: ndarray of shape  (n_coords, ), optional
            A parameter indicating the position on the outline.
            If `t=None`, then t is calculated based on
            the coordinate values with the linear interpolation.

        Returns
        -------
        X_transformed: ndarray of shape (4*(n_harmonics+1), )
            Coefficients of Fourier series.

        """
        n_harmonics = self.n_harmonics

        X_arr, diffs, dt = _preprocess_outline(X, t, duplicated_points)
        dx, dy = diffs[:, 0], diffs[:, 1]

        # Fourier series expansion
        T = np.sum(dt)
        a0 = 2 * np.sum(X_arr[1:, 0] * dt) / T
        c0 = 2 * np.sum(X_arr[1:, 1] * dt) / T
        an = np.append(a0, _cse(dx, dt, n_harmonics))
        bn = np.append(0, _sse(dx, dt, n_harmonics))
        cn = np.append(c0, _cse(dy, dt, n_harmonics))
        dn = np.append(0, _sse(dy, dt, n_harmonics))

        # Normalize
        if self.norm:
            an, bn, cn, dn, psi, scale = self._normalize_2d(an, bn, cn, dn)

        if self.return_orientation_scale:
            X_transformed = np.hstack([an, bn, cn, dn, psi, scale])
        else:
            X_transformed = np.hstack([an, bn, cn, dn])

        return X_transformed

    def _normalize_2d(self, an, bn, cn, dn, keep_start_point=False):
        """Normalize Fourier coefficients.

        Todo:
            - [x] 1st ellipse, major axis
            - [ ] 1st ellipse, area
            - [ ] Procrustes alignment -> in coordinate values?

        Returns
        -------
        An, Bn, Cn, Dn : np.ndarray
            Normalized coefficient arrays (offset + harmonics).
        psi : float
            Orientation (phase) of the 1st harmonic ellipse in radians.
        scale : float
            Semi-major axis length of the 1st harmonic ellipse.
        """
        a1 = an[1]
        b1 = bn[1]
        c1 = cn[1]
        d1 = dn[1]

        theta = (1 / 2) * np.arctan(
            2 * (a1 * b1 + c1 * d1) / (a1**2 + c1**2 - b1**2 - d1**2)
        )

        [[a_s, b_s], [c_s, d_s]] = np.array([[a1, b1], [c1, d1]]).dot(
            rotation_matrix_2d(theta)
        )
        s1 = a_s**2 + c_s**2
        s2 = b_s**2 + d_s**2

        if s1 < s2:
            if theta < 0:
                theta = theta + np.pi / 2
            else:
                theta = theta - np.pi / 2

        a_s = a1 * np.cos(theta) + b1 * np.sin(theta)
        c_s = c1 * np.cos(theta) + d1 * np.sin(theta)
        scale = np.sqrt(a_s**2 + c_s**2)
        psi = np.arctan2(c_s, a_s)

        if keep_start_point:
            theta = 0

        coef_norm_list = []
        r_psi = rotation_matrix_2d(-psi)
        for n in range(1, len(an)):
            r_ntheta = rotation_matrix_2d(n * theta)
            coef_orig = np.array([[an[n], bn[n]], [cn[n], dn[n]]])
            coef_norm_tmp = (1 / scale) * np.dot(np.dot(r_psi, coef_orig), r_ntheta)
            coef_norm_list.append(coef_norm_tmp.reshape(-1))

        coef_norm = np.stack(coef_norm_list)
        An = np.append(an[0], coef_norm[:, 0])
        Bn = np.append(bn[0], coef_norm[:, 1])
        Cn = np.append(cn[0], coef_norm[:, 2])
        Dn = np.append(dn[0], coef_norm[:, 3])

        return An, Bn, Cn, Dn, psi, scale

    def _inverse_transform_single(self, X_transformed, t_num=100):
        coef_array = np.asarray(X_transformed, dtype=float)
        n_axes = 2 * self.n_dim
        n_extras = {2: 2, 3: 5}[self.n_dim]
        expected_base = n_axes * (self.n_harmonics + 1)

        if coef_array.shape[0] == expected_base + n_extras:
            coef_core = coef_array[:expected_base]
        elif coef_array.shape[0] == expected_base:
            coef_core = coef_array
        else:
            raise ValueError(
                f"Expected {expected_base} or {expected_base + n_extras} "
                f"coefficients, got {coef_array.shape[0]}."
            )

        # Reshape to (n_axes, n_harmonics+1).
        # Axes are ordered [cos0, sin0, cos1, sin1, ...] per coordinate.
        axes = coef_core.reshape([n_axes, -1])

        # Offsets sit at index 0 of the cos rows.
        offsets = axes[::2, 0].copy()
        if self.norm:
            offsets[:] = 0.0

        # (n_dim, n_harmonics)
        cos_coefs = axes[::2, 1:]
        sin_coefs = axes[1::2, 1:]

        n_max = cos_coefs.shape[1]
        theta = np.linspace(2 * np.pi / t_num, 2 * np.pi, t_num)
        ns = np.arange(1, n_max + 1)
        # (n_max, t_num)
        cos_basis = np.cos(np.outer(ns, theta))
        sin_basis = np.sin(np.outer(ns, theta))

        # Reconstruct coordinates: (n_dim, t_num)
        coords = offsets[:, None] / 2 + cos_coefs @ cos_basis + sin_coefs @ sin_basis

        return coords.T

    ###########################################################
    #
    #   3D
    #
    ###########################################################

    def _transform_single_3d(
        self,
        X: np.ndarray,
        t: np.ndarray | None = None,
        duplicated_points: str = "infinitesimal",
    ):
        """Fit the model with a single 3D outline.

        Parameters
        ----------
        X : ndarray of shape (n_coords, 3)
            Coordinate values of a 3D outline.

        t : ndarray of shape (n_coords,), optional
            A parameter indicating the position on the outline.
            If ``None``, arc-length parameterization is computed automatically.

        Returns
        -------
        X_transformed : ndarray of shape (6*(n_harmonics+1),) or (6*(n_harmonics+1)+5,)
            Coefficients of Fourier series.
        """
        n_harmonics = self.n_harmonics

        X_arr, diffs, dt = _preprocess_outline(X, t, duplicated_points)
        dx, dy, dz = diffs[:, 0], diffs[:, 1], diffs[:, 2]

        # Fourier series expansion
        T = np.sum(dt)
        a0 = 2 * np.sum(X_arr[1:, 0] * dt) / T
        c0 = 2 * np.sum(X_arr[1:, 1] * dt) / T
        e0 = 2 * np.sum(X_arr[1:, 2] * dt) / T
        an = np.append(a0, _cse(dx, dt, n_harmonics))
        bn = np.append(0, _sse(dx, dt, n_harmonics))
        cn = np.append(c0, _cse(dy, dt, n_harmonics))
        dn = np.append(0, _sse(dy, dt, n_harmonics))
        en = np.append(e0, _cse(dz, dt, n_harmonics))
        fn = np.append(0, _sse(dz, dt, n_harmonics))

        # Normalize
        if self.norm:
            an, bn, cn, dn, en, fn, alpha, beta, gamma, phi, scale = self._normalize_3d(
                an, bn, cn, dn, en, fn
            )

        if self.return_orientation_scale:
            X_transformed = np.hstack(
                [an, bn, cn, dn, en, fn, alpha, beta, gamma, phi, scale]
            )
        else:
            X_transformed = np.hstack([an, bn, cn, dn, en, fn])

        return X_transformed

    def _normalize_3d(self, an, bn, cn, dn, en, fn):
        """Normalize 3D EFA coefficients.

        Applies the 4-step normalization algorithm:
        1. Rescaling by a scale factor determined by ``self.norm_method``:

           - ``"area"``: ``scale = sqrt(pi * a1 * b1)``
           - ``"semi_major_axis"``: ``scale = a1``

        2. Reorientation using the 1st harmonic's Euler angles
        3. Phase shift using the 1st harmonic's phase angle
        4. Direction correction (sign of y-sine component)

        Parameters
        ----------
        an, bn, cn, dn, en, fn : np.ndarray of shape (n_harmonics+1,)
            Raw Fourier coefficient arrays. Index 0 is the offset.

        Returns
        -------
        An, Bn, Cn, Dn, En, Fn : np.ndarray of shape (n_harmonics+1,)
            Normalized coefficient arrays.
        alpha, beta, gamma : float
            ZXZ Euler angles of the 1st harmonic ellipse.
        phi : float
            Phase angle of the 1st harmonic ellipse.
        scale : float
            Scaling factor. ``sqrt(pi * a1 * b1)`` when ``norm_method="area"``,
            or ``a1`` when ``norm_method="semi_major_axis"``.

        Notes
        -----
        When ``return_orientation_scale=True`` in 3D, these five values are
        appended to the transform output in the order:
        ``[alpha, beta, gamma, phi, scale]``.
        """
        # Extract geometric parameters of the 1st harmonic
        phi1, a1, b1, alpha1, beta1, gamma1 = _compute_ellipse_geometry_3d(
            an[1], bn[1], cn[1], dn[1], en[1], fn[1]
        )

        # Handle degenerate 1st harmonic
        if a1 < _DEGENERACY_TOL:
            raise ValueError(
                "Degenerate 1st harmonic: the ellipse has near-zero semi-axes. "
                "Cannot normalize 3D EFA coefficients."
            )

        # 1. Rescaling
        if self.norm_method == "semi_major_axis":
            scale = a1
        else:
            # Default: area-based (Godefroy et al. 2012)
            area1 = np.pi * a1 * b1
            scale = np.sqrt(area1)

        # 2. Reorientation matrix (Omega1_inv = Omega1^T)
        Omega1 = rotation_matrix_3d_euler_zxz(alpha1, beta1, gamma1)
        Omega1_inv = Omega1.T

        n_harmonics = len(an) - 1
        An = np.empty_like(an)
        Bn = np.empty_like(bn)
        Cn = np.empty_like(cn)
        Dn = np.empty_like(dn)
        En = np.empty_like(en)
        Fn = np.empty_like(fn)

        An[0] = an[0]
        Bn[0] = bn[0]
        Cn[0] = cn[0]
        Dn[0] = dn[0]
        En[0] = en[0]
        Fn[0] = fn[0]

        for k in range(1, n_harmonics + 1):
            # Build 3x2 coefficient matrix
            # C_k = [[an_k, bn_k], [cn_k, dn_k], [en_k, fn_k]]
            C_k = np.array(
                [
                    [an[k], bn[k]],
                    [cn[k], dn[k]],
                    [en[k], fn[k]],
                ]
            )

            # 3. Phase rotation uses k*phi1 for harmonic k
            # Removing phase phi1 means substituting t -> t + phi1:
            #   new_xc = xc*cos(k*phi1) + xs*sin(k*phi1)
            #   new_xs = -xc*sin(k*phi1) + xs*cos(k*phi1)
            # In matrix form: C_k @ R(-k*phi1) where R is the standard rotation matrix
            angle_k = k * phi1
            cos_k = np.cos(angle_k)
            sin_k = np.sin(angle_k)
            R_phase_k = np.array(
                [
                    [cos_k, sin_k],
                    [-sin_k, cos_k],
                ]
            )

            # Apply: C'_k = (1/scale) * Omega1_inv @ C_k @ R_phase_k
            C_norm = (1.0 / scale) * Omega1_inv @ C_k @ R_phase_k

            An[k] = C_norm[0, 0]
            Bn[k] = C_norm[0, 1]
            Cn[k] = C_norm[1, 0]
            Dn[k] = C_norm[1, 1]
            En[k] = C_norm[2, 0]
            Fn[k] = C_norm[2, 1]

        # 4. Direction correction
        # If the y-sine coefficient of the 1st harmonic is negative,
        # negate all sine columns
        if Dn[1] < 0:
            Bn = -Bn
            Dn = -Dn
            Fn = -Fn

        return An, Bn, Cn, Dn, En, Fn, alpha1, beta1, gamma1, phi1, scale

    ###########################################################
    #
    #   set_output API
    #
    ###########################################################

    def __sklearn_is_fitted__(self):
        """Return True since this is a stateless transformer."""
        return True

    def get_feature_names_out(
        self, input_features: None | npt.ArrayLike = None
    ) -> np.ndarray:
        """Get output feature names.

        Parameters
        ----------
        input_features : ignored

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.

        """
        include_orientation = self.return_orientation_scale and self.norm
        return np.asarray(self._build_feature_names(include_orientation), dtype=str)

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        base = (self.n_harmonics + 1) * (2 * self.n_dim)
        if self.return_orientation_scale and self.norm:
            if self.n_dim == 3:
                return base + 5
            if self.n_dim == 2:
                return base + 2
        return base

    def _build_feature_names(self, include_orientation: bool) -> list[str]:
        an = [f"a_{i}" for i in range(self.n_harmonics + 1)]
        bn = [f"b_{i}" for i in range(self.n_harmonics + 1)]
        cn = [f"c_{i}" for i in range(self.n_harmonics + 1)]
        dn = [f"d_{i}" for i in range(self.n_harmonics + 1)]
        feature_names = an + bn + cn + dn
        if self.n_dim == 3:
            en = [f"e_{i}" for i in range(self.n_harmonics + 1)]
            fn = [f"f_{i}" for i in range(self.n_harmonics + 1)]
            feature_names = feature_names + en + fn
        if include_orientation:
            if self.n_dim == 3:
                feature_names += ["alpha", "beta", "gamma", "phi", "scale"]
            elif self.n_dim == 2:
                feature_names += ["psi", "scale"]
        return feature_names


###########################################################
#
#   utility functions
#
###########################################################


def _preprocess_outline(X, t, duplicated_points="infinitesimal"):
    """Prepare an outline for EFA.

    Wraps the contour (prepends last point), computes coordinate differences
    and arc-length parameterization, validates inputs, and handles duplicated
    (zero-length) segments.

    Parameters
    ----------
    X : ndarray of shape (n_coords, n_dim)
        Coordinate values of an outline.
    t : ndarray of shape (n_coords,) or None
        Positional parameter.
        If None, arc-length parameterization is used.
    duplicated_points : str
        Strategy for zero-length segments:
        ``"infinitesimal"`` (default) or ``"deletion"``.

    Returns
    -------
    X_arr : ndarray of shape (n_coords + 1, n_dim)
        Wrapped coordinate array (last point prepended).
    diffs : ndarray of shape (m, n_dim)
        Per-axis coordinate differences (m <= n_coords after deletion).
    dt : ndarray of shape (m,)
        Parameter increments.
    """
    X_arr = np.vstack([X[-1:], np.asarray(X)])

    if not np.all(np.isfinite(X_arr)):
        raise ValueError("Input coordinates must not contain NaN or Inf values.")

    diffs = X_arr[1:] - X_arr[:-1]

    if t is None:
        dt = np.linalg.norm(diffs, axis=1)
    else:
        t_ = np.append(0, t)
        dt = t_[1:] - t_[:-1]

    tp = np.cumsum(dt)

    if len(tp) != len(X):
        raise ValueError(
            "len(t) must have a same len(X), len(t): "
            + str(len(tp))
            + ", len(X): "
            + str(len(X))
        )

    if tp[-1] < _DEGENERACY_TOL:
        raise ValueError(
            "Degenerate outline: total arc length is near zero. "
            "All points may be identical."
        )

    if duplicated_points == "infinitesimal":
        dt[dt < _INFINITESIMAL_DT] = _INFINITESIMAL_DT
    elif duplicated_points == "deletion":
        idx_duplicated_points = np.where(dt == 0)[0]
        if len(idx_duplicated_points) > 0:
            diffs = np.delete(diffs, idx_duplicated_points, axis=0)
            dt = np.delete(dt, idx_duplicated_points)
            X_arr = np.delete(X_arr, idx_duplicated_points, 0)
    else:
        raise ValueError("'duplicated_points' must be 'infinitesimal' or 'deletion'")

    return X_arr, diffs, dt


def rotation_matrix_2d(theta):
    rot_mat = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return rot_mat


def rotation_matrix_3d_euler_zxz(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Construct 3x3 rotation matrix from ZXZ Euler angles.

    The rotation is composed as Omega = R_gamma @ R_beta @ R_alpha,
    following the convention in Godefroy et al. (2012) Fig. 1.

    Parameters
    ----------
    alpha, beta, gamma : float
        ZXZ Euler angles in radians.

    Returns
    -------
    rotation_matrix : np.ndarray of shape (3, 3)
        Orthogonal rotation matrix with determinant +1.
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)

    return np.array(
        [
            [ca * cg - sa * cb * sg, -ca * sg - sa * cb * cg, sa * sb],
            [sa * cg + ca * cb * sg, -sa * sg + ca * cb * cg, -ca * sb],
            [sb * sg, sb * cg, cb],
        ]
    )


def _compute_ellipse_geometry_3d(
    xc: float,
    xs: float,
    yc: float,
    ys: float,
    zc: float,
    zs: float,
) -> tuple[float, float, float, float, float, float]:
    """Compute geometric parameters of a 3D ellipse from Fourier coefficients.

    Returns the unique solution satisfying:
    phi in ]-pi/4, pi/4[, a >= b > 0, beta in [0, pi].

    Parameters
    ----------
    xc, xs, yc, ys, zc, zs : float
        Cosine and sine Fourier coefficients for x, y, z coordinates.

    Returns
    -------
    phi : float
        Phase angle in ]-pi/4, pi/4[.
    a : float
        Semi-major axis length (a > 0).
    b : float
        Semi-minor axis length (b > 0).
    alpha : float
        First Euler angle (ZXZ convention).
    beta : float
        Second Euler angle, beta in [0, pi].
    gamma : float
        Third Euler angle (ZXZ convention).
    """
    sum_c2 = xc**2 + yc**2 + zc**2
    sum_s2 = xs**2 + ys**2 + zs**2
    dot_cs = xc * xs + yc * ys + zc * zs

    # phase angle phi
    # 2*dot_cs / denom = -tan(2*phi)
    # -> phi_0 = -(1/2) * arctan2(2*dot_cs, denom)
    denom = sum_c2 - sum_s2
    if abs(denom) < _DEGENERACY_TOL and abs(dot_cs) < _DEGENERACY_TOL:
        phi_0 = 0.0
    else:
        phi_0 = -0.5 * np.arctan2(2 * dot_cs, denom)

    cos_phi = np.cos(phi_0)
    sin_phi = np.sin(phi_0)
    sin_2phi = np.sin(2 * phi_0)

    a2 = sum_c2 * cos_phi**2 + sum_s2 * sin_phi**2 - dot_cs * sin_2phi
    b2 = sum_c2 * sin_phi**2 + sum_s2 * cos_phi**2 + dot_cs * sin_2phi

    # Enforce a >= b; if not, shift phi by pi/2
    if a2 < b2:
        a2, b2 = b2, a2
        phi_0 = phi_0 + np.pi / 2 if phi_0 < 0 else phi_0 - np.pi / 2

    # Normalize phi to ]-pi/4, pi/4[
    phi = phi_0
    if phi > np.pi / 4:
        phi -= np.pi / 2
        a2, b2 = b2, a2
    elif phi <= -np.pi / 4:
        phi += np.pi / 2
        a2, b2 = b2, a2

    a = np.sqrt(max(a2, 0.0))
    b = np.sqrt(max(b2, 0.0))

    if a < _DEGENERACY_TOL:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Recover rotation matrix Omega columns from coefficient vectors.
    # The parametric equation relates coefficients to Omega and local-frame values:
    #   [xc, yc, zc]^T = Omega @ [a*cos(phi), b*sin(phi), 0]^T
    #   [xs, ys, zs]^T = Omega @ [-a*sin(phi), b*cos(phi), 0]^T
    # Solving for the first two columns of Omega:
    #   col0 = (cos(phi)*[xc,yc,zc] - sin(phi)*[xs,ys,zs]) / a
    #   col1 = (sin(phi)*[xc,yc,zc] + cos(phi)*[xs,ys,zs]) / b

    coef_c = np.array([xc, yc, zc])
    coef_s = np.array([xs, ys, zs])

    col0 = (cos_phi * coef_c - sin_phi * coef_s) / a

    if b < _DEGENERACY_TOL:
        col1 = np.zeros(3)
        col2 = np.zeros(3)
    else:
        col1 = (sin_phi * coef_c + cos_phi * coef_s) / b
        col2 = np.cross(col0, col1)

    # Rotation matrix entries
    Omega_11, Omega_21, Omega_31 = col0
    Omega_12, Omega_22, Omega_32 = col1
    Omega_13, Omega_23, Omega_33 = col2

    # Extract Euler angles (ZXZ) from the rotation matrix
    cos_beta = np.clip(Omega_33, -1.0, 1.0)
    beta = float(np.arccos(cos_beta))

    if abs(np.sin(beta)) < _GIMBAL_TOL:
        # Gimbal lock
        gamma = 0.0
        alpha = float(np.arctan2(Omega_21, Omega_11))
    else:
        sin_beta = np.sin(beta)
        gamma = float(np.arctan2(Omega_31 / sin_beta, Omega_32 / sin_beta))
        alpha = float(np.arctan2(Omega_13 / sin_beta, -Omega_23 / sin_beta))

    return float(phi), float(a), float(b), alpha, beta, float(gamma)


def _cse(dx: np.ndarray, dt: np.ndarray, n_harmonics: int) -> np.ndarray:
    """Cos series expansion n>=1

    Parameters
    ----------
    dx : np.ndarray
        differences of coordinates
    dt : np.ndarray
        differences of parameter
    n_harmonics : int
        number of harmonics

    Returns
    -------
    coef : np.ndarray
        coefficients of cos series expansion
    """
    t = np.concatenate([[0], np.cumsum(dt)])
    T = t[-1]

    cn = [
        (T / (2 * (np.pi**2) * (n**2)))
        * np.sum(
            (dx / dt)
            * (np.cos(2 * np.pi * n * t[1:] / T) - np.cos(2 * np.pi * n * t[:-1] / T))
        )
        for n in range(1, n_harmonics + 1, 1)
    ]

    coef = np.array(cn)

    return coef


def _sse(dx: np.ndarray, dt: np.ndarray, n_harmonics: int) -> np.ndarray:
    """Sin series expansion n>=1"""
    t = np.concatenate([[0], np.cumsum(dt)])
    T = t[-1]

    cn = [
        (T / (2 * (np.pi**2) * (n**2)))
        * np.sum(
            (dx / dt)
            * (np.sin(2 * np.pi * n * t[1:] / T) - np.sin(2 * np.pi * n * t[:-1] / T))
        )
        for n in range(1, n_harmonics + 1, 1)
    ]

    coef = np.array(cn)

    return coef
