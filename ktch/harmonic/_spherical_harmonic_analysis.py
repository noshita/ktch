"""Spherical Harmonic (SPHARM) Analysis"""

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

import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy as sp
from scipy.spatial.transform import Rotation
from scipy.special import factorial
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
)
from sklearn.utils.parallel import Parallel, delayed

from ._registration import moment_register, validate_registration

# Tolerance for detecting pole singularity in xyz2spherical.
_POLE_TOL = 1e-12

# Tolerance for a degenerate first-order ellipsoid (near-zero semi-major axis).
_FIRST_ORDER_TOL = 1e-12

# Tolerance below which a principal-axis skewness is treated as zero.
_SKEW_TOL = 1e-9


class SphericalHarmonicAnalysis(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
):
    r"""Spherical Harmonic (SPHARM) Analysis

    Parameters
    ----------
    n_harmonics: int, default=10
        Number of harmonics to use ($l_\mathrm{max}$).
    n_dim: int, default=3
        Dimension of the codomain, i.e. the number of components
        of the :math:`\mathbb{R}^D`-valued function expanded on the sphere.
        Any positive integer is supported; ``3`` is the common surface
        mapping and ``1`` corresponds to a scalar field on the sphere.
    registration : {"auto", None, "first_order", "moment"}, default="auto"
        Shape-registration method (2D/3D shape data only). ``"auto"`` (default)
        registers the 3D surface case (``n_dim=3``) with ``"first_order"`` and
        leaves other dimensions unregistered (``None``). ``None`` returns raw
        coefficients. ``"first_order"`` uses the l=1 ellipsoid (first-order
        ellipsoid canonicalization, Brechbühler et al. 1995) to align both the
        codomain orientation and the parameter sphere (SO(3)); it requires
        ``n_dim=3``. ``"moment"`` aligns the codomain to the inertia-tensor
        principal axes and scales by centroid size.
    scale : bool, default=True
        Whether registration removes size (shape space) or keeps it (form
        space). Only used when ``registration != None``.
    scale_method : {None, "semi_major_axis", "ellipsoid_volume", "centroid_size"}, default=None
        Size measure when ``scale=True``. ``None`` resolves to the method
        default (``"first_order"``: ``"semi_major_axis"``; ``"moment"``:
        ``"centroid_size"``).
    align_parameter : bool, default=True
        Parameter-domain (SO(3)) alignment. ``"first_order"`` always applies
        it; the toggle is not yet separately honored.
    reflect : bool, default=False
        Whether to also remove reflection (chirality). Not yet honored for
        the codomain (orientation is preserved).
    return_transform : bool, default=False
        Reserved; not yet implemented.
    n_jobs: int, default=None
        The number of jobs to run in parallel. None means 1 unless in a
        joblib.parallel_backend context. -1 means using all processors.
    verbose: int, default=0
        The verbosity level.


    Notes
    -----
    [Ritche_Kemp_1999]_, [Shen_etal_2009]_

    A surface point :math:`\mathbf{p}(\theta, \phi)` is expanded as

    .. math::
        \mathbf{p}(\theta, \phi) = \sum_{l=0}^{L} \sum_{m=-l}^l
            a_l^m \, S_l^m(\theta, \phi)

    where :math:`S_l^m` are real orthonormal spherical harmonics defined
    in terms of the complex harmonics :math:`Y_l^m`:

    * :math:`S_l^0 = Y_l^0`
    * :math:`S_l^m = \sqrt{2}\,(-1)^m\,\mathrm{Re}(Y_l^m)` for :math:`m > 0`
    * :math:`S_l^m = \sqrt{2}\,(-1)^{|m|}\,\mathrm{Im}(Y_l^{|m|})` for :math:`m < 0`

    The coefficients :math:`a_l^m` are real-valued, so ``transform``
    returns a ``float64`` array.  Conversion utilities
    ``_complex_to_real_sph_coef`` and ``_real_to_complex_sph_coef`` are
    available for interoperability with complex-basis representations.

    References
    ----------

    .. [Ritche_Kemp_1999] Ritchie, D.W., Kemp, G.J.L. (1999) Fast computation, r
       otation, and comparison of low resolution spherical harmonic molecular surfaces.
       J. Comput. Chem. 20: 383–395.
    .. [Shen_etal_2009] Shen, L., Farid, H., McPeek, M.A. (2009)
       Modeling three-dimensional morphological structures using spherical harmonics.
       Evolution (N. Y). 63: 1003–1016.



    """

    # Size measures permitted per registration method (SPHARM = ellipsoid-based).
    _SCALE_METHODS_BY_REGISTRATION = {
        "first_order": {None, "semi_major_axis", "ellipsoid_volume"},
        "moment": {None, "centroid_size"},
    }

    def __init__(
        self,
        n_harmonics=10,
        n_dim=3,
        registration="auto",
        scale=True,
        scale_method=None,
        align_parameter=True,
        reflect=False,
        return_transform=False,
        n_jobs=None,
        verbose=0,
    ):
        self.n_harmonics = n_harmonics
        self.n_dim = n_dim
        self.registration = registration
        self.scale = scale
        self.scale_method = scale_method
        self.align_parameter = align_parameter
        self.reflect = reflect
        self.return_transform = return_transform
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _resolve_method(self):
        """Resolve ``"auto"`` to a concrete method: ``"first_order"`` for the
        3D surface case (``n_dim=3``) with at least the l=1 modes, ``None``
        otherwise.
        """
        if self.registration == "auto":
            if self.n_dim == 3 and self.n_harmonics >= 1:
                return "first_order"
            return None
        return self.registration

    def _validate_registration(self):
        """Validate registration settings (raises on invalid combinations)."""
        method = self._resolve_method()
        validate_registration(
            method,
            self.scale_method,
            self._SCALE_METHODS_BY_REGISTRATION,
            n_dim=self.n_dim,
            return_transform=self.return_transform,
            allow_first_order=True,
        )
        if method == "first_order" and self.n_dim != 3:
            raise ValueError(
                "registration='first_order' for SPHARM requires n_dim=3 (the "
                "l=1 ellipsoid spans a full 3D frame). Use registration="
                "'moment' or None for n_dim=2."
            )

    def _register(self, coef_flat):
        """Apply the configured registration to one flat coefficient vector."""
        method = self._resolve_method()
        if method is None:
            return coef_flat
        if method == "moment":
            return moment_register(
                coef_flat, self.n_dim, scale=self.scale, reflect=self.reflect
            )
        if method == "first_order":
            return self._first_order_register(coef_flat)
        raise NotImplementedError(f"registration='{method}' is not implemented yet.")

    def _first_order_register(self, coef_flat):
        """first_order registration for SPHARM (n_dim=3): A + B, coef-only.

        Implements the first-order-ellipsoid canonicalization theory
        (Brechbühler et al. 1995, §4.1): the degree-1 part of the expansion is
        an ellipsoid (the affine image of the sphere). Writing its l=1
        coordinate matrix (columns x, y, z) as ``M1 = U Σ Vᵀ``:

        - object/codomain rotation (A): align the ellipsoid's principal axes to
          the coordinate axes by applying ``Uᵀ`` to every coefficient vector;
        - parameter-sphere rotation (B): apply the corresponding SO(3) rotation
          ``V`` to all degrees via the Wigner-D representation
          (:func:`rotate_real_sph_coef`);
        - axis ordering by descending semi-axis (largest -> x), via the SVD;
        - translation removed by dropping the constant (l=0) mode;
        - size by the longest semi-axis (``semi_major_axis``) or the ellipsoid
          volume.

        After this the registered first-order ellipsoid is diagonal (canonical)
        with descending positive semi-axes. The remaining sign freedom is the
        ellipsoid's intrinsic Klein-four symmetry (180-deg rotations about each
        principal axis), which degree 1 alone cannot resolve; it is broken with
        a higher-order, rotation- and reparameterization-invariant shape
        moment (see :func:`_axis_third_moments`). Operates purely on
        coefficients (no re-fit), so it composes and is reusable for rotation
        optimization.
        """
        l_max = self.n_harmonics
        if l_max < 1:
            raise ValueError("registration='first_order' requires n_harmonics >= 1.")
        n_coeffs = (l_max + 1) ** 2
        mat = np.asarray(coef_flat, dtype=float).reshape(3, n_coeffs)

        # l=1 ellipsoid: columns m=-1,0,1 -> permute to (x, y, z).
        # Real l=1 SH: S_1^{-1} ~ y, S_1^0 ~ z, S_1^1 ~ x.
        m1_xyz = mat[:, [1, 2, 3]][:, [2, 0, 1]]  # columns x, y, z

        u_mat, sig, wt = np.linalg.svd(m1_xyz)  # m1_xyz = u_mat @ diag(sig) @ wt
        if sig[0] < _FIRST_ORDER_TOL:
            raise ValueError(
                "Degenerate first-order ellipsoid (near-zero semi-major axis); "
                "cannot register. Use registration='moment' or None."
            )
        w_mat = wt.T

        # Sign convention: break the ellipsoid's intrinsic Klein-four symmetry
        # (degree 1 fixes axes only up to 180-deg flips) using higher-order
        # shape information, as Brechbühler et al. (1995) suggest. We make the
        # shape's third moment along each codomain axis positive. The third
        # moment is a geometric integral over the sphere, hence invariant to
        # BOTH codomain rotation and sphere reparameterization (unlike a sum of
        # cubes over modes). Flip the coupled (U, V) columns together.
        m3 = _axis_third_moments(mat, u_mat, l_max)
        for i in range(3):
            if abs(m3[i]) > _SKEW_TOL:
                if m3[i] < 0:
                    u_mat[:, i] = -u_mat[:, i]
                    w_mat[:, i] = -w_mat[:, i]
            else:
                col = u_mat[:, i]
                k = int(np.argmax(np.abs(col)))
                if col[k] < 0:
                    u_mat[:, i] = -u_mat[:, i]
                    w_mat[:, i] = -w_mat[:, i]
        # Proper codomain rotation unless reflection is allowed.
        if not self.reflect and np.linalg.det(u_mat) < 0:
            u_mat[:, -1] = -u_mat[:, -1]
            w_mat[:, -1] = -w_mat[:, -1]

        # (B) Parameter SO(3) alignment in the coefficient domain: rotate the
        # sphere by R = w_mat^T via Wigner-D (per axis).
        rotated = rotate_real_sph_coef(mat.T, w_mat.T)  # (n_coeffs, 3)

        # (A) Codomain rotation + scale + translation removal.
        if self.scale:
            scale_method = self.scale_method or "semi_major_axis"
            if scale_method == "ellipsoid_volume":
                s = (4.0 / 3.0) * np.pi * sig[0] * sig[1] * sig[2]
            else:  # "semi_major_axis"
                s = sig[0]
        else:
            s = 1.0

        out = (u_mat.T @ rotated.T) / s
        out[:, 0] = 0.0  # drop the constant (l=0) mode
        return out.ravel()

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

    def __sklearn_is_fitted__(self):
        """Return True since this is a stateless transformer."""
        return True

    def fit_transform(self, X, y=None, theta_phi=None):
        """Fit and transform in a single step.

        Overridden to support metadata routing of ``theta_phi``.

        Parameters
        ----------
        X : list of array-like of shape (n_coords_i, 3)
            Coordinate values of n_samples.
        y : ignored
        theta_phi : list of array-like of shape (n_coords_i, 2)
            Surface parameterization of n_samples.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_coefficients)
        """
        return self.fit(X, y).transform(X, theta_phi=theta_phi)

    def _transform_single(self, X, theta_phi):
        """Compute real SPHARM coefficients for a single sample.

        Parameters
        ----------
        X : array-like of shape (n_coords, 3)
            Coordinate values of a surface.
        theta_phi : array-like of shape (n_coords, 2)
            Parameters indicating the position on the surface.

        Returns
        -------
        X_transformed : ndarray of shape (3 * (l_max + 1)**2,), float64
            Flat real-valued SPHARM coefficient vector.
            Layout: ``[cx_0_0, ..., cy_0_0, ..., cz_0_0, ...]``.
        """
        l_max = self.n_harmonics
        theta = theta_phi[:, 0]
        phi = theta_phi[:, 1]

        n_coords = len(theta)
        n_coeffs = (l_max + 1) ** 2
        if n_coords < n_coeffs:
            warnings.warn(
                f"Underdetermined system: n_coords ({n_coords}) < "
                f"(n_harmonics+1)**2 ({n_coeffs}). "
                f"lstsq will return a least-norm solution, not a least-squares fit. "
                f"Consider reducing n_harmonics or providing more sample points.",
                UserWarning,
                stacklevel=2,
            )

        B = _real_sph_harm_basis_matrix(l_max, theta, phi)

        sol = sp.linalg.lstsq(B, X)
        X_transformed = sol[0].T.ravel()

        return self._register(X_transformed)

    def transform(self, X, theta_phi=None):
        """Compute SPHARM coefficients.

        Parameters
        ----------
        X: list of array-like
            Coordinate values of n_samples.
            The i-th array-like whose shape (n_coords_i, 3) represents
            3D coordinate values of the i-th sample .

        theta_phi: list of array-like of shape (n_coords, 2)
            Surface parameter of n_samples.
            The i-th array-like of theta and phi values whose shape is (n_coords_i, 2).

        Returns
        -------
        X_transformed: array-like of shape (n_samples, n_coefficients)
            Returns the array-like of SPHARM coefficients.
        """
        if theta_phi is None:
            raise ValueError(
                "theta_phi is required for SphericalHarmonicAnalysis.transform(). "
                "Provide surface parameterization for each sample."
            )

        if self.n_dim < 1:
            raise ValueError(f"n_dim must be a positive integer, got {self.n_dim}")

        self._validate_registration()

        n_dim = self.n_dim
        if isinstance(X, pd.DataFrame):
            X_ = [
                row.dropna().to_numpy().reshape(n_dim, -1).T
                for idx, row in X.iterrows()
            ]
        else:
            X_ = X

        if len(theta_phi) != len(X_):
            raise ValueError(
                f"theta_phi ({len(theta_phi)}) must have the same length "
                f"as X ({len(X_)})"
            )

        if len(X_) > 0:
            d_data = np.asarray(X_[0]).shape[1]
            if d_data != n_dim:
                raise ValueError(
                    f"Each sample must have n_dim={n_dim} columns; got {d_data}."
                )

        X_transformed = np.stack(
            Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(self._transform_single)(X_[i], theta_phi[i])
                for i in range(len(X_))
            )
        )

        return X_transformed

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
        return np.asarray(self._build_feature_names(), dtype=str)

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        return self.n_dim * (self.n_harmonics + 1) ** 2

    def _build_feature_names(self) -> list[str]:
        l_max = self.n_harmonics
        names = []
        for axis in _axis_prefixes(self.n_dim):
            for l in range(l_max + 1):
                for m in range(-l, l + 1):
                    names.append(f"{axis}_{l}_{m}")
        return names

    def _inverse_transform_single(
        self,
        X_transformed,
        theta_range,
        phi_range,
        l_max=None,
    ):
        """Reconstruct a single surface from SPHARM coefficients.

        Parameters
        ----------
        X_transformed : ndarray of shape (n_dim * (n_harmonics + 1)**2,)
            Flat SPHARM coefficient vector for one sample, in axis-major
            layout (one ``(n_harmonics + 1)**2`` block per coordinate).
        theta_range : array-like of shape (n_theta,)
            Polar angle values (colatitude, 0 to pi).
        phi_range : array-like of shape (n_phi,)
            Azimuthal angle values (0 to 2*pi).
        l_max : int, optional
            Maximum degree of harmonics to use. Defaults to
            ``self.n_harmonics``. When less than ``self.n_harmonics``,
            the leading ``(l_max + 1) ** 2`` coefficients of each axis
            block are kept and higher-degree terms are dropped.

        Returns
        -------
        X_coords : ndarray of shape (n_theta, n_phi, n_dim)
            Reconstructed surface coordinates.
        """
        if l_max is None:
            l_max = self.n_harmonics

        n_per_lm_full = (self.n_harmonics + 1) ** 2
        n_per_lm = (l_max + 1) ** 2

        # Axis-major layout: (n_dim, n_per_lm_full) → take leading n_per_lm cols.
        coef_per_lm = (
            np.asarray(X_transformed).reshape(self.n_dim, n_per_lm_full)[:, :n_per_lm].T
        )
        coords = spharm(
            l_max,
            cvt_spharm_coef_to_list(coef_per_lm),
            theta_range,
            phi_range,
        )
        X_coords = np.stack(coords, axis=-1)
        return X_coords

    def inverse_transform(
        self,
        X_transformed,
        theta_range=None,
        phi_range=None,
        l_max=None,
    ):
        """Reconstruct surfaces from SPHARM coefficients.

        Parameters
        ----------
        X_transformed : array-like of shape (n_samples, 3 * (l_max + 1)**2)
            Flat SPHARM coefficient vectors as returned by
            :meth:`transform`.  Layout is axis-major:
            ``[cx_0_0, cx_1_-1, ..., cy_0_0, ..., cz_0_0, ...]``.
        theta_range : array-like of shape (n_theta,), optional
            Polar angle values (colatitude). Defaults to
            ``np.linspace(0, pi, 90)``.
        phi_range : array-like of shape (n_phi,), optional
            Azimuthal angle values. Defaults to
            ``np.linspace(0, 2*pi, 180)``.
        l_max : int, optional
            Maximum degree of harmonics to use. Defaults to
            ``self.n_harmonics``. When smaller, the input coefficient
            vector is truncated to the leading ``(l_max + 1) ** 2``
            terms per axis. Values greater than ``self.n_harmonics``
            raise ``ValueError``.

        Returns
        -------
        X_coords : ndarray of shape (n_samples, n_theta, n_phi, 3)
            Reconstructed surface coordinates.

        Raises
        ------
        ValueError
            If ``l_max`` is negative or greater than ``self.n_harmonics``.
        """
        if theta_range is None:
            theta_range = np.linspace(0, np.pi, 90)
        if phi_range is None:
            phi_range = np.linspace(0, 2 * np.pi, 180)
        if l_max is None:
            l_max = self.n_harmonics
        if l_max < 0:
            raise ValueError(f"l_max must be >= 0, got {l_max}")
        if l_max > self.n_harmonics:
            raise ValueError(
                f"l_max ({l_max}) cannot exceed n_harmonics ({self.n_harmonics})"
            )

        X_coords = np.stack(
            Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(self._inverse_transform_single)(
                    X_transformed[i], theta_range, phi_range, l_max
                )
                for i in range(len(X_transformed))
            )
        )

        return X_coords


###########################################################
#
#   utility functions
#
###########################################################


def _axis_prefixes(n_dim: int) -> list[str]:
    """Return per-axis feature-name prefixes for a ``n_dim``-valued field.

    Uses the legacy ``cx``/``cy``/``cz`` names for ``n_dim <= 3`` (so that
    ``1`` -> ``["cx"]``) and systematic ``c0``, ``c1``, ... names otherwise.
    """
    base = ["cx", "cy", "cz"]
    if n_dim <= len(base):
        return base[:n_dim]
    return [f"c{d}" for d in range(n_dim)]


def _axis_third_moments(mat, axes, l_max, n_theta=30, n_phi=60):
    """Shape third moments along given codomain axes (rotation-invariant sign).

    Reconstructs the surface on a uniform sphere grid and integrates
    ``(p · axis)**3`` with the ``sin(theta)`` area weight. The result is a
    geometric integral over the sphere, hence invariant to both codomain
    rotation and sphere reparameterization; its sign canonicalizes each axis.

    Parameters
    ----------
    mat : ndarray of shape (n_dim, (l_max+1)**2)
        Flat SPHARM coefficients reshaped per axis.
    axes : ndarray of shape (n_dim, k)
        Codomain axes (columns) to evaluate the third moment along.
    l_max : int
        Maximum degree.

    Returns
    -------
    ndarray of shape (k,)
        Third moment along each axis.
    """
    # Center the shape (drop the l=0 constant mode) so the moment is
    # translation-invariant.
    mat_centered = np.asarray(mat, dtype=float).copy()
    mat_centered[:, 0] = 0.0

    theta_g = np.linspace(0.0, np.pi, n_theta)
    phi_g = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    tg, pg = np.meshgrid(theta_g, phi_g, indexing="ij")
    basis = _real_sph_harm_basis_matrix(l_max, tg.ravel(), pg.ravel())
    p = basis @ mat_centered.T  # (n_grid, n_dim)
    weights = np.sin(tg).ravel()
    proj = p @ axes  # (n_grid, k)
    return np.sum(weights[:, None] * proj**3, axis=0)


def _wigner_d_small(l: int, beta: float) -> npt.NDArray[np.float64]:
    """Wigner small-d matrix ``d^l_{m'm}(beta)`` (rows m', cols m, -l..l).

    Standard real Wigner small-d (closed-form factorial series), matching
    Ritchie & Kemp (1999) Eq. (10) and Shen et al. (2009) Eq. (14); verified
    against the textbook ``d^1`` to machine precision. Adequate for moderate
    degrees (``l <= ~30``); rows/columns are ordered ``m = -l, ..., l``.
    """
    dim = 2 * l + 1
    d = np.zeros((dim, dim))
    cb, sb = np.cos(beta / 2.0), np.sin(beta / 2.0)
    orders = range(-l, l + 1)
    for i, mp in enumerate(orders):
        for j, m in enumerate(orders):
            pref = np.sqrt(
                factorial(l + mp)
                * factorial(l - mp)
                * factorial(l + m)
                * factorial(l - m)
            )
            s_min, s_max = max(0, m - mp), min(l + m, l - mp)
            total = 0.0
            for s in range(s_min, s_max + 1):
                den = (
                    factorial(l + m - s)
                    * factorial(s)
                    * factorial(mp - m + s)
                    * factorial(l - mp - s)
                )
                total += (
                    (-1.0) ** (mp - m + s)
                    / den
                    * cb ** (2 * l - mp + m - 2 * s)
                    * sb ** (mp - m + 2 * s)
                )
            d[i, j] = pref * total
    return d


def _wigner_D(
    l: int, alpha: float, beta: float, gamma: float
) -> npt.NDArray[np.complexfloating]:
    """Complex Wigner-D matrix ``D^l_{m'm} = e^{-i m' a} d^l_{m'm}(b) e^{-i m g}``.

    ZYZ Euler convention (alpha, gamma about z; beta about y), matching
    Ritchie & Kemp (1999) Eq. (9) and Shen et al. (2009) Eq. (14).
    """
    d = _wigner_d_small(l, beta)
    m = np.arange(-l, l + 1)
    return np.exp(-1j * m * alpha)[:, None] * d * np.exp(-1j * m * gamma)[None, :]


def rotate_real_sph_coef(
    coef_per_lm: npt.NDArray[np.float64], rotation: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Rotate real SPHARM coefficients by a 3D rotation, in the coefficient domain.

    Applies the Wigner-D rotational property of spherical harmonics (Ritchie &
    Kemp 1999, Eqs. 9/13; Shen et al. 2009, Eq. 14): per degree ``l`` the
    coefficients transform as ``c'_m = sum_n D^l_{m n}(R) c_n``. This is the
    method the literature uses for SPHARM rotation/registration (rotating
    coefficients, NOT re-fitting a re-parameterized surface). It is reusable
    for rotation optimization (e.g. axis-constrained rotational matching,
    Ritchie & Kemp 1999). As a property, the result equals re-expanding after
    rotating the sphere parameterization by ``rotation`` (``p -> rotation @ p``).

    Parameters
    ----------
    coef_per_lm : ndarray of shape ((l_max+1)**2,) or ((l_max+1)**2, D)
        Real SPHARM coefficients in flat ``(l, m)`` ordering.
    rotation : ndarray of shape (3, 3)
        Orthogonal matrix applied to the parameter sphere. Proper rotations
        (``det=+1``) and improper ones (``det=-1``, i.e. with a reflection)
        are both accepted; an improper map is handled as inversion composed
        with a proper rotation (parity ``(-1)**l`` per degree).

    Returns
    -------
    ndarray of same shape as ``coef_per_lm``
        Rotated real coefficients.
    """
    coef = np.asarray(coef_per_lm)
    squeeze = coef.ndim == 1
    if squeeze:
        coef = coef[:, None]
    l_max = int(round(np.sqrt(coef.shape[0]))) - 1

    rot = np.asarray(rotation, dtype=float)
    parity = np.linalg.det(rot) < 0
    if parity:
        rot = -rot  # -R is proper for 3x3; the inversion adds (-1)**l per degree

    with warnings.catch_warnings():
        # At gimbal lock (beta = 0 or pi) the ZYZ split is non-unique, but any
        # valid decomposition yields the same D^l(R); scipy's warning is benign.
        warnings.simplefilter("ignore", UserWarning)
        alpha, beta, gamma = Rotation.from_matrix(rot).as_euler("ZYZ")
    cc = _real_to_complex_sph_coef(coef.astype(np.complex128))
    out = np.empty_like(cc)
    for l in range(l_max + 1):
        block = _wigner_D(l, alpha, beta, gamma) @ cc[l * l : (l + 1) ** 2]
        if parity:
            block = block * ((-1) ** l)
        out[l * l : (l + 1) ** 2] = block
    rotated = _complex_to_real_sph_coef(out)
    return rotated[:, 0] if squeeze else rotated


def _real_sph_harm_y(
    l: int,
    m: int,
    theta: npt.NDArray[np.float64],
    phi: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""Evaluate a real spherical harmonic :math:`S_l^m`.

    Defined via :func:`scipy.special.sph_harm_y`:

    * :math:`m = 0`:  :math:`S_l^0 = Y_l^0`
    * :math:`m > 0`:  :math:`S_l^m = \sqrt{2}\,(-1)^m\,\mathrm{Re}(Y_l^m)`
    * :math:`m < 0`:  :math:`S_l^m = \sqrt{2}\,(-1)^{|m|}\,\mathrm{Im}(Y_l^{|m|})`

    This evaluates to:

    * :math:`S_l^m = \sqrt{2}\,N_l^m\,P_l^m(\cos\theta)\,\cos(m\varphi)`
      for :math:`m > 0`
    * :math:`S_l^{-|m|} = \sqrt{2}\,N_l^{|m|}\,P_l^{|m|}(\cos\theta)\,
      \sin(|m|\varphi)` for :math:`m < 0`

    The :math:`(-1)^m` factor cancels the Condon-Shortley phase
    included in ``sph_harm_y``, yielding positive cosine/sine.
    """
    if m == 0:
        return sp.special.sph_harm_y(l, 0, theta, phi).real
    elif m > 0:
        return np.sqrt(2) * (-1) ** m * np.real(sp.special.sph_harm_y(l, m, theta, phi))
    else:
        return (
            np.sqrt(2)
            * (-1) ** abs(m)
            * np.imag(sp.special.sph_harm_y(l, abs(m), theta, phi))
        )


def _real_sph_harm_basis_matrix(
    l_max: int,
    theta: npt.NDArray[np.float64],
    phi: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    r"""Build real-valued spherical harmonic design matrix.

    Columns correspond to ``(l, m)`` pairs in the ordering
    ``(0,0), (1,-1), (1,0), (1,1), (2,-2), ...``.

    Parameters
    ----------
    l_max : int
        Maximum degree.
    theta : ndarray of shape (N,)
        Polar angle (colatitude) values.
    phi : ndarray of shape (N,)
        Azimuthal angle values.

    Returns
    -------
    ndarray of shape (N, (l_max+1)**2), float64
        Real-valued design matrix.
    """
    n_pts = len(theta)
    n_coeffs = (l_max + 1) ** 2
    B = np.empty((n_pts, n_coeffs))

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            idx = l**2 + l + m
            B[:, idx] = _real_sph_harm_y(l, m, theta, phi)

    return B


def _complex_to_real_sph_coef(
    coef_complex: npt.NDArray[np.complexfloating],
) -> npt.NDArray[np.float64]:
    r"""Convert complex SH coefficients to real SH coefficients.

    Includes the :math:`(-1)^m` Condon-Shortley phase factor,
    which makes this different from DHA's ``_complex_to_real_coef``.

    The mapping for degree ``l``, order ``m`` is:

    * ``m = 0``:  ``a_{l,0} = Re(c_{l,0})``
    * ``m > 0``:  ``a_{l,m} = \sqrt{2}\,(-1)^m\,Re(c_{l,m})``
    * ``m < 0``:  ``a_{l,m} = -\sqrt{2}\,(-1)^{|m|}\,Im(c_{l,|m|})``

    Parameters
    ----------
    coef_complex : ndarray of shape ((l_max+1)**2,) or ((l_max+1)**2, D)
        Complex coefficients in flat ordering.

    Returns
    -------
    ndarray of same shape, float64
        Real-valued coefficients.
    """
    coef_real = np.empty_like(coef_complex, dtype=np.float64)
    n_coef = coef_complex.shape[0]
    l_max = int(np.sqrt(n_coef)) - 1

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            idx = l**2 + l + m
            if m == 0:
                coef_real[idx] = np.real(coef_complex[idx])
            elif m > 0:
                coef_real[idx] = np.sqrt(2) * (-1) ** m * np.real(coef_complex[idx])
            else:
                idx_pos = l**2 + l + (-m)
                coef_real[idx] = (
                    -np.sqrt(2) * (-1) ** abs(m) * np.imag(coef_complex[idx_pos])
                )

    return coef_real


def _real_to_complex_sph_coef(
    coef_real: npt.NDArray[np.float64],
) -> npt.NDArray[np.complexfloating]:
    r"""Convert real SH coefficients to complex SH coefficients.

    Inverse of :func:`_complex_to_real_sph_coef`.  The output satisfies
    conjugate symmetry: ``c_{l,-m} = (-1)^m \overline{c_{l,m}}``.

    Parameters
    ----------
    coef_real : ndarray of shape ((l_max+1)**2,) or ((l_max+1)**2, D)
        Real-valued coefficients in flat ordering.

    Returns
    -------
    ndarray of same shape, complex128
        Complex coefficients.
    """
    coef_complex = np.empty_like(coef_real, dtype=np.complex128)
    n_coef = coef_real.shape[0]
    l_max = int(np.sqrt(n_coef)) - 1

    for l in range(l_max + 1):
        # m = 0
        idx_0 = l**2 + l
        coef_complex[idx_0] = coef_real[idx_0] + 0j

        # m > 0 and corresponding m < 0
        for m in range(1, l + 1):
            idx_pos = l**2 + l + m
            idx_neg = l**2 + l - m
            c_pos = (
                (-1) ** m * (coef_real[idx_pos] - 1j * coef_real[idx_neg]) / np.sqrt(2)
            )
            coef_complex[idx_pos] = c_pos
            coef_complex[idx_neg] = (-1) ** m * np.conj(c_pos)

    return coef_complex


def xyz2spherical(xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    xyz : ndarray of shape (n, 3)
        Cartesian coordinates (x, y, z). Points are assumed to lie on or
        near the unit sphere.

    Returns
    -------
    theta_phi : ndarray of shape (n, 2)
        Spherical coordinates ``[theta, phi]`` where ``theta`` is the
        polar angle (colatitude, 0 to pi) and ``phi`` is the azimuthal
        angle (-pi to pi).
    """
    theta = np.arccos(xyz[:, 2])
    xy_norm = np.linalg.norm(xyz[:, 0:2], axis=1)
    is_pole = xy_norm < _POLE_TOL
    phi = np.where(
        is_pole,
        0.0,
        np.sign(xyz[:, 1]) * np.arccos(xyz[:, 0] / np.where(is_pole, 1.0, xy_norm)),
    )
    return np.array([theta, phi]).T


def spharm(
    l_max: int,
    coef: list[npt.ArrayLike],
    theta_range=None,
    phi_range=None,
):
    """Reconstruct surface coordinates from SPHARM coefficients.

    Parameters
    ----------
    l_max : int
        Maximum degree of spherical harmonics.
    coef : list of array-like
        Real SPHARM coefficients. ``coef[l]`` has shape ``(2*l+1, D)``
        and ``coef[l][l+m]`` holds the ``D`` components for
        degree ``l`` and order ``m`` (``D=3`` for 3D Cartesian surfaces).
    theta_range : array-like of shape (n_theta,), optional
        Polar angle values (colatitude, 0 to pi). Defaults to
        ``np.linspace(0, pi, 90)``.
    phi_range : array-like of shape (n_phi,), optional
        Azimuthal angle values (0 to 2*pi). Defaults to
        ``np.linspace(0, 2*pi, 180)``.

    Returns
    -------
    tuple of ndarray of shape (n_theta, n_phi)
        Reconstructed coordinates. The tuple length equals the codomain
        dimension ``D`` (e.g. ``(x, y, z)`` for ``D=3``).
    """
    if theta_range is None:
        theta_range = np.linspace(0, np.pi, 90)
    if phi_range is None:
        phi_range = np.linspace(0, 2 * np.pi, 180)

    theta_grid, phi_grid = np.meshgrid(theta_range, phi_range, indexing="ij")
    B = _real_sph_harm_basis_matrix(l_max, theta_grid.ravel(), phi_grid.ravel())

    coef_matrix = np.vstack([coef[l] for l in range(l_max + 1)])  # ((l_max+1)^2, D)

    coords = B @ coef_matrix  # (N, D)
    n_theta = len(theta_range)
    n_phi = len(phi_range)

    return tuple(
        coords[:, d].reshape(n_theta, n_phi) for d in range(coef_matrix.shape[1])
    )


def cvt_spharm_coef_to_list(
    coef: npt.NDArray[np.float64],
) -> list[npt.NDArray[np.float64]]:
    """Convert SPHARM coefficient matrix to a nested list by degree.

    Parameters
    ----------
    coef : ndarray of shape ((l_max+1)**2, D) or (D, (l_max+1)**2)
        SPHARM coefficient matrix. ``D`` is the number of components of
        the field expanded on the sphere (``D=3`` for 3D Cartesian
        coordinates).  Both orientations are accepted; if the second
        axis matches ``(l_max+1)**2``, the matrix is transposed.

    Returns
    -------
    coef_list : list of ndarray
        ``coef_list[l]`` has shape ``(2*l+1, D)`` for degree ``l``.

    Raises
    ------
    ValueError
        If ``coef`` is not 2-D, or neither axis is a perfect square
        (``(l_max+1)**2``).
    """
    coef_arr = np.asarray(coef)
    if coef_arr.ndim != 2:
        raise ValueError(
            f"coef must be 2-D ((n_lm, D) or (D, n_lm)); got shape {coef_arr.shape}."
        )

    n_rows, n_cols = coef_arr.shape
    rows_sqrt = np.sqrt(n_rows)
    cols_sqrt = np.sqrt(n_cols)
    if rows_sqrt.is_integer():
        coef_per_lm = coef_arr
        lmax = int(rows_sqrt) - 1
    elif cols_sqrt.is_integer():
        coef_per_lm = coef_arr.T
        lmax = int(cols_sqrt) - 1
    else:
        raise ValueError(
            f"Invalid coefficient shape {coef_arr.shape}: neither axis is a "
            f"perfect square ((l_max+1)**2)."
        )

    coef_list = [
        np.array([coef_per_lm[l**2 + l + m] for m in range(-l, l + 1, 1)])
        for l in range(0, lmax + 1, 1)
    ]
    return coef_list
