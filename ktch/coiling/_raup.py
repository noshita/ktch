"""Raup's model."""

# Copyright 2026 Koji Noshita
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

import warnings

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
)
from sklearn.utils.parallel import Parallel, delayed

from ._generating_curve import _pad_orientation, _surfaces_to_frame, whorl_theta_range
from ._panel import _check_panel, _check_surface_panel

_VALID_ESTIMATORS = ("ml_2d", "surface")

# Surface fit warns when its residual RMS exceeds this fraction of the mean
# tube radius.
_SURFACE_FIT_RTOL = 1e-3


def _validate_raup_params(w_r: float, t_r: float, d_r: float, r0: float) -> None:
    if not w_r > 1.0:
        raise ValueError(f"w_r (whorl expansion rate) must be > 1, got {w_r}")
    if not (-1.0 < d_r < 1.0):
        raise ValueError(f"d_r must be in (-1, 1), got {d_r}")
    if not r0 > 0.0:
        raise ValueError(f"r0 must be > 0, got {r0}")


def _raup_discriminant(w_r: float, t_r: float, d_r: float) -> float:
    r"""Trajectory discriminant :math:`\Lambda` of Raup's model.

    .. math::

        \Lambda = 4\pi^2 (1 + D_R)^2
            + (\ln W_R)^2 \bigl[(1 + D_R)^2 + 4 T_R^2\bigr].

    :math:`\Lambda` is the radicand of the arc-length relation and
    the denominator of the Raup-to-growing tube conversion (Noshita 2014).
    The published ``4 T_R`` term is corrected to ``4 T_R**2``.
    """
    log_w = np.log(w_r)
    one_p_d = 1.0 + d_r
    return 4.0 * np.pi**2 * one_p_d**2 + log_w**2 * (one_p_d**2 + 4.0 * t_r**2)


def _rotation_x(angle: float) -> npt.NDArray[np.float64]:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_z(angle: float) -> npt.NDArray[np.float64]:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _raup_surface(
    w_r: float,
    t_r: float,
    d_r: float,
    delta_r: float = 0.0,
    gamma_r: float = 0.0,
    r0: float = 1.0,
    theta_range: npt.ArrayLike | None = None,
    phi_range: npt.ArrayLike | None = None,
    aperture=None,
) -> npt.NDArray[np.float64]:
    r"""Surface of Raup's model (see :func:`raup`).

    Raup's model [Raup_1965]_ [Raup_1966]_, built in the global frame
    following the definition in [Noshita_2014]_:

    .. math::

        p(\theta, \phi) = r_0\, w_r^{\theta / 2\pi}\, R_z(\theta) \cdot
        \bigl( R_z(\gamma_r)\, R_x(\delta_r)\, c(\phi) + o \bigr),

    with offset :math:`o = ((1+d_r)/(1-d_r),\ 0,\ 2 t_r/(1-d_r))` and coiling
    axis :math:`z`.

    References
    ----------
    .. [Raup_1965] Raup, D.M., Michelson, A., 1965. Theoretical Morphology of
       the Coiled Shell. Science 147, 1294–1295.
    .. [Raup_1966] Raup, D.M., 1966. Geometric analysis of shell coiling: general
       problems. Journal of Paleontology 40, 1178–1190.
    .. [Noshita_2014] Noshita, K., 2014. Quantification and geometric analysis of
       coiling patterns in gastropod shells based on 3D and 2D image data.
       Journal of Theoretical Biology 363, 93–104.
    """
    _validate_raup_params(w_r, t_r, d_r, r0)
    if aperture is not None:
        raise NotImplementedError("general aperture shapes are not supported yet")
    if theta_range is None:
        theta_range = whorl_theta_range(4.0)
    if phi_range is None:
        phi_range = np.linspace(0.0, 2.0 * np.pi, 90)
    theta = np.asarray(theta_range, dtype=float)
    phi = np.asarray(phi_range, dtype=float)

    vx = 2.0 * d_r / (1.0 - d_r) + 1.0
    vz = 2.0 * t_r * (d_r / (1.0 - d_r) + 1.0)
    offset = np.array([vx, 0.0, vz])

    circle = np.column_stack([np.cos(phi), np.zeros_like(phi), np.sin(phi)])
    rotation = _rotation_z(gamma_r) @ _rotation_x(delta_r)
    point_local = circle @ rotation.T + offset  # (n_phi, 3)

    scale = r0 * w_r ** (theta / (2.0 * np.pi))  # (n_theta,)
    plx, ply, plz = point_local[:, 0], point_local[:, 1], point_local[:, 2]
    cos_t = np.cos(theta)[:, None]
    sin_t = np.sin(theta)[:, None]
    x = scale[:, None] * (plx[None, :] * cos_t - ply[None, :] * sin_t)
    y = scale[:, None] * (plx[None, :] * sin_t + ply[None, :] * cos_t)
    z = scale[:, None] * plz[None, :]
    return np.stack([x, y, z], axis=-1)


def raup(
    w_r: float,
    t_r: float,
    d_r: float,
    delta_r: float = 0.0,
    gamma_r: float = 0.0,
    r0: float = 1.0,
    theta_range: npt.ArrayLike | None = None,
    phi_range: npt.ArrayLike | None = None,
    aperture=None,
    output: str = "surface",
) -> npt.NDArray[np.float64]:
    r"""Generate a form from Raup's model.

    Raup’s logarithmic shell coiling model [Raup_1965]_ [Raup_1966]_ describes a shell
    by a trajectory of a generating curve that expands, rotates, and translates
    along a fixed coiling axis.

    Parameters
    ----------
    w_r : float
        Whorl expansion rate :math:`W_R` (> 1).
    t_r : float
        Translation rate :math:`T_R`. ``t_r = 0`` gives a planispiral.
    d_r : float
        Relative position of generating curve :math:`D_R`, in ``(-1, 1)``.
    delta_r, gamma_r : float, default = 0.0
        Aperture orientation :math:`(\Delta, \Gamma)` (the global rotation is
        ``Rz(gamma_r) Rx(delta_r)``). ``(0, 0)`` is the classical radial-axial
        aperture plane.
    r0 : float, default = 1.0
        Initial tube radius (scale).
    theta_range : array-like of shape (n_theta,), optional
        Coiling-angle (radians). Defaults to four whorls.
    phi_range : array-like of shape (n_phi,), optional
        Aperture-angle samples. Defaults to ``np.linspace(0, 2*pi, 90)``.
    aperture : None
        Aperture shape.
    output : {"surface"}, default = "surface"
        Form representation to return.
        Only ``"surface"`` is implemented; other representations are reserved.

    Returns
    -------
    X : ndarray of shape (n_theta, n_phi, 3)
        Surface coordinates.

    References
    ----------
    .. [Raup_1965] Raup, D.M., Michelson, A., 1965. Theoretical Morphology of
       the Coiled Shell. Science 147, 1294–1295.
    .. [Raup_1966] Raup, D.M., 1966. Geometric analysis of shell coiling: general
       problems. Journal of Paleontology 40, 1178–1190.
    """
    if output != "surface":
        raise NotImplementedError(
            f"output={output!r} is reserved; only 'surface' is implemented"
        )
    return _raup_surface(
        w_r, t_r, d_r, delta_r, gamma_r, r0, theta_range, phi_range, aperture
    )


def l_r(theta, w_r, t_r, d_r, r0=1.0):
    r"""Arc length of growth trajectory at coiling angle ``theta``.

    Maps the coiling angle :math:`\theta` of the Raup's model to
    the arc length :math:`l_R` of the reference-point trajectory ([Noshita_2014]_):

    .. math::

        l_R(\theta) = r_0\,(W_R^{\theta / 2\pi} - 1)\,
            \frac{\sqrt{\Lambda}}{(1 - D_R)\,\ln W_R},

    where :math:`\Lambda` is the trajectory discriminant. The relation is
    closed-form for constant parameters; ``theta`` may be array-like.

    Parameters
    ----------
    theta : array-like
        Coiling angle :math:`\theta` (radians).
    w_r, t_r, d_r : float
        Raup parameters (``w_r > 1``, ``-1 < d_r < 1``).
    r0 : float, default = 1.0
        Initial tube radius (arc length scales with ``r0``).

    Returns
    -------
    l_r : float or ndarray
        Trajectory arc length at ``theta``.

    See Also
    --------
    theta_r : Inverse function (arc length to coiling angle).

    References
    ----------
    .. [Noshita_2014] Noshita, K., 2014. Quantification and geometric analysis of
       coiling patterns in gastropod shells based on 3D and 2D image data.
       Journal of Theoretical Biology 363, 93–104.
    """
    _validate_raup_params(w_r, t_r, d_r, r0)
    theta = np.asarray(theta, dtype=float)
    log_w = np.log(w_r)
    sqrt_lambda = np.sqrt(_raup_discriminant(w_r, t_r, d_r))
    out = (
        r0
        * (w_r ** (theta / (2.0 * np.pi)) - 1.0)
        * sqrt_lambda
        / ((1.0 - d_r) * log_w)
    )
    return float(out) if out.ndim == 0 else out


def theta_r(l_r, w_r, t_r, d_r, r0=1.0):
    r"""Coiling angle of growth trajectory at arc length ``l_r``.

    Inverse of :func:`l_r` (analytic, since the arc length is affine in
    :math:`W_R^{\theta/2\pi}`):

    .. math::

        \theta(l_R) = \frac{2\pi}{\ln W_R}\,
            \ln\!\left(1 + \frac{l_R\,(1 - D_R)\,\ln W_R}{r_0\,\sqrt{\Lambda}}\right).

    Parameters
    ----------
    l_r : array-like
        Trajectory arc length.
    w_r, t_r, d_r : float
        Raup parameters (``w_r > 1``, ``-1 < d_r < 1``).
    r0 : float, default = 1.0
        Initial tube radius.

    Returns
    -------
    theta : float or ndarray
        Coiling angle :math:`\theta` (radians).

    See Also
    --------
    l_r : Inverse function (coiling angle to arc length).

    References
    ----------
    .. [Noshita_2014] Noshita, K., 2014. Quantification and geometric analysis of
       coiling patterns in gastropod shells based on 3D and 2D image data.
       Journal of Theoretical Biology 363, 93–104.
    """
    _validate_raup_params(w_r, t_r, d_r, r0)
    l_r = np.asarray(l_r, dtype=float)
    log_w = np.log(w_r)
    sqrt_lambda = np.sqrt(_raup_discriminant(w_r, t_r, d_r))
    out = (2.0 * np.pi / log_w) * np.log1p(
        l_r * (1.0 - d_r) * log_w / (r0 * sqrt_lambda)
    )
    return float(out) if out.ndim == 0 else out


def _estimate_raup_ml2d(lateral_series, c, b):
    """Estimate ``(w_r, t_r, d_r)`` from digitizing points of a specimen.

    ``d_r = c / (c + b)`` comes from the umbilical side measurements.
    ``(w_r, t_r, r0)`` and a height datum ``f0`` are fit by least squares
    (the MLE under Gaussian error) to the lateral ``(d_i, f_i)`` series through
    the Raup's model: :func:`_raup_surface` at ``theta = pi * i`` and
    ``phi = 0`` (the lateral edge).

    Parameters
    ----------
    lateral_series : ndarray of shape (n_points, 2)
        Lateral side measurements of digitizing points ``(d, f)``.
    c, b : float
        Umbilical side measurements.

    Returns
    -------
    ndarray of shape (5,)
        ``(w_r, t_r, d_r, 0, 0)``; orientation columns are not estimated.
    """
    lateral = np.asarray(lateral_series, dtype=float)
    d = lateral[:, 0]
    f = lateral[:, 1]
    n = len(d)
    theta = np.pi * np.arange(n, dtype=float)  # lateral digitizing: theta_i = pi i
    phi0 = np.array([0.0])  # aperture edge

    d_r = c / (c + b)

    # Initialization: w_r from the radial ratios, t_r from the f-vs-d slope, r0
    # from the scale d0 = 2 r0 / (1 - d_r), f0 from the height datum.
    w0 = (d[-1] / d[0]) ** (2.0 / (n - 1)) if n > 1 and d[0] > 0 else 1.5
    w0 = max(w0, 1.0 + 1e-6)
    d_span = d[-1] - d[0]
    t0 = (f[-1] - f[0]) / d_span if abs(d_span) > 1e-12 else 1.0
    r0_0 = max(d[0] * (1.0 - d_r) / 2.0, 1e-9) if d[0] > 0 else 1.0
    f0_0 = f[0] - t0 * d[0]

    def residuals(params):
        w_r, t_r, r0, f0 = params
        pts = _raup_surface(w_r, t_r, d_r, 0.0, 0.0, r0, theta, phi0)[:, 0, :]
        d_hat = np.hypot(pts[:, 0], pts[:, 1])
        return np.concatenate([d_hat - d, (pts[:, 2] + f0) - f])

    sol = least_squares(
        residuals,
        x0=[w0, t0, r0_0, f0_0],
        bounds=(
            [1.0 + 1e-9, -np.inf, 1e-12, -np.inf],
            [np.inf, np.inf, np.inf, np.inf],
        ),
    )
    return np.array([float(sol.x[0]), float(sol.x[1]), d_r, 0.0, 0.0])


def _validate_cb(c, b, n_samples):
    """Validate the per-specimen umbilical measurements ``c``, ``b``."""
    if c is None or b is None:
        raise ValueError(
            "RaupModel.transform requires c and b (umbilical measurements) to "
            "estimate d_r = c / (c + b)."
        )
    c = np.asarray(c, dtype=float)
    b = np.asarray(b, dtype=float)
    if c.shape != (n_samples,) or b.shape != (n_samples,):
        raise ValueError(
            f"c and b must each have shape ({n_samples},); got {c.shape} and {b.shape}."
        )
    if np.any(c < 0) or np.any(b <= 0):
        raise ValueError("c must be >= 0 and b must be > 0 (so d_r in [0, 1)).")
    return c, b


def _estimate_raup_surface(surface):
    r"""Estimate ``(w_r, t_r, d_r, delta_r, gamma_r)`` from a structured surface.

    Fit :func:`raup` to the surface coordinates by least squares, with the rigid
    pose. Only the coordinates are used; the coiling angle ``theta`` and aperture
    angle ``phi`` grids are not assumed, and the aperture orientation is
    recovered.

    Parameters
    ----------
    surface : ndarray of shape (n_theta, n_phi, 3)
        Structured surface coordinates.

    Returns
    -------
    ndarray of shape (5,)
        Estimated ``(w_r, t_r, d_r, delta_r, gamma_r)``.
    """
    S = np.asarray(surface, dtype=float)
    n_theta, n_phi = S.shape[0], S.shape[1]
    u = np.linspace(0.0, 1.0, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)  # matches inverse_transform default

    centers = S.mean(axis=1)  # section centroids == generating-curve reference locus
    radius = np.linalg.norm(S - centers[:, None, :], axis=2).mean(axis=1)
    cbar = centers.mean(axis=0)
    # Coiling axis from the centerline's areal velocity about cbar.
    axis = np.sum(np.cross(centers[:-1] - cbar, np.diff(centers, axis=0)), axis=0)
    naxis = np.linalg.norm(axis)
    axis = axis / naxis if naxis > 1e-12 else np.array([0.0, 0.0, 1.0])

    # Unknowns parameters `q`:
    # w_r, t_r, d_r, delta_r, gamma_r, theta_span, r0, rotvec[3], translation[3]
    def residuals(q):
        theta = q[5] * u
        model = _raup_surface(q[0], q[1], q[2], q[3], q[4], q[6], theta, phi)
        rot = Rotation.from_rotvec(q[7:10]).as_matrix()
        return (model @ rot.T + q[10:13] - S).ravel()

    lower = [1.0 + 1e-9, -np.inf, -0.999, -0.9, -0.9, 1e-6, 1e-9, *([-np.inf] * 6)]
    upper = [np.inf, np.inf, 0.999, 0.9, 0.9, np.inf, np.inf, *([np.inf] * 6)]

    best_x, best_rms = None, np.inf
    for sign in (1.0, -1.0):
        # Align the coiling axis to z, then read the canonical warm start.
        align = Rotation.align_vectors([[0.0, 0.0, 1.0]], [sign * axis])[0]
        canon = (centers - cbar) @ align.as_matrix().T  # centerline aligned to z
        # Coiling angle from the winding; w_r, r0 from the radius law given it.
        ang = np.unwrap(np.arctan2(canon[:, 1], canon[:, 0]))
        theta_span = abs(float(ang[-1] - ang[0]))
        theta_n = u * theta_span
        coef = np.linalg.lstsq(
            np.column_stack([np.ones_like(theta_n), theta_n / (2.0 * np.pi)]),
            np.log(radius),
            rcond=None,
        )[0]
        r0_seed = float(np.exp(coef[0]))
        w_r_seed = float(np.exp(coef[1]))
        # Offset (vx, vz) from the centerline relative to the section scale.
        vx = float((np.hypot(canon[:, 0], canon[:, 1]) / radius).mean())
        vz = float((canon[:, 2] / radius).mean())
        d_r_seed = float(np.clip((vx - 1.0) / (vx + 1.0), -0.95, 0.95))
        t_r_seed = vz / (2.0 * (d_r_seed / (1.0 - d_r_seed) + 1.0))
        rv0 = align.inv().as_rotvec()  # canonical -> posed
        x0 = np.array(
            [
                max(w_r_seed, 1.0 + 1e-6),
                t_r_seed,
                d_r_seed,
                0.0,
                0.0,
                max(theta_span, 0.1),
                max(r0_seed, 1e-6),
                *rv0,
                *cbar,
            ]
        )
        sol = least_squares(residuals, x0, bounds=(lower, upper), method="trf")
        rms = float(np.sqrt(np.mean(sol.fun**2)))
        if rms < best_rms:
            best_x, best_rms = sol.x, rms

    scale = max(float(radius.mean()), 1e-12)
    if best_rms > _SURFACE_FIT_RTOL * scale:
        warnings.warn(
            "RaupModel surface estimation did not converge to a good fit "
            f"(residual RMS {best_rms:.3e} vs tube-radius scale {scale:.3e}); the "
            "returned parameters may be unreliable.",
            RuntimeWarning,
            stacklevel=2,
        )
    return np.asarray(best_x[:5], dtype=float)


class RaupModel(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Raup's model.

    Raup’s logarithmic shell coiling model [Raup_1965]_ [Raup_1966]_.
    ``inverse_transform`` is the generative map
    ``Phi: (w_r, t_r, d_r, delta_r, gamma_r) -> form``. ``transform`` estimates
    the parameters from lateral and umbilical measurements (``ml_2d``) or,
    from a structured surface (``surface``).

    Parameters
    ----------
    r0 : float, default = 1.0
        Initial tube radius (scale) used for generation.
    estimator : {"ml_2d", "surface"}, default = "ml_2d"
        Estimation method used by ``transform``. ``"ml_2d"`` fits the lateral
        ``(d, f)`` series and combines it with ``d_r = c / (c + b)``.
        ``"surface"`` fits the model directly to a structured surface
        panel (the coordinate output of ``inverse_transform``), recovering the
        aperture orientation ``(delta_r, gamma_r)`` as well. It is consistent
        with ``inverse_transform`` (``transform(inverse_transform(params)) ~= params``).
    n_jobs : int, optional
        Number of jobs for the per-specimen estimation in ``transform``.
    verbose : int, default = 0
        Verbosity level.

    References
    ----------
    .. [Raup_1965] Raup, D.M., Michelson, A., 1965. Theoretical Morphology of
       the Coiled Shell. Science 147, 1294–1295.
    .. [Raup_1966] Raup, D.M., 1966. Geometric analysis of shell coiling: general
       problems. Journal of Paleontology 40, 1178–1190.
    """

    def __init__(
        self,
        r0: float = 1.0,
        estimator: str = "ml_2d",
        n_jobs: int | None = None,
        verbose: int = 0,
    ):
        self.r0 = r0
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y=None):
        """No-op (stateless). Returns self."""
        return self

    def __sklearn_is_fitted__(self) -> bool:
        """Return True since this is a stateless transformer."""
        return True

    def transform(self, X, c=None, b=None, aperture=None):
        """Estimate Raup parameters from measured shells.

        For ``estimator="ml_2d"``, fit the lateral ``(d, f)`` series and combine
        with ``d_r = c / (c + b)``. For ``estimator="surface"``, fit the
        model directly to a structured surface.

        Parameters
        ----------
        X : list of array-like, ndarray, or DataFrame
            The input panel; its encoding depends on ``estimator``. For
            ``"ml_2d"``, a per-specimen panel of ``(n_points_i, 2)`` lateral
            digitizing points ``(d, f)``. For ``"surface"``, a
            panel of ``(n_theta, n_phi, 3)`` structured surfaces.
        c, b : array-like of shape (n_samples,)
            Per-specimen umbilical measurements (axis-to-inner-margin distance
            and aperture width) giving ``d_r = c / (c + b)``. Required by
            ``"ml_2d"`` and ignored by ``"surface"``.
        aperture : None
            Aperture shape.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, 5)
            Estimated ``(w_r, t_r, d_r, delta_r, gamma_r)``. The ``"ml_2d"``
            estimator returns zeros for the orientation columns; the
            ``"surface"`` estimator recovers them.
        """
        if aperture is not None:
            raise NotImplementedError("general aperture shapes are not supported yet")
        if self.estimator not in _VALID_ESTIMATORS:
            raise ValueError(
                f"estimator must be one of {_VALID_ESTIMATORS}, got {self.estimator!r}"
            )

        if self.estimator == "surface":
            surfaces = _check_surface_panel(X)
            if len(surfaces) == 0:
                return np.empty((0, 5))
            estimates = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(_estimate_raup_surface)(surf) for surf in surfaces
            )
            return np.stack(estimates)

        panel = _check_panel(X, channel_names=["d", "f"])
        c, b = _validate_cb(c, b, panel.n_samples)
        if panel.n_samples == 0:
            return np.empty((0, 5))
        estimates = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_estimate_raup_ml2d)(panel.values[i], c[i], b[i])
            for i in range(panel.n_samples)
        )
        return np.stack(estimates)

    def fit_transform(self, X, y=None, c=None, b=None, aperture=None):
        """Fit and transform in one step.

        Overridden to support metadata routing of ``c``, ``b``, ``aperture``.
        """
        return self.fit(X, y).transform(X, c=c, b=b, aperture=aperture)

    def inverse_transform(
        self,
        X_transformed,
        theta_range=None,
        phi_range=None,
        aperture=None,
        as_frame=False,
    ):
        """Generate shell surfaces from Raup's model parameters.

        Parameters
        ----------
        X_transformed : array-like of shape (n_samples, 5) or (5,)
            Rows of ``(w_r, t_r, d_r, delta_r, gamma_r)``. A 3-column input
            ``(w_r, t_r, d_r)`` is also accepted, with orientation defaulted to 0.
        theta_range, phi_range : array-like, optional
            Sampling grids. See :func:`raup`.
        aperture : None
            Aperture shape; only the circular default is supported.
        as_frame : bool, default = False
            If True, return a long-format ``pandas.DataFrame``.

        Returns
        -------
        X : ndarray of shape (n_samples, n_theta, n_phi, 3) or pd.DataFrame
        """
        params = np.atleast_2d(np.asarray(X_transformed, dtype=float))
        single = np.ndim(X_transformed) == 1
        params = _pad_orientation(params)

        surfaces = [
            _raup_surface(
                w_r,
                t_r,
                d_r,
                delta_r,
                gamma_r,
                self.r0,
                theta_range,
                phi_range,
                aperture,
            )
            for w_r, t_r, d_r, delta_r, gamma_r in params
        ]

        if as_frame:
            return _surfaces_to_frame(surfaces)
        X = np.stack(surfaces)
        return X[0] if single else X

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Parameter names ``(w_r, t_r, d_r, delta_r, gamma_r)``."""
        return np.asarray(["w_r", "t_r", "d_r", "delta_r", "gamma_r"], dtype=str)
