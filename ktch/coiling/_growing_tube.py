"""Growing tube model."""

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

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
)

from ._generating_curve import (
    _assemble_surface,
    _pad_orientation,
    _surfaces_to_frame,
    whorl_s_range,
)

_VALID_METHODS = ("ode", "closed")


def _default_frame0() -> npt.NDArray[np.float64]:
    return np.eye(3)


def _is_varying(p) -> bool:
    """True if a parameter is non-constant (a callable or an array)."""
    return callable(p) or np.ndim(p) > 0


def _as_param_fn(p, s_range):
    """Normalize a parameter (scalar | callable | array) to a callable ``s -> value``.

    An array is interpolated over ``s_range`` (and must match its length).
    """
    if callable(p):
        return p
    arr = np.asarray(p, dtype=float)
    if arr.ndim == 0:
        value = float(arr)
        return lambda s: value
    s_arr = np.asarray(s_range, dtype=float)
    if arr.shape != s_arr.shape:
        raise ValueError(
            "array-valued growing-tube parameters must match s_range in length"
        )
    return lambda s: np.interp(s, s_arr, arr)


def _growing_tube_trajectory_ode(e_g, c_g, t_g, r0, s_range, p0, frame0):
    """Integrate the growing tube ODE system (frame + radius + trajectory).

    Parameters ``e_g, c_g, t_g`` may be scalars, callables ``s -> value``, or
    arrays aligned to ``s_range``.
    """
    s = np.asarray(s_range, dtype=float)
    e_fn = _as_param_fn(e_g, s)
    c_fn = _as_param_fn(c_g, s)
    t_fn = _as_param_fn(t_g, s)

    def rhs(s_, y):
        xi1 = y[3:6]
        xi2 = y[6:9]
        xi3 = y[9:12]
        r = y[12]
        c, t = c_fn(s_), t_fn(s_)
        dp = r * xi1
        dxi1 = c * xi2
        dxi2 = -c * xi1 + t * xi3
        dxi3 = -t * xi2
        dr = e_fn(s_) * r
        return np.concatenate([dp, dxi1, dxi2, dxi3, [dr]])

    y0 = np.concatenate([p0, frame0[0], frame0[1], frame0[2], [r0]])
    sol = solve_ivp(
        rhs, (s[0], s[-1]), y0, t_eval=s, rtol=1e-9, atol=1e-12, method="DOP853"
    )
    y = sol.y.T  # (n, 13)
    trajectory = y[:, 0:3]
    frames = np.stack([y[:, 3:6], y[:, 6:9], y[:, 9:12]], axis=1)
    frames = _orthonormalize(frames)
    radius = y[:, 12]
    return trajectory, frames, radius


def _growing_tube_trajectory_closed(e_g, c_g, t_g, r0, s_range, p0, frame0):
    r"""Closed-form frame and trajectory.

    The frame uses the closed form [Noshita_2014]_. The trajectory integral
    :math:`p(s) = p_0 + \int_0^s r(u)\,\xi_1(u)\,du` is evaluated analytically
    (the integrand is :math:`e^{e_g u}` times sines/cosines of :math:`D_G u`),
    avoiding the error-prone published closed form for the trajectory.

    References
    ----------
    .. [Noshita_2014] Noshita, K., 2014. Quantification and geometric analysis of
      coiling patterns in gastropod shells based on 3D and 2D image data.
      Journal of Theoretical Biology 363, 93–104.
    """
    s = np.asarray(s_range, dtype=float)
    d_g = np.hypot(c_g, t_g)
    a = e_g
    radius = r0 * np.exp(a * s)
    i0 = s.copy() if a == 0.0 else np.expm1(a * s) / a

    if d_g == 0.0:
        frames = np.repeat(frame0[None, :, :], len(s), axis=0)
        trajectory = p0 + r0 * i0[:, None] * frame0[0][None, :]
        return trajectory, frames, radius

    c, t, d = c_g, t_g, d_g
    ds = d * s
    cos = np.cos(ds)
    sin = np.sin(ds)
    xi1 = np.column_stack(
        [(t**2 + c**2 * cos) / d**2, c * sin / d, c * t * (1 - cos) / d**2]
    )
    xi2 = np.column_stack([-c * sin / d, cos, t * sin / d])
    xi3 = np.column_stack(
        [c * t * (1 - cos) / d**2, -t * sin / d, (c**2 + t**2 * cos) / d**2]
    )
    frames = np.stack([xi1 @ frame0, xi2 @ frame0, xi3 @ frame0], axis=1)

    # Analytic integrals of e^{a u} {1, cos(D u), sin(D u)} from 0 to s.
    denom = a**2 + d**2
    exp_as = np.exp(a * s)
    i_cos = (exp_as * (a * cos + d * sin) - a) / denom
    i_sin = (exp_as * (a * sin - d * cos) + d) / denom
    cx = r0 * (t**2 * i0 + c**2 * i_cos) / d**2
    cy = r0 * (c * i_sin) / d
    cz = r0 * (c * t * (i0 - i_cos)) / d**2
    canonical_trajectory = np.column_stack([cx, cy, cz])
    trajectory = p0 + canonical_trajectory @ frame0
    return trajectory, frames, radius


def _orthonormalize(frames: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Gram-Schmidt each (3, 3) frame to counter integration drift."""
    xi1 = frames[:, 0, :]
    xi2 = frames[:, 1, :]
    xi1 = xi1 / np.linalg.norm(xi1, axis=1, keepdims=True)
    xi2 = xi2 - np.sum(xi2 * xi1, axis=1, keepdims=True) * xi1
    xi2 = xi2 / np.linalg.norm(xi2, axis=1, keepdims=True)
    xi3 = np.cross(xi1, xi2)
    return np.stack([xi1, xi2, xi3], axis=1)


def _growing_tube_trajectory(e_g, c_g, t_g, r0, s_range, p0, frame0, method):
    if method == "ode":
        return _growing_tube_trajectory_ode(e_g, c_g, t_g, r0, s_range, p0, frame0)
    return _growing_tube_trajectory_closed(e_g, c_g, t_g, r0, s_range, p0, frame0)


def _growing_tube_surface(
    e_g: float,
    c_g: float,
    t_g: float,
    delta_g: float = 0.0,
    gamma_g: float = 0.0,
    r0: float = 1.0,
    s_range: npt.ArrayLike | None = None,
    phi_range: npt.ArrayLike | None = None,
    aperture=None,
    p0: npt.ArrayLike | None = None,
    frame0: npt.ArrayLike | None = None,
    method: str = "ode",
) -> npt.NDArray[np.float64]:
    """Surface realization of the growing-tube model (see :func:`growing_tube`)."""
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")
    if not r0 > 0.0:
        raise ValueError(f"r0 must be > 0, got {r0}")

    p0 = np.zeros(3) if p0 is None else np.asarray(p0, dtype=float)
    frame0 = _default_frame0() if frame0 is None else np.asarray(frame0, dtype=float)

    varying = _is_varying(e_g) or _is_varying(c_g) or _is_varying(t_g)
    if varying:
        if method != "ode":
            raise ValueError(
                "non-constant (callable/array) parameters require method='ode'"
            )
        if s_range is None:
            raise ValueError("non-constant parameters require an explicit s_range")
    else:
        if c_g < 0.0:
            raise ValueError(f"c_g (standardized curvature) must be >= 0, got {c_g}")
        if s_range is None:
            if np.hypot(c_g, t_g) > 0.0:
                s_range = whorl_s_range(3.0, c_g, t_g)
            else:
                s_range = np.linspace(0.0, 3.0, 300)

    trajectory, frames, radius = _growing_tube_trajectory(
        e_g, c_g, t_g, r0, s_range, p0, frame0, method
    )
    return _assemble_surface(
        trajectory, frames, radius, aperture, (gamma_g, delta_g), phi_range
    )


def growing_tube(
    e_g: float,
    c_g: float,
    t_g: float,
    delta_g: float = 0.0,
    gamma_g: float = 0.0,
    r0: float = 1.0,
    s_range: npt.ArrayLike | None = None,
    phi_range: npt.ArrayLike | None = None,
    aperture=None,
    p0: npt.ArrayLike | None = None,
    frame0: npt.ArrayLike | None = None,
    method: str = "ode",
    output: str = "surface",
) -> npt.NDArray[np.float64]:
    r"""Generate a form from the growing tube model.

    Parameters
    ----------
    e_g, c_g, t_g : float, callable, or array
        Expansion rate, standardized curvature (``>= 0``; ``c_g = 0`` is a
        straight tube), and standardized torsion (``t_g = 0`` is a planispiral);
        Each may be a scalar, a callable ``s -> value``, or an array aligned to
        ``s_range`` (heteromorph growth). Non-constant parameters require
        ``method="ode"`` and an explicit ``s_range``.
        ``e_g`` is the logarithm of the original :math:`E` described in [Okamoto_1988]_
        ([Noshita_2014]_).
    delta_g, gamma_g : float, default = 0.0
        Aperture orientation in the Frenet frame. ``(0, 0)`` is perpendicular to
        the tangent.
    r0 : float, default = 1.0
        Initial tube radius.
    s_range : array-like of shape (n_s,), optional
        Growth-stage samples. Defaults to three whorls (a fixed span if the
        tube is straight).
    phi_range : array-like of shape (n_phi,), optional
        Aperture-angle samples. Defaults to ``np.linspace(0, 2*pi, 90)``.
    aperture : None
        Aperture shape; only the circular default is supported.
    p0 : array-like of shape (3,), optional
        Initial position ``p(0)``. Defaults to the origin.
    frame0 : array-like of shape (3, 3), optional
        Initial frame matrix :math:`\Xi(0)`; rows are
        :math:`(\xi_1, \xi_2, \xi_3)` (tangent, normal, binormal). Defaults to
        the identity.
    method : {"ode", "closed"}, default = "ode"
        ``"ode"`` integrates the frame ODE with ``scipy.integrate.solve_ivp``;
        ``"closed"`` uses the Appendix A closed-form frame with an analytic
        trajectory.
    output : {"surface"}, default = "surface"
        Form representation to return. Only ``"surface"`` is implemented.

    Returns
    -------
    X : ndarray of shape (n_s, n_phi, 3)
        Surface coordinates.

    References
    ----------
    .. [Okamoto_1988] Okamoto, T., 1988. Analysis of heteromorph ammonoids by
      differential geometry. Palaeontology 31, 35–52.
    .. [Noshita_2014] Noshita, K., 2014. Quantification and geometric analysis of
      coiling patterns in gastropod shells based on 3D and 2D image data.
      Journal of Theoretical Biology 363, 93–104.
    """
    if output != "surface":
        raise NotImplementedError(
            f"output={output!r} is reserved; only 'surface' is implemented"
        )
    return _growing_tube_surface(
        e_g,
        c_g,
        t_g,
        delta_g,
        gamma_g,
        r0,
        s_range,
        phi_range,
        aperture,
        p0,
        frame0,
        method,
    )


def l_g(s, e_g, r0=1.0):
    r"""Arc length of the growth trajectory at growth stage ``s``.

    Maps the growth stage :math:`s` to the trajectory arc length :math:`l_G`
    ([Noshita_2014]_):

    .. math::

        l_G(s) = \frac{r_0}{E_G}\left(e^{E_G s} - 1\right),

    with the limit :math:`l_G(s) = r_0 s` as :math:`E_G \to 0`. Because
    :math:`|dp/ds| = r(s)`, the arc length depends only on the expansion rate
    ``e_g`` (and ``r0``), not on ``c_g``/``t_g``. ``s`` may be array-like.

    Parameters
    ----------
    s : array-like
        Growth stage :math:`s`.
    e_g : float
        Expansion rate :math:`E_G` (the logarithm of Okamoto's
        original :math:`E`).
    r0 : float, default = 1.0
        Initial tube radius (arc length scales with ``r0``).

    Returns
    -------
    l_g : float or ndarray
        Trajectory arc length at ``s``.

    See Also
    --------
    s_g : Inverse function (arc length to growth stage).

    References
    ----------
    .. [Noshita_2014] Noshita, K., 2014. Quantification and geometric analysis of
       coiling patterns in gastropod shells based on 3D and 2D image data.
       Journal of Theoretical Biology 363, 93–104.
    """
    if not r0 > 0.0:
        raise ValueError(f"r0 must be > 0, got {r0}")
    s = np.asarray(s, dtype=float)
    if e_g == 0.0:
        out = r0 * s
    else:
        out = (r0 / e_g) * np.expm1(e_g * s)
    return float(out) if out.ndim == 0 else out


def s_g(l_g, e_g, r0=1.0):
    r"""Growth stage of the growth trajectory at arc length ``l_g``.

    Inverse of :func:`l_g` ([Noshita_2014]_):

    .. math::

        s(l_G) = \frac{1}{E_G}\,\ln\!\left(1 + \frac{E_G\, l_G}{r_0}\right),

    with the limit :math:`s = l_G / r_0` as :math:`E_G \to 0`. ``l_g`` may be
    array-like.

    Parameters
    ----------
    l_g : array-like
        Trajectory arc length.
    e_g : float
        Expansion rate :math:`E_G`.
    r0 : float, default = 1.0
        Initial tube radius.

    Returns
    -------
    s : float or ndarray
        Growth stage :math:`s`.

    See Also
    --------
    l_g : Inverse function (growth stage to arc length).

    References
    ----------
    .. [Noshita_2014] Noshita, K., 2014. Quantification and geometric analysis of
       coiling patterns in gastropod shells based on 3D and 2D image data.
       Journal of Theoretical Biology 363, 93–104.
    """
    if not r0 > 0.0:
        raise ValueError(f"r0 must be > 0, got {r0}")
    l_g = np.asarray(l_g, dtype=float)
    if e_g == 0.0:
        out = l_g / r0
    else:
        out = (1.0 / e_g) * np.log1p(e_g * l_g / r0)
    return float(out) if out.ndim == 0 else out


class GrowingTubeModel(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
):
    """Growing tube model.

    The growing tube model [Okamoto_1988]_. ``inverse_transform`` is the
    generative map ``Phi: (e_g, c_g, t_g, delta_g, gamma_g) -> form``.
    ``transform`` (parameter estimation from observed shells) is reserved for a
    later release.

    Parameters
    ----------
    r0 : float, default = 1.0
        Initial tube radius (scale) used for generation.
    method : {"ode", "closed"}, default = "ode"
        Forward solver passed to :func:`growing_tube`.
    estimator : str, default = "nls_3d"
        Fitting method used by ``transform`` (not yet implemented).
    n_jobs : int, optional
        Reserved for parallelism.
    verbose : int, default = 0
        Verbosity level.

    References
    ----------
    .. [Okamoto_1988] Okamoto, T., 1988. Analysis of heteromorph ammonoids by
      differential geometry. Palaeontology 31, 35–52.
    """

    def __init__(
        self,
        r0: float = 1.0,
        method: str = "ode",
        estimator: str = "nls_3d",
        n_jobs: int | None = None,
        verbose: int = 0,
    ):
        self.r0 = r0
        self.method = method
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y=None):
        """No-op (stateless). Returns self."""
        return self

    def transform(self, X, thickness=None):
        """Estimate ``(e_g, c_g, t_g)`` from observed shells (not implemented)."""
        raise NotImplementedError(
            "GrowingTubeModel.transform (parameter estimation) is not implemented yet"
        )

    def inverse_transform(
        self,
        X_transformed,
        s_range=None,
        phi_range=None,
        aperture=None,
        as_frame=False,
    ):
        """Generate shell surfaces from growing tube parameters.

        Parameters
        ----------
        X_transformed : array-like of shape (n_samples, 5) or (5,)
            Rows of ``(e_g, c_g, t_g, delta_g, gamma_g)``. A 3-column input
            ``(e_g, c_g, t_g)`` is also accepted, with orientation defaulted to 0.
            ``e_g`` is the logarithm of the original :math:`E` described in
            [Okamoto_1988]_ ([Noshita_2014]_).
        s_range, phi_range : array-like, optional
            Sampling grids. See :func:`growing_tube`.
        aperture : None
            Aperture shape; only the circular default is supported.
        as_frame : bool, default = False
            If True, return a long-format ``pandas.DataFrame``.

        Returns
        -------
        X : ndarray of shape (n_samples, n_s, n_phi, 3) or pd.DataFrame

        References
        ----------
        .. [Okamoto_1988] Okamoto, T., 1988. Analysis of heteromorph ammonoids by
            differential geometry. Palaeontology 31, 35–52.
        .. [Noshita_2014] Noshita, K., 2014. Quantification and geometric analysis of
            coiling patterns in gastropod shells based on 3D and 2D image data.
            Journal of Theoretical Biology 363, 93–104.
        """
        params = np.atleast_2d(np.asarray(X_transformed, dtype=float))
        single = np.ndim(X_transformed) == 1
        params = _pad_orientation(params)

        surfaces = [
            _growing_tube_surface(
                e_g,
                c_g,
                t_g,
                delta_g,
                gamma_g,
                r0=self.r0,
                s_range=s_range,
                phi_range=phi_range,
                aperture=aperture,
                method=self.method,
            )
            for e_g, c_g, t_g, delta_g, gamma_g in params
        ]

        if as_frame:
            return _surfaces_to_frame(surfaces)
        X = np.stack(surfaces)
        return X[0] if single else X

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Parameter names ``(e_g, c_g, t_g, delta_g, gamma_g)``."""
        return np.asarray(["e_g", "c_g", "t_g", "delta_g", "gamma_g"], dtype=str)
