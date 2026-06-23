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

import numpy as np
import numpy.typing as npt
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
)

from ._generating_curve import _pad_orientation, _surfaces_to_frame, whorl_theta_range


def _validate_raup_params(w_r: float, t_r: float, d_r: float, r0: float) -> None:
    if not w_r > 1.0:
        raise ValueError(f"w_r (whorl expansion rate) must be > 1, got {w_r}")
    if not (0.0 <= d_r < 1.0):
        raise ValueError(f"d_r must be in [0, 1), got {d_r}")
    if not r0 > 0.0:
        raise ValueError(f"r0 must be > 0, got {r0}")


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

    Raup's model [Raup_1965]_ [Raup_1967]_, built in the global frame
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
    .. [Raup_1967] Raup, D.M., 1967. Geometric analysis of shell coiling: coiling in
       ammonoids. Journal of Paleontology 41, 43–65.
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

    Raup’s logarithmic shell coiling model [Raup_1965]_ [Raup_1967]_ describes a shell
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
    .. [Raup_1967] Raup, D.M., 1967. Geometric analysis of shell coiling: coiling in
       ammonoids. Journal of Paleontology 41, 43–65.
    """
    if output != "surface":
        raise NotImplementedError(
            f"output={output!r} is reserved; only 'surface' is implemented"
        )
    return _raup_surface(
        w_r, t_r, d_r, delta_r, gamma_r, r0, theta_range, phi_range, aperture
    )


class RaupModel(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Raup's model.

    Raup’s logarithmic shell coiling model [Raup_1965]_ [Raup_1967]_.
    ``inverse_transform`` is the generative map
    ``Phi: (w_r, t_r, d_r, delta_r, gamma_r) -> form``.
    ``transform`` (parameter estimation from measurement data) is not implemented
    yet.

    Parameters
    ----------
    r0 : float, default = 1.0
        Initial tube radius (scale) used for generation.
    estimator : str, default = "ml_2d"
        Fitting method used by ``transform`` (not yet implemented).
    n_jobs : int, optional
        Reserved for parallelism.
    verbose : int, default = 0
        Verbosity level.

    References
    ----------
    .. [Raup_1965] Raup, D.M., Michelson, A., 1965. Theoretical Morphology of
       the Coiled Shell. Science 147, 1294–1295.
    .. [Raup_1967] Raup, D.M., 1967. Geometric analysis of shell coiling: coiling in
       ammonoids. Journal of Paleontology 41, 43–65.
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

    def transform(self, X, d=None, f=None, h=None, b=None, c=None):
        """Estimate Raup parameters from observed shells (not implemented)."""
        raise NotImplementedError(
            "RaupModel.transform (parameter estimation) is not implemented yet"
        )

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
