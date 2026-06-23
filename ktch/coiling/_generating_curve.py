"""Utility functions for generating curves, surface assemblies, and sampling grids."""

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
import pandas as pd


def _aperture_plane_basis(
    gamma_g: float, delta_g: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    r"""Orthonormal basis ``(u, v)`` of the aperture plane, in frame coordinates.

    The aperture-plane normal in frame coordinates :math:`(\xi_1, \xi_2, \xi_3)`
    is :math:`n = (\sqrt{1 - \sin^2\gamma_g - \sin^2\delta_g},\ \sin\gamma_g,\
    -\sin\delta_g)`. The signs and the positive :math:`\xi_1` root are fixed
    against the growing-tube ODE frame (the Raup aperture normal projected onto
    it); see the test suite. ``(gamma_g, delta_g) = (0, 0)`` gives
    :math:`n = \xi_1` (aperture perpendicular to the tangent), :math:`u = \xi_2`,
    :math:`v = \xi_3`.

    Returns
    -------
    u, v : ndarray of shape (3,)
        Frame-coordinate basis vectors spanning the plane perpendicular to ``n``.
    """
    sg, sd = np.sin(gamma_g), np.sin(delta_g)
    n1 = np.sqrt(max(1.0 - sg**2 - sd**2, 0.0))
    n = np.array([n1, sg, -sd])
    e2 = np.array([0.0, 1.0, 0.0])
    u = e2 - np.dot(e2, n) * n
    if np.linalg.norm(u) < 1e-8:
        e3 = np.array([0.0, 0.0, 1.0])
        u = e3 - np.dot(e3, n) * n
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def _assemble_surface(
    trajectory: npt.NDArray[np.float64],
    frames: npt.NDArray[np.float64],
    radius: npt.NDArray[np.float64],
    aperture=None,
    orientation: tuple[float, float] = (0.0, 0.0),
    phi_range: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    r"""Sweep a generating curve along a trajectory to build a tube surface.

    Implements :math:`u(s, \phi) = p(s) + q(s, \phi)` [Noshita_2016]_ with a
    circular generating curve. ``orientation = (0, 0)`` gives a circle
    perpendicular to the trajectory tangent; a non-zero orientation tilts the
    aperture plane in the frame (introducing a non-zero :math:`\xi_1` component).

    Parameters
    ----------
    trajectory : ndarray of shape (n, 3)
        Points on the reference-point trajectory :math:`p(s)` (the generating
        spiral, for constant parameters). The reference point need not be the
        tube centre; the choice is part of the model.
    frames : ndarray of shape (n, 3, 3)
        Orthonormal frame at each trajectory point. ``frames[i]`` holds the rows
        :math:`(\xi_1, \xi_2, \xi_3)` (tangent, normal, binormal).
    radius : ndarray of shape (n,)
        Tube radius :math:`r(s)` at each trajectory point.
    aperture : None
        Generating curve shape.
    orientation : tuple of float, default = (0.0, 0.0)
        Aperture orientation ``(gamma_g, delta_g)`` in the frame.
    phi_range : array-like of shape (n_phi,), optional
        Aperture-angle samples. Defaults to ``np.linspace(0, 2*pi, 90)``.

    Returns
    -------
    X : ndarray of shape (n, n_phi, 3)
        Surface coordinates.

    References
    ----------
    .. [Noshita_2016] Noshita, K., Shimizu, K., Sasaki, T., 2016. Geometric
       analysis and estimation of the growth rate gradient on gastropod shells.
       Journal of Theoretical Biology 389, 11–19.
    """
    if aperture is not None:
        raise NotImplementedError(
            "general aperture shapes are not supported yet; pass aperture=None "
            "for the default circular generating curve"
        )

    trajectory = np.asarray(trajectory, dtype=float)
    frames = np.asarray(frames, dtype=float)
    radius = np.asarray(radius, dtype=float)
    if phi_range is None:
        phi_range = np.linspace(0.0, 2.0 * np.pi, 90)
    phi_range = np.asarray(phi_range, dtype=float)

    u, v = _aperture_plane_basis(*orientation)
    cos_p = np.cos(phi_range)[:, None]
    sin_p = np.sin(phi_range)[:, None]
    q_frame = cos_p * u[None, :] + sin_p * v[None, :]  # (n_phi, 3) in frame coords
    # Express frame-coordinate vectors in global coordinates per trajectory point.
    q_global = np.einsum("jk,ikl->ijl", q_frame, frames)  # (n, n_phi, 3)
    X = trajectory[:, None, :] + radius[:, None, None] * q_global
    return X


def whorl_theta_range(
    n_whorls: float, n_points_per_whorl: int = 100, start: float = 0.0
) -> npt.NDArray[np.float64]:
    """Coiling angle samples spanning ``n_whorls`` revolutions (Raup's model).

    Parameters
    ----------
    n_whorls : float
        Number of revolutions to span.
    n_points_per_whorl : int, default = 100
        Samples per revolution.
    start : float, default = 0.0
        Starting coiling angle in radians.

    Returns
    -------
    theta_range : ndarray
        Coiling-angle values in radians.
    """
    n = max(int(round(n_whorls * n_points_per_whorl)), 2)
    return np.linspace(start, start + 2.0 * np.pi * n_whorls, n)


def whorl_s_range(
    n_whorls: float, c_g: float, t_g: float, n_points_per_whorl: int = 100
) -> npt.NDArray[np.float64]:
    """Growth stage samples spanning ``n_whorls`` revolutions (growing tube model).

    One revolution of the local frame advances ``s`` by ``2*pi / D_G`` with
    ``D_G = sqrt(c_g**2 + t_g**2)``. Constant-parameter models only.

    Parameters
    ----------
    n_whorls : float
        Number of revolutions to span.
    c_g, t_g : float
        Standardized curvature and torsion (``D_G`` is computed from these).
    n_points_per_whorl : int, default = 100
        Samples per revolution.

    Returns
    -------
    s_range : ndarray
        Growth-stage values.

    Raises
    ------
    ValueError
        If ``D_G`` is zero (a straight tube has no coiling to count).
    """
    d_g = np.hypot(c_g, t_g)
    if d_g == 0.0:
        raise ValueError(
            "D_G = sqrt(c_g**2 + t_g**2) is zero; a straight tube has no whorls"
        )
    n = max(int(round(n_whorls * n_points_per_whorl)), 2)
    s_max = 2.0 * np.pi * n_whorls / d_g
    return np.linspace(0.0, s_max, n)


def _pad_orientation(params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Accept a 3-column (trajectory only) or 5-column (full) parameter array.

    A 3-column input has its two orientation columns appended as zeros.
    """
    ncol = params.shape[1]
    if ncol == 5:
        return params
    if ncol == 3:
        return np.column_stack([params, np.zeros((len(params), 2))])
    raise ValueError(f"expected 3 (trajectory) or 5 (full) columns, got {ncol}")


def _surfaces_to_frame(surfaces) -> pd.DataFrame:
    """Stack per-sample ``(n_traj, n_phi, 3)`` surfaces into a long DataFrame.

    The result is indexed by ``(specimen_id, trajectory_id, phi_id)`` with columns
    ``x, y, z``.
    """
    frames = []
    for sample_id, X in enumerate(surfaces):
        n_traj, n_phi, _ = X.shape
        trajectory_id, phi_id = np.meshgrid(
            np.arange(n_traj), np.arange(n_phi), indexing="ij"
        )
        df = pd.DataFrame(
            {
                "specimen_id": sample_id,
                "trajectory_id": trajectory_id.ravel(),
                "phi_id": phi_id.ravel(),
                "x": X[:, :, 0].ravel(),
                "y": X[:, :, 1].ravel(),
                "z": X[:, :, 2].ravel(),
            }
        )
        frames.append(df)
    return pd.concat(frames).set_index(["specimen_id", "trajectory_id", "phi_id"])
