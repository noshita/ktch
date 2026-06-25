"""Parameter conversion between theoretical morphological models."""

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
from scipy.optimize import least_squares

from ._raup import _raup_discriminant


def raup_to_growing_tube(
    w_r: float, t_r: float, d_r: float
) -> tuple[float, float, float]:
    r"""Convert Raup parameters to growing-tube parameters.

    Raup's model is the constant-parameter special case of the growing tube
    model [Noshita_2014]_. The closed forms here are re-derived from
    the standardized curvature/torsion of the reference-point trajectory (a
    logarithmic conical helix).

    Parameters
    ----------
    w_r, t_r, d_r : float
        Raup whorl expansion rate, translation rate, and generating-curve
        position. ``w_r > 1`` and ``-1 < d_r < 1``.

    Returns
    -------
    e_g, c_g, t_g : float
        Expansion rate, standardized curvature, and torsion.

    Notes
    -----
    Correct two errors in [Noshita_2014]_:
    (the ``4*t_r`` term in the trajectory discriminant :math:`\Lambda` should be
    ``4*t_r**2``; the ``(1-d_r)**2`` factor in ``c_g`` should be ``(1-d_r**2)``).
    See :func:`~ktch.coiling._raup._raup_discriminant`.

    References
    ----------
    .. [Noshita_2014] Noshita, K., 2014. Quantification and geometric analysis of
       coiling patterns in gastropod shells based on 3D and 2D image data.
       Journal of Theoretical Biology 363, 93–104.
    """
    if not w_r > 1.0:
        raise ValueError(f"w_r must be > 1, got {w_r}")
    if not (-1.0 < d_r < 1.0):
        raise ValueError(f"d_r must be in (-1, 1), got {d_r}")

    log_w = np.log(w_r)
    one_m_d = 1.0 - d_r
    disc = _raup_discriminant(w_r, t_r, d_r)

    e_g = one_m_d * log_w / np.sqrt(disc)
    c_g = 2.0 * np.pi * (1.0 - d_r**2) * np.sqrt(4.0 * np.pi**2 + log_w**2) / disc
    t_g = 4.0 * np.pi * t_r * one_m_d * log_w / disc
    return float(e_g), float(c_g), float(t_g)


def growing_tube_to_raup(
    e_g: float, c_g: float, t_g: float
) -> tuple[float, float, float]:
    """Convert growing tube model parameters to Raup' model parameters.

    Numerical inverse of :func:`raup_to_growing_tube` (Eq. 18a-c is not
    analytically invertible in closed form here).

    Parameters
    ----------
    e_g, c_g, t_g : float
        Expansion rate, standardized curvature, and torsion. ``c_g > 0``.

    Returns
    -------
    w_r, t_r, d_r : float
        Raup parameters.

    Raises
    ------
    ValueError
        If ``c_g == 0`` (the coiling axis, and hence ``w_r``/``d_r``, is
        undefined for a straight tube).
    """
    if c_g <= 0.0:
        raise ValueError(
            "c_g must be > 0; the coiling axis is undefined for a straight tube"
        )

    target = np.array([e_g, c_g, t_g], dtype=float)

    def residual(x):
        w_r, t_r, d_r = x
        return np.array(raup_to_growing_tube(w_r, t_r, d_r)) - target

    eps = 1e-9
    sol = least_squares(
        residual,
        x0=np.array([1.5, 1.0, 0.5]),
        bounds=(
            np.array([1.0 + eps, -np.inf, -1.0 + eps]),
            np.array([np.inf, np.inf, 1.0 - eps]),
        ),
        xtol=1e-12,
        ftol=1e-12,
    )
    w_r, t_r, d_r = sol.x
    return float(w_r), float(t_r), float(d_r)


def raup_aperture_to_growing_tube(
    w_r: float, t_r: float, d_r: float, delta_r: float, gamma_r: float
) -> tuple[float, float]:
    r"""Convert Raup's aperture orientation to growing tube (frame) orientation.

    The Raup's aperture plane normal, projected onto the Frenet frame
    :math:`(\xi_1, \xi_2, \xi_3)` of the (shared) trajectory, gives
    ``(delta_g, gamma_g)``.

    Parameters
    ----------
    w_r, t_r, d_r : float
        Raup's model parameters.
    delta_r, gamma_r : float
        Raup's aperture orientation :math:`(\Delta, \Gamma)`.

    Returns
    -------
    delta_g, gamma_g : float
        Growing tube (Frenet-frame) aperture orientation.
    """
    if not w_r > 1.0:
        raise ValueError(f"w_r must be > 1, got {w_r}")
    log_w = np.log(w_r)
    four_pi2 = 4.0 * np.pi**2
    cd, sd = np.cos(delta_r), np.sin(delta_r)
    cg, sg = np.cos(gamma_r), np.sin(gamma_r)

    gamma_g = np.arcsin(
        cd * (log_w * cg + 2.0 * np.pi * sg) / np.sqrt(four_pi2 + log_w**2)
    )
    num = (
        2.0 * t_r * cd * log_w * (2.0 * np.pi * cg - log_w * sg)
        - (1.0 + d_r) * (four_pi2 + log_w**2) * sd
    )
    den = np.sqrt((four_pi2 + log_w**2) * _raup_discriminant(w_r, t_r, d_r))
    delta_g = np.arcsin(num / den)
    return float(delta_g), float(gamma_g)


def growing_tube_aperture_to_raup(
    w_r: float, t_r: float, d_r: float, gamma_g: float, delta_g: float
) -> tuple[float, float]:
    """Convert growing tube aperture orientation to Raup's orientation.

    Numerical inverse of :func:`raup_aperture_to_growing_tube` at the given
    trajectory parameters.

    Returns
    -------
    delta_r, gamma_r : float
        Raup's aperture orientation.
    """
    target = np.array([delta_g, gamma_g], dtype=float)

    def residual(x):
        d_r_, g_r_ = x
        return (
            np.array(raup_aperture_to_growing_tube(w_r, t_r, d_r, d_r_, g_r_)) - target
        )

    sol = least_squares(
        residual,
        x0=np.array([0.0, 0.0]),
        bounds=(np.array([-np.pi / 2, -np.pi / 2]), np.array([np.pi / 2, np.pi / 2])),
        xtol=1e-12,
        ftol=1e-12,
    )
    delta_r, gamma_r = sol.x
    return float(delta_r), float(gamma_r)
