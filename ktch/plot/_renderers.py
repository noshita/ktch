"""Built-in shape renderers for morphometric visualization."""

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

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

_EXPLICIT_RENDER_KEYS = frozenset({"color", "alpha", "links"})


def _resolve_render_kw(
    render_kw: dict[str, Any],
    *,
    color: str,
    alpha: float,
    links: Any,
) -> dict[str, Any]:
    """Merge explicit params with render_kw. Explicit params take precedence."""
    resolved = dict(render_kw)
    conflicts = _EXPLICIT_RENDER_KEYS & resolved.keys()
    if conflicts:
        warnings.warn(
            f"{conflicts} in render_kw ignored; use the explicit parameter(s) instead",
            UserWarning,
            stacklevel=3,
        )
        for key in conflicts:
            del resolved[key]
    resolved["color"] = color
    resolved["alpha"] = alpha
    resolved["links"] = links
    return resolved


def _render_curve_2d(coords, ax, *, color="gray", alpha=1.0, **kw):
    """Render a 2D curve (closed or open).

    Parameters
    ----------
    coords : np.ndarray of shape (t, 2+)
        Curve coordinates. Only the first 2 columns are used.
    ax : matplotlib.axes.Axes
        Target axes (2D).
    color : str
        Line color.
    alpha : float
        Line transparency.
    **kw
        Forwarded to ``ax.plot``.
    """
    kw.pop("links", None)
    ax.plot(coords[:, 0], coords[:, 1], color=color, alpha=alpha, **kw)
    ax.set_aspect("equal")


def _render_curve_3d(coords, ax, *, color="gray", alpha=1.0, **kw):
    """Render a 3D curve.

    Parameters
    ----------
    coords : np.ndarray of shape (t, 3)
        Curve coordinates.
    ax : matplotlib.axes.Axes
        Target axes (3D projection).
    color : str
        Line color.
    alpha : float
        Line transparency.
    **kw
        Forwarded to ``ax.plot``.
    """
    kw.pop("links", None)
    ax.plot(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        color=color,
        alpha=alpha,
        **kw,
    )
    ptp = [np.max(coords[:, i]) - np.min(coords[:, i]) for i in range(3)]
    # Avoid zero-extent axes
    ptp = [max(p, 1e-10) for p in ptp]
    ax.set_box_aspect(ptp)


def _render_surface_3d(coords, ax, *, color="orange", alpha=1.0, **kw):
    """Render a 3D surface from a structured grid.

    Parameters
    ----------
    coords : np.ndarray of shape (m, n, 3)
        Structured grid surface coordinates.
    ax : matplotlib.axes.Axes
        Target axes (3D projection).
    color : str
        Surface color.
    alpha : float
        Surface transparency.
    **kw
        Forwarded to ``ax.plot_surface``.
    """
    kw.pop("links", None)
    X, Y, Z = coords[..., 0], coords[..., 1], coords[..., 2]
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, **kw)
    ptp = [np.max(X) - np.min(X), np.max(Y) - np.min(Y), np.max(Z) - np.min(Z)]
    ptp = [max(p, 1e-10) for p in ptp]
    ax.set_box_aspect(ptp)


def _render_landmarks_2d(coords, ax, *, color="gray", alpha=1.0, links=None, **kw):
    """Render 2D landmarks with optional link segments.

    Parameters
    ----------
    coords : np.ndarray of shape (n_lm, 2+)
        Landmark coordinates. Only the first 2 columns are used.
    ax : matplotlib.axes.Axes
        Target axes (2D).
    color : str
        Point and link color.
    alpha : float
        Transparency.
    links : sequence of sequence of int, optional
        Pairs of landmark indices to connect with line segments.
    **kw
        Forwarded to ``ax.scatter`` (except ``s`` which defaults to 10).
    """
    from matplotlib.collections import LineCollection

    x, y = coords[:, 0], coords[:, 1]
    if links:
        pts = coords[:, :2]
        segments = [
            pts[list(link)] for link in links if all(i < len(pts) for i in link)
        ]
        if segments:
            lc = LineCollection(segments, colors=color, alpha=alpha)
            ax.add_collection(lc)
    s = kw.pop("s", 10)
    ax.scatter(x, y, c=color, alpha=alpha, s=s, **kw)
    ax.set_aspect("equal")


def _render_landmarks_3d(coords, ax, *, color="gray", alpha=1.0, links=None, **kw):
    """Render 3D landmarks with optional link segments.

    Parameters
    ----------
    coords : np.ndarray of shape (n_lm, 3)
        Landmark coordinates.
    ax : matplotlib.axes.Axes
        Target axes (3D projection).
    color : str
        Point and link color.
    alpha : float
        Transparency.
    links : sequence of sequence of int, optional
        Pairs of landmark indices to connect with line segments.
    **kw
        Forwarded to ``ax.scatter`` (except ``s`` which defaults to 10).
    """
    if links:
        for link in links:
            if all(i < len(coords) for i in link):
                pts = coords[list(link)]
                ax.plot(
                    pts[:, 0],
                    pts[:, 1],
                    pts[:, 2],
                    color=color,
                    alpha=alpha,
                )
    s = kw.pop("s", 10)
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=color,
        alpha=alpha,
        s=s,
        **kw,
    )
    ptp = [np.max(coords[:, i]) - np.min(coords[:, i]) for i in range(3)]
    ptp = [max(p, 1e-10) for p in ptp]
    ax.set_box_aspect(ptp)


# Maps shape_type to (renderer, projection)
_SHAPE_TYPE_REGISTRY: dict[str, tuple[Any, str | None]] = {
    "curve_2d": (_render_curve_2d, None),
    "curve_3d": (_render_curve_3d, "3d"),
    "surface_3d": (_render_surface_3d, "3d"),
    "landmarks_2d": (_render_landmarks_2d, None),
    "landmarks_3d": (_render_landmarks_3d, "3d"),
}

VALID_SHAPE_TYPES = frozenset(_SHAPE_TYPE_REGISTRY.keys()) | {"auto"}
