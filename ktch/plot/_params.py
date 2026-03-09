"""Shared parameter resolution for plot functions."""

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

from collections.abc import Callable
from typing import Any

import numpy as np

from ._renderers import _SHAPE_TYPE_REGISTRY, VALID_SHAPE_TYPES


def _resolve_reducer_params(
    reducer: Any | None,
    reducer_inverse_transform: Callable | None,
    explained_variance: np.ndarray | None,
    n_components: int | None,
    *,
    require_variance: bool = True,
) -> tuple[Callable, np.ndarray | None, int]:
    """Resolve reducer convenience parameter into explicit values.

    Parameters
    ----------
    reducer : estimator or None
        PCA-compatible fitted estimator.
    reducer_inverse_transform : callable or None
        Explicit inverse_transform override.
    explained_variance : ndarray or None
        Explicit explained_variance override.
    n_components : int or None
        Explicit n_components override.
    require_variance : bool
        If True, raise ValueError when explained_variance cannot be resolved.

    Returns
    -------
    reducer_inverse_transform : callable
    explained_variance : ndarray or None
    n_components : int
    """
    if reducer is not None:
        if reducer_inverse_transform is None:
            reducer_inverse_transform = reducer.inverse_transform
        if explained_variance is None:
            explained_variance = getattr(reducer, "explained_variance_", None)
        if n_components is None:
            n_components = getattr(
                reducer,
                "n_components_",
                getattr(reducer, "n_components", None),
            )

    if reducer_inverse_transform is None:
        raise ValueError("Either reducer or reducer_inverse_transform must be provided")
    if require_variance and explained_variance is None:
        raise ValueError("Either reducer or explained_variance must be provided")
    if n_components is None:
        raise ValueError("Either reducer or n_components must be provided")

    return reducer_inverse_transform, explained_variance, n_components


def _resolve_descriptor_params(
    descriptor: Any | None,
    descriptor_inverse_transform: Callable | None,
    n_dim: int | None,
    shape_type: str,
) -> tuple[Callable | None, int | None]:
    """Resolve descriptor convenience parameter.

    Parameters
    ----------
    descriptor : estimator or None
        Fitted shape descriptor (EFA, SHA, etc.).
    descriptor_inverse_transform : callable or None
        Explicit inverse_transform override.
    n_dim : int or None
        Spatial dimensionality (for GPA identity case).
    shape_type : str
        Shape type string (may be "auto").

    Returns
    -------
    descriptor_inverse_transform : callable or None
    n_dim : int or None
    """
    if descriptor is not None and descriptor_inverse_transform is None:
        descriptor_inverse_transform = descriptor.inverse_transform

    if descriptor_inverse_transform is None:
        # Identity (GPA) case: n_dim required unless inferable from shape_type
        if n_dim is None:
            if shape_type == "landmarks_2d":
                n_dim = 2
            elif shape_type == "landmarks_3d":
                n_dim = 3
            else:
                raise ValueError(
                    "n_dim is required when descriptor is not provided "
                    "and shape_type is not an explicit landmarks type"
                )

    return descriptor_inverse_transform, n_dim


def _detect_shape_type(
    sample_coords: np.ndarray,
    descriptor_inverse_transform: Callable | None,
    n_dim: int | None,
) -> str:
    """Auto-detect shape_type from a sample output array.

    Parameters
    ----------
    sample_coords : ndarray
        A single reconstructed shape (no batch dim).
    descriptor_inverse_transform : callable or None
        If None, the identity (GPA) case is assumed.
    n_dim : int or None
        Spatial dimensionality.

    Returns
    -------
    shape_type : str
    """
    if descriptor_inverse_transform is not None:
        ndim = sample_coords.ndim
        if ndim == 3:
            # (m, n, 3) -> surface
            return "surface_3d"
        elif ndim == 2:
            last = sample_coords.shape[-1]
            if last == 2:
                return "curve_2d"
            elif last >= 3:
                return "curve_3d"
    else:
        # GPA identity case
        if n_dim == 2:
            return "landmarks_2d"
        elif n_dim == 3:
            return "landmarks_3d"

    raise ValueError(
        "Cannot auto-detect shape_type. Please specify shape_type explicitly."
    )


def _get_renderer_and_projection(
    shape_type: str,
    render_fn: Callable | None,
) -> tuple[Callable, str | None]:
    """Get renderer callable and matplotlib projection string.

    Parameters
    ----------
    shape_type : str
        Resolved shape type (not "auto").
    render_fn : callable or None
        Custom renderer override.

    Returns
    -------
    renderer : callable
    projection : str or None
    """
    if render_fn is not None:
        # Custom renderer: infer projection from shape_type
        if shape_type in _SHAPE_TYPE_REGISTRY:
            _, proj = _SHAPE_TYPE_REGISTRY[shape_type]
        else:
            proj = None
        return render_fn, proj

    if shape_type not in _SHAPE_TYPE_REGISTRY:
        raise ValueError(
            f"Invalid shape_type {shape_type!r}. "
            f"Valid options: {sorted(VALID_SHAPE_TYPES)}"
        )

    return _SHAPE_TYPE_REGISTRY[shape_type]


def _validate_components(
    components: Any,
    n_components: int,
) -> None:
    """Validate that component indices are within bounds.

    Parameters
    ----------
    components : sequence of int
        Component indices.
    n_components : int
        Total number of components.

    Raises
    ------
    ValueError
        If any index >= n_components.
    """
    for idx in components:
        if idx >= n_components:
            raise ValueError(
                f"Component index {idx} exceeds number of components ({n_components})"
            )
