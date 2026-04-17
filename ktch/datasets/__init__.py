"""
The :mod:`ktch.datasets` module implements utility functions
to load morphometric datasets and fetch example data files.
"""

import warnings

from ._base import (
    load_image_passiflora_leaves,
    load_landmark_mosquito_wings,
    load_landmark_trilobite_cephala,
    load_outline_leaf_bending,
    load_outline_mosquito_wings,
    load_surface_leaf_bending,
)
from ._examples import fetch
from ._sample_generator import (
    make_landmarks_from_reference,
)

__all__ = [
    "fetch",
    "load_landmark_mosquito_wings",
    "load_landmark_trilobite_cephala",
    "load_outline_mosquito_wings",
    "load_outline_leaf_bending",
    "load_image_passiflora_leaves",
    "load_surface_leaf_bending",
    "make_landmarks_from_reference",
]

_DEPRECATED_NAMES = {
    "convert_coords_df_to_list": "ktch.io.convert_coords_df_to_list",
    "convert_coords_df_to_df_sklearn_transform": "ktch.io.convert_coords_df_to_df_sklearn_transform",
}


def __getattr__(name):
    if name in _DEPRECATED_NAMES:
        new_path = _DEPRECATED_NAMES[name]
        warnings.warn(
            f"ktch.datasets.{name} is deprecated. "
            f"Use {new_path} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from ktch.io import _converters

        return getattr(_converters, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
