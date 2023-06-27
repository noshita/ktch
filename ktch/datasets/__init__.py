"""
The :mod:`ktch.datasets` module implements utility functions
to load morphometric datasets.
"""

from ._base import (
    load_landmark_mosquito_wings,
    load_outline_mosquito_wings,
    load_outline_bottles,
    load_coefficient_bottles,
    convert_coords_df_to_list,
    convert_coords_df_to_df_sklearn_transform,
)

from ._sample_generator import (
    make_landmarks_from_reference,
)

__all__ = [
    "load_landmark_mosquito_wings",
    "load_outline_mosquito_wings",
    "load_outline_bottles",
    "load_coefficient_bottles",
    "convert_coords_df_to_list",
    "convert_coords_df_to_df_sklearn_transform",
    "make_landmarks_from_reference",
]
