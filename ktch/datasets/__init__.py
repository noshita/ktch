"""
The :mod:`ktch.datasets` module implements utility functions
to load morphometric datasets.
"""

from ._base import (
    load_landmark_mosquito_wings,
    load_outline_mosquito_wings,
    load_outline_bottles,
    load_coefficient_bottles,
)

__all__ = [
    "load_landmark_mosquito_wings",
    "load_outline_mosquito_wings",
    "load_outline_bottles",
    "load_coefficient_bottles",
]
