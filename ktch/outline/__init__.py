"""
The :mod:`katatch.outline` module implements outline-based morphometrics.

The outline, which are boundary lines or boundary surfaces separating
the target object from others, are modeled as closed functions.
"""

# Copyright 2020 Koji Noshita
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

# from ._polar_Fourier_analysis import PFA
from ._elliptic_Fourier_analysis import EllipticFourierAnalysis, rotation_matrix_2d
from ._spherical_harmonic_analysis import spharm, SPHARMCoefficients

# from ._spherical_harmonic_analysis import SphericalHarmonicAnalysis, PCContribDisplay

__all__ = [
    "EllipticFourierAnalysis",
    "rotation_matrix_2d",
    # "SphericalHarmonicAnalysis",
    # "PCContribDisplay",
    "spharm",
    "SPHARMCoefficients",
]
