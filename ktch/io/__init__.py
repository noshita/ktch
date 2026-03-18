"""
The :mod:`ktch.io` module implements I/O interface for morphometrics file formats.
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

from ._chc import ChainCodeData, read_chc, write_chc
from ._converters import (
    convert_coords_df_to_df_sklearn_transform,
    convert_coords_df_to_list,
    efa_coeffs_to_nef,
    nef_to_efa_coeffs,
)
from ._nef import NefData, read_nef, write_nef
from ._protocols import MorphoData, MorphoDataMixin
from ._spharm_pdm import (
    SpharmPdmData,
    cvt_spharm_coef_list_to_spharmpdm,
    cvt_spharm_coef_spharmpdm_to_list,
    read_spharmpdm_coef,
)
from ._tps import TPSData, read_tps, write_tps

__all__ = [
    # Protocols and mixins
    "MorphoData",
    "MorphoDataMixin",
    # Data containers
    "NefData",
    "ChainCodeData",
    "TPSData",
    "SpharmPdmData",
    # I/O functions
    "read_tps",
    "write_tps",
    "read_spharmpdm_coef",
    "cvt_spharm_coef_spharmpdm_to_list",
    "cvt_spharm_coef_list_to_spharmpdm",
    "read_chc",
    "write_chc",
    "read_nef",
    "write_nef",
    # Converter functions
    "nef_to_efa_coeffs",
    "efa_coeffs_to_nef",
    "convert_coords_df_to_list",
    "convert_coords_df_to_df_sklearn_transform",
]
