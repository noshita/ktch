"""SPHARM-PDM file I/O functions"""

# Copyright 2023 Koji Noshita
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

from ..outline import SPHARMCoefficients


def read_coef_SPHARM_PDM(path):
    """Read .coef file of SPHARM-PDM.
    .coef file, which is an output of the `ParaToSPHARMMesh` step
    of `SPHARM-PDM <https://www.nitrc.org/projects/spharm-pdm>`_,
    contains SPHARM coefficients.

    Parameters
    ----------
    path : str/pathlib.Path
        Path to the .coef file.

    Returns
    -------
    coef : list of float
        List of coefficients.
    """
    with open(path, "r") as f:
        coef_txt = f.read()
    coef = [
        float(val)
        for val in coef_txt.replace("\n", "")
        .replace("{", "")
        .replace("}", "")
        .split(",")
    ]

    coef_lists = _cvt_spharm_coef_SPHARMPDM_to_list(np.array(coef[1:]))

    coefficients = [
        (
            [
                [coef_lists[l][m_, i] for m_ in range(2 * l + 1)]
                for l in range(len(coef_lists))
            ]
        )
        for i in range(3)
    ]

    coef_x = SPHARMCoefficients()
    coef_y = SPHARMCoefficients()
    coef_z = SPHARMCoefficients()
    coef_x.from_list(coefficients[0])
    coef_y.from_list(coefficients[1])
    coef_z.from_list(coefficients[2])

    return coef_x, coef_y, coef_z


def _cvt_spharm_coef_SPHARMPDM_to_list(coef_SlicerSALT):
    coef_ = coef_SlicerSALT.reshape((-1, 3))
    lmax = int(np.sqrt(coef_.shape)[0] - 1)
    coef_list = [
        np.array(
            [
                coef_[l**2]
                if m == 0
                else (coef_[l**2 + 2 * m - 1] - coef_[l**2 + 2 * m] * 1j) / 2
                if m > 0
                else ((-1) ** m)
                * (
                    coef_[l**2 + 2 * np.abs(m) - 1]
                    + coef_[l**2 + 2 * np.abs(m)] * 1j
                )
                / 2
                for m in range(-l, l + 1, 1)
            ]
        )
        for l in range(0, lmax + 1, 1)
    ]
    return coef_list


def _cvt_spharm_coef_list_to_SPHARM_PDM(coef_list):
    lmax = len(coef_list) - 1
    coef_SlicerSALT = np.zeros(((lmax + 1) ** 2, 3))
    for l in range(0, lmax + 1, 1):
        for m in range(0, l + 1, 1):
            if m == 0:
                coef_SlicerSALT[l**2] = coef_list[l][m + l].real
            else:
                coef_SlicerSALT[l**2 + 2 * m - 1] = (
                    coef_list[l][m + l] + ((-1) ** m) * coef_list[l][-m + l]
                ).real
                coef_SlicerSALT[l**2 + 2 * m] = (
                    (coef_list[l][m + l] - ((-1) ** m) * coef_list[l][-m + l]) * 1j
                ).real

    return coef_SlicerSALT.reshape(-1)
