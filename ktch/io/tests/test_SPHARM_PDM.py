import ast
from pathlib import Path

import pytest
from numpy.testing import assert_array_almost_equal

from ktch.io import read_coef_SPHARM_PDM


def test_read_coef_SPHARM_PDM_m_0():
    path = Path(__file__).parent / "data" / "andesred_07_allSegments_SPHARM.coef"

    with open(path, "r") as f:
        coef_txt = f.read()
    coef_list_ = ast.literal_eval(coef_txt.replace("{", "[").replace("}", "]"))
    coef_list = coef_list_[1:]

    l_max = int((coef_list_[0]) ** 0.5 - 1)

    coef_x, coef_y, coef_z = read_coef_SPHARM_PDM(path)

    assert coef_x.n_degree == l_max
    assert coef_y.n_degree == l_max
    assert coef_z.n_degree == l_max

    assert coef_x[0] == coef_list[0][0]
    assert coef_y[0] == coef_list[0][1]
    assert coef_z[0] == coef_list[0][2]
    for l in range(1, l_max + 1):
        assert coef_x[l, 0] == coef_list[l**2][0]
        assert coef_y[l, 0] == coef_list[l**2][1]
        assert coef_z[l, 0] == coef_list[l**2][2]
