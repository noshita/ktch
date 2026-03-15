import ast
from pathlib import Path

from numpy.testing import assert_array_almost_equal

from ktch.io import (
    cvt_spharm_coef_list_to_spharmpdm,
    cvt_spharm_coef_spharmpdm_to_list,
    read_spharmpdm_coef,
)


def test_read_spharmpdm_coef_valid_shape():
    path = Path(__file__).parent / "data" / "andesred_07_allSegments_SPHARM.coef"

    coef_list = read_spharmpdm_coef(path)

    assert len(coef_list) > 0
    for l, coef in enumerate(coef_list):
        assert coef.shape == (2 * l + 1, 3)


def test_read_spharmpdm_coef_m_0():
    path = Path(__file__).parent / "data" / "andesred_07_allSegments_SPHARM.coef"

    with open(path, "r") as f:
        coef_txt = f.read()
    coef_list_ = ast.literal_eval(coef_txt.replace("{", "[").replace("}", "]"))
    coef_list_raw = coef_list_[1:]

    l_max = int((coef_list_[0]) ** 0.5 - 1)

    coef_list = read_spharmpdm_coef(path)

    assert len(coef_list) - 1 == l_max

    assert coef_list[0][0, 0] == coef_list_raw[0][0]
    assert coef_list[0][0, 1] == coef_list_raw[0][1]
    assert coef_list[0][0, 2] == coef_list_raw[0][2]
    for l in range(1, l_max + 1):
        assert_array_almost_equal(coef_list[l][l], coef_list_raw[l**2])


def test_spharm_coef_roundtrip_list():
    """list -> spharmpdm -> list produces identical coefficients."""
    path = Path(__file__).parent / "data" / "andesred_07_allSegments_SPHARM.coef"
    coef_list = read_spharmpdm_coef(path)

    coef_spharmpdm = cvt_spharm_coef_list_to_spharmpdm(coef_list)
    coef_list_rt = cvt_spharm_coef_spharmpdm_to_list(coef_spharmpdm)

    for l in range(len(coef_list)):
        assert_array_almost_equal(coef_list[l], coef_list_rt[l])


def test_spharm_coef_roundtrip_spharmpdm():
    """spharmpdm -> list -> spharmpdm produces identical array."""
    path = Path(__file__).parent / "data" / "andesred_07_allSegments_SPHARM.coef"
    coef_list = read_spharmpdm_coef(path)

    coef_pdm = cvt_spharm_coef_list_to_spharmpdm(coef_list)
    coef_list_rt = cvt_spharm_coef_spharmpdm_to_list(coef_pdm)
    coef_pdm_rt = cvt_spharm_coef_list_to_spharmpdm(coef_list_rt)

    assert_array_almost_equal(coef_pdm, coef_pdm_rt)
