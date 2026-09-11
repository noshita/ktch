import ast
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from ktch.io import (
    read_spharmpdm_coef,
    sha_coeffs_to_spharmpdm,
    spharmpdm_to_sha_coeffs,
    write_spharmpdm_coef,
)
from ktch.io._protocols import MorphoData
from ktch.io._spharm_pdm import SpharmPdmData


def test_read_spharmpdm_coef_returns_spharmpdm_data():
    path = Path(__file__).parent / "data" / "andesred_07_allSegments_SPHARM.coef"
    data = read_spharmpdm_coef(path)

    assert isinstance(data, SpharmPdmData)
    assert len(data.coeffs) > 0
    for l, coef in enumerate(data.coeffs):
        assert coef.shape == (2 * l + 1, 3)


def test_read_spharmpdm_coef_specimen_name():
    path = Path(__file__).parent / "data" / "andesred_07_allSegments_SPHARM.coef"
    data = read_spharmpdm_coef(path)

    assert data.specimen_name == "andesred_07_allSegments_SPHARM"


def test_read_spharmpdm_coef_m_0():
    path = Path(__file__).parent / "data" / "andesred_07_allSegments_SPHARM.coef"

    with open(path, "r") as f:
        coef_txt = f.read()
    coef_list_ = ast.literal_eval(coef_txt.replace("{", "[").replace("}", "]"))
    coef_list_raw = coef_list_[1:]

    l_max = int((coef_list_[0]) ** 0.5 - 1)

    data = read_spharmpdm_coef(path)

    assert data.l_max == l_max

    assert data.coeffs[0][0, 0] == coef_list_raw[0][0]
    assert data.coeffs[0][0, 1] == coef_list_raw[0][1]
    assert data.coeffs[0][0, 2] == coef_list_raw[0][2]
    for l in range(1, l_max + 1):
        assert_array_almost_equal(data.coeffs[l][l], coef_list_raw[l**2])


# --- SpharmPdmData protocol and methods ---


def test_spharmpdm_data_satisfies_protocol():
    data = SpharmPdmData(
        specimen_name="test",
        coeffs=[np.zeros((1, 3)), np.zeros((3, 3))],
    )
    assert isinstance(data, MorphoData)


def test_spharmpdm_data_to_numpy():
    coeffs = [np.ones((1, 3)), np.ones((3, 3)) * 2]
    data = SpharmPdmData(specimen_name="test", coeffs=coeffs)
    arr = data.to_numpy()

    assert arr.shape == (4, 3)  # (1+3, 3) = ((1+1)^2, 3) for lmax=1
    assert arr[0, 0] == 1.0
    assert arr[1, 0] == 2.0


def test_spharmpdm_data_array_protocol():
    coeffs = [np.ones((1, 3))]
    data = SpharmPdmData(specimen_name="test", coeffs=coeffs)
    arr = np.asarray(data)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1, 3)


def test_spharmpdm_data_to_dataframe():
    coeffs = [
        np.array([[1.0, 2.0, 3.0]]),
        np.array([[4, 5, 6], [7, 8, 9], [10, 11, 12]]),
    ]
    data = SpharmPdmData(specimen_name="S", coeffs=coeffs)
    df = data.to_dataframe()

    assert df.shape == (4, 3)
    assert list(df.columns) == ["x", "y", "z"]
    assert df.index.names == ["specimen_id", "l_m"]


def test_spharmpdm_data_repr():
    data = SpharmPdmData(
        specimen_name="test",
        coeffs=[np.zeros((1, 3)), np.zeros((3, 3))],
    )
    r = repr(data)
    assert "SpharmPdmData" in r
    assert "l_max=1" in r


# --- write_spharmpdm_coef ---

_COEF_PATH = Path(__file__).parent / "data" / "andesred_07_allSegments_SPHARM.coef"


def _coeffs_equal(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a.coeffs, b.coeffs))


def test_write_read_round_trip_is_exact(tmp_path):
    data = read_spharmpdm_coef(_COEF_PATH)
    out = tmp_path / "rt.coef"
    write_spharmpdm_coef(out, data)
    back = read_spharmpdm_coef(out)
    assert back.l_max == data.l_max
    assert _coeffs_equal(back, data)


def test_write_precision_6_reproduces_spharmpdm_layout(tmp_path):
    data = read_spharmpdm_coef(_COEF_PATH)
    out = tmp_path / "six.coef"
    write_spharmpdm_coef(out, data, precision=6)
    original = _COEF_PATH.read_text()
    written = out.read_text()
    # SPHARM-PDM prints negative zero for values that round to zero from
    # below. That sign does not survive the complex packing and carries no
    # information. Compare with it normalized.
    normalize = lambda t: t.replace("-0.000000", "0.000000")  # noqa: E731
    assert normalize(written) == normalize(original)
    assert written.startswith("{ 676,{")
    assert written.endswith("}}")
    assert written.count("\n") == original.count("\n")


def test_write_from_sha_coeffs_round_trip(tmp_path):
    data = read_spharmpdm_coef(_COEF_PATH)
    flat = spharmpdm_to_sha_coeffs(data)
    out = tmp_path / "sha.coef"
    write_spharmpdm_coef(out, sha_coeffs_to_spharmpdm(flat, ["andesred_07"])[0])
    back = spharmpdm_to_sha_coeffs(read_spharmpdm_coef(out))
    np.testing.assert_allclose(back, flat, atol=1e-14)


def test_write_rejects_non_spharmpdm_data(tmp_path):
    with pytest.raises(ValueError, match="SpharmPdmData"):
        write_spharmpdm_coef(tmp_path / "x.coef", np.zeros((1, 3)))
