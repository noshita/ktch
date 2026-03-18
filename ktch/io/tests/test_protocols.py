"""Tests for MorphoData protocol and MorphoDataMixin."""

import numpy as np
import pytest

from ktch.io._chc import ChainCodeData
from ktch.io._nef import NefData
from ktch.io._protocols import MorphoData
from ktch.io._tps import TPSData


def _make_nef():
    coeffs = np.array([[1.0, 0.0, 0.0, 0.2], [0.01, 0.02, 0.03, 0.04]])
    return NefData(specimen_name="nef_specimen", coeffs=coeffs)


def _make_chc():
    return ChainCodeData(
        specimen_name="chc_specimen",
        x=0.0,
        y=0.0,
        area_per_pixel=1.0,
        chain_code=np.array([0, 0, 6, 6]),
    )


def _make_tps(with_curves=False):
    landmarks = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    curves = None
    if with_curves:
        curves = [np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])]
    return TPSData(specimen_name="tps_specimen", landmarks=landmarks, curves=curves)


#
# Protocol satisfaction
#


class TestProtocolSatisfaction:
    def test_nefdata_satisfies_morphodata(self):
        assert isinstance(_make_nef(), MorphoData)

    def test_chaincode_satisfies_morphodata(self):
        assert isinstance(_make_chc(), MorphoData)

    def test_tpsdata_satisfies_morphodata(self):
        assert isinstance(_make_tps(), MorphoData)


#
#  __array__ protocol
#


class TestArrayProtocol:
    def test_nefdata_asarray(self):
        nef = _make_nef()
        arr = np.asarray(nef)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 4)
        np.testing.assert_array_equal(arr, nef.coeffs)

    def test_nefdata_asarray_dtype(self):
        nef = _make_nef()
        arr = np.asarray(nef, dtype=np.float32)
        assert arr.dtype == np.float32

    def test_chaincode_asarray(self):
        chc = _make_chc()
        arr = np.asarray(chc)
        assert isinstance(arr, np.ndarray)
        assert arr.shape[1] == 2

    def test_tpsdata_asarray_no_curves(self):
        tps = _make_tps(with_curves=False)
        arr = np.asarray(tps)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 2)

    def test_tpsdata_asarray_with_curves_returns_landmarks(self):
        tps = _make_tps(with_curves=True)
        arr = np.asarray(tps)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 2)  # landmarks only, curves not included


#
# __repr__
#


class TestRepr:
    def test_nefdata_repr(self):
        nef = _make_nef()
        r = repr(nef)
        assert "NefData" in r
        assert "nef_specimen" in r
        assert "n_harmonics=2" in r

    def test_chaincode_repr(self):
        chc = _make_chc()
        r = repr(chc)
        assert "ChainCodeData" in r
        assert "chc_specimen" in r
        assert "n_points=4" in r

    def test_tpsdata_repr(self):
        tps = _make_tps()
        r = repr(tps)
        assert "TPSData" in r
        assert "tps_specimen" in r
        assert "n_landmarks=3" in r
        assert "n_dim=2" in r


#
# specimen_name property
#


class TestSpecimenName:
    def test_nefdata_specimen_name(self):
        nef = _make_nef()
        assert nef.specimen_name == "nef_specimen"

    def test_chaincode_specimen_name(self):
        chc = _make_chc()
        assert chc.specimen_name == "chc_specimen"

    def test_tpsdata_specimen_name(self):
        tps = _make_tps()
        assert tps.specimen_name == "tps_specimen"

    def test_tpsdata_idx_deprecated(self):
        tps = _make_tps()
        with pytest.warns(DeprecationWarning, match="specimen_name"):
            name = tps.idx
        assert name == "tps_specimen"


#
# Deprecated sample_name property
#


class TestDeprecatedSampleName:
    def test_nefdata_sample_name_deprecated(self):
        nef = _make_nef()
        with pytest.warns(DeprecationWarning, match="specimen_name"):
            name = nef.sample_name
        assert name == "nef_specimen"

    def test_chaincode_sample_name_deprecated(self):
        chc = _make_chc()
        with pytest.warns(DeprecationWarning, match="specimen_name"):
            name = chc.sample_name
        assert name == "chc_specimen"
