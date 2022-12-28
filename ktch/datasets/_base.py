"""Base IO code for small sample datasets"""

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

from importlib import resources

import pandas as pd
from sklearn.utils import Bunch
from sklearn.datasets._base import load_descr


def load_landmark_mosquito_wings(*, as_frame=True):
    """Load and return the mosquito wing landmark dataset
    (landmark-based morphometrics).

    ========================   ============
    Specimens                      127
    Landmarks per specimen          18
    Landmark dimensionality          2
    Features                      real
    ========================   ============

    Parameters
    ----------
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

    Returns
    ----------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        coords : {ndarray, dataframe} of shape (150, 4)
            The data matrix. If `as_frame=True`, `data` will be a pandas
            DataFrame.
        meta : {dataframe, list} of shape ()
        DESCR: str
            The full description of the dataset.
        filename: str
            The path to the location of the data.

    """
    data_module = "ktch.datasets.data"
    data_file_name = "data_landmark_mosquito_wings.csv"
    metadata_file_name = "meta_landmark_mosquito_wings.csv"
    descr_module = "ktch.datasets.descr"
    descr_file_name = "data_landmark_mosquito_wings.rst"

    coords = pd.read_csv(
        resources.open_text(data_module, data_file_name), index_col=[0, 1]
    )
    meta = pd.read_csv(
        resources.open_text(data_module, metadata_file_name), index_col=[0]
    )
    fdescr = load_descr(
        descr_module=descr_module,
        descr_file_name=descr_file_name,
    )

    if not as_frame:
        coords = coords.to_numpy()
        meta = meta.to_dict()

    return Bunch(
        coords=coords,
        meta=meta,
        DESCR=fdescr,
        filename=data_file_name,
    )


def load_outline_mosquito_wings(*, as_frame=True):
    """Load and return the mosquito wing outline dataset
    (outline-based morphometrics).

    ========================   ===========
    Specimens                      126
    Landmarks per specimen         100
    Landmark dimensionality          2
    Features                      real
    ========================   ===========

    Parameters
    ----------
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

    Returns
    ----------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        coords : {ndarray, dataframe} of shape (150, 4)
            The data matrix. If `as_frame=True`, `data` will be a pandas
            DataFrame.
        meta : {dataframe, list} of shape ()
        DESCR: str
            The full description of the dataset.
        filename: str
            The path to the location of the data.

    """
    data_module = "ktch.datasets.data"
    data_file_name = "data_outline_mosquito_wings.csv"
    metadata_file_name = "meta_outline_mosquito_wings.csv"
    descr_module = "ktch.datasets.descr"
    descr_file_name = "data_outline_mosquito_wings.rst"

    coords = pd.read_csv(
        resources.open_text(data_module, data_file_name), index_col=[0, 1]
    )
    meta = pd.read_csv(
        resources.open_text(data_module, metadata_file_name), index_col=[0]
    )
    fdescr = load_descr(
        descr_module=descr_module,
        descr_file_name=descr_file_name,
    )

    if not as_frame:
        coords = coords.to_numpy()
        meta = meta.to_dict()

    return Bunch(
        coords=coords,
        meta=meta,
        DESCR=fdescr,
        filename=data_file_name,
    )


def load_outline_bottles(*, as_frame=True):
    """Load and return the Momocs bottle outline dataset
    (outline-based morphometrics).

    =================   ==============
    Specimens                       40
    dimensionality                   4
    Features                      real
    =================   ==============

    Parameters
    ----------
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        coords : {list of ndarray, dataframe} of shape (40, n_coords, 2)
            The data matrix. If `as_frame=True`, `coords` will be a pandas
            DataFrame.
        meta : {dataframe, list} of shape ()
        DESCR: str
            The full description of the dataset.
        filename: str
            The path to the location of the data.

    """
    data_module = "ktch.datasets.data"
    data_file_name = "data_outline_bottles.csv"
    metadata_file_name = "meta_outline_bottles.csv"
    descr_module = "ktch.datasets.descr"
    descr_file_name = "data_outline_bottles.rst"

    coords = pd.read_csv(
        resources.open_text(data_module, data_file_name), index_col=[0, 1]
    )
    meta = pd.read_csv(
        resources.open_text(data_module, metadata_file_name), index_col=[0]
    )
    fdescr = load_descr(
        descr_module=descr_module,
        descr_file_name=descr_file_name,
    )

    if not as_frame:
        coords = coords.to_numpy()
        meta = meta.to_dict()

    return Bunch(
        coords=coords,
        meta=meta,
        DESCR=fdescr,
        filename=data_file_name,
    )


def load_coefficient_bottles(*, as_frame=True, norm=True):
    """Load and return the coefficients of  Momocs bottle outline datasets
    for testing the EFA

    Parameters
    ----------
    as_frame (bool, optional):
        If True, the data is a pandas DataFrame
        including columns with appropriate dtypes (numeric).
        The target is a pandas DataFrame or Series
        depending on the number of target columns.
    norm: bool, optional

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        coef : {list of ndarray, dataframe} of shape (40, 6, 4)
            The data matrix. If `as_frame=True`, `coords` will be a pandas
            DataFrame.
        meta : {dataframe, list} of shape ()
        DESCR: str
            The full description of the dataset.
        filename: str
            The path to the location of the data.
    """

    data_module = "ktch.datasets.data"
    if norm:
        data_file_name = "test_coef_nharm_6_norm_true_outline_bottles.csv"
    else:
        data_file_name = "test_coef_nharm_6_norm_false_outline_bottles.csv"
    metadata_file_name = "meta_outline_bottles.csv"
    descr_module = "ktch.datasets.descr"
    descr_file_name = "data_outline_bottles.rst"

    coef = pd.read_csv(
        resources.open_text(data_module, data_file_name), index_col=[0, 1]
    )
    meta = pd.read_csv(
        resources.open_text(data_module, metadata_file_name), index_col=[0]
    )
    fdescr = load_descr(
        descr_module=descr_module,
        descr_file_name=descr_file_name,
    )

    if not as_frame:
        coef = coef.to_numpy()
        meta = meta.to_dict()

    return Bunch(
        coef=coef,
        meta=meta,
        DESCR=fdescr,
        filename=data_file_name,
    )
