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


def load_landmark_mosquito_wings(*, as_frame=False):
    """Load and return the mosquito wing landmark dataset used in
    [Rohlf_and_Slice_1990]_.

    The original of this dataset is available at
    `SB Morphometrics <https://www.sbmorphometrics.org/data/RohlfSlice1990Mosq.nts>`_.

    ========================   ============
    Specimens                      127
    Landmarks per specimen          18
    Landmark dimensionality          2
    Features                      real
    ========================   ============

    Parameters
    --------------------
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

    Returns
    --------------------
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

    References
    --------------------
    .. [Rohlf_and_Slice_1990] Rohlf, F.J., Slice, D., 1990. Extensions of the Procrustes Method for the Optimal Superimposition of Landmarks. Systematic Zoology 39, 40. https://doi.org/10.2307/2992207

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


def load_outline_mosquito_wings(*, as_frame=False):
    """Load and return the mosquito wing outline dataset used in
    [Rohlf_and_Archie_1984]_, however includes only 126 of 127 specimens
    because of missing in the original NTS file.

    The original NTS file of this data is available at
    `SB Morphometrics <https://www.sbmorphometrics.org/data/RohlfArchieWingOutlines.nts>`_.

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

    References
    --------------------
    .. [Rohlf_and_Archie_1984] Rohlf, F.J., Archie, J.W., 1984. A Comparison of Fourier Methods for the Description of Wing Shape in Mosquitoes (Diptera: Culicidae). Syst Zool 33, 302. https://doi.org/10.2307/2413076

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


def load_outline_bottles(*, as_frame=False):
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
        coords = [
            coords.loc[specimen_id].to_numpy().reshape(-1, 2)
            for specimen_id in coords.index.get_level_values(0).unique()
        ]
        meta = meta.to_dict()

    return Bunch(
        coords=coords,
        meta=meta,
        DESCR=fdescr,
        filename=data_file_name,
    )


def load_coefficient_bottles(*, as_frame=False, norm=True):
    """Load and return the coefficients of  Momocs bottle outline datasets
    for testing the EFA.

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
        coef = [
            coef.loc[i]
            .to_numpy()
            .T.reshape(
                -1,
            )
            for i in coef.index.get_level_values(0).unique()
        ]
        meta = meta.to_dict()

    return Bunch(
        coef=coef,
        meta=meta,
        DESCR=fdescr,
        filename=data_file_name,
    )


###########################################################
#
#   utility functions
#
###########################################################


def convert_coords_df_to_list(df_coords):
    """Convert a dataframe of coordinates to a list of numpy arrays.

    Parameters
    ----------
    df_coords: pandas.DataFrame of index (specimen_id, coord_id), columns (axis (x, y (, z)))
        The dataframe of coordinates.

    Returns
    -------
    list_coords: list
        The list of numpy arrays.
    """
    dim = df_coords.shape[1]
    coords_list = [
        df_coords.loc[specimen_id].to_numpy().reshape(-1, dim)
        for specimen_id in df_coords.index.get_level_values(0).unique()
    ]
    return coords_list


def convert_coords_df_to_df_sklearn_transform(df_coords):
    """Convert a dataframe of coordinates to a dataframe of coordinates
    for sklearn transformers.

    Parameters
    ----------
    df_coords: pandas.DataFrame of index (specimen_id, coord_id), columns (axis (x, y (, z)))
        The dataframe of coordinates.

    Returns
    -------
    df_coords_new: pandas.DataFrame of index (specimen_id), columns (coord_id, axis)
        The dataframe of coordinates compatible with input of scikit-learn transformers.
    """

    df_coords_new = df_coords.unstack().swaplevel(axis=1).sort_index(axis=1)

    return df_coords_new
