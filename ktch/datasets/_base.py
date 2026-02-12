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

import zipfile
from importlib import resources
from importlib.metadata import version as get_package_version
from pathlib import Path

import pandas as pd
from sklearn.datasets._base import load_descr
from sklearn.utils import Bunch

from ._registry import get_registry, get_url

try:
    import pooch
except ImportError:
    pooch = None

DATA_MODULE = "ktch.datasets.data"
DESCR_MODULE = "ktch.datasets.descr"


def _safe_extractall(zip_ref, target_dir):
    """Extract zip contents with path traversal validation.

    Parameters
    ----------
    zip_ref : zipfile.ZipFile
        An open ZipFile object to extract.
    target_dir : str or Path
        The directory to extract into.

    Raises
    ------
    ValueError
        If any member path attempts to escape the target directory.
    """
    target = Path(target_dir).resolve()
    for member in zip_ref.namelist():
        member_path = (target / member).resolve()
        if not member_path.is_relative_to(target):
            raise ValueError(
                f"Attempted path traversal in zip file: {member}"
            )
    zip_ref.extractall(target)


def _resolve_data_path(dataset_name, filename):
    """Resolve the path to a bundled data file.

    Parameters
    ----------
    dataset_name : str
        Subdirectory name under data/ (e.g., "landmark_mosquito_wings").
    filename : str
        Filename within the subdirectory.

    Returns
    -------
    Traversable
        Path-like object usable with pd.read_csv, open(), etc.
    """
    return resources.files(DATA_MODULE).joinpath(dataset_name, filename)


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
    dataset_name = "landmark_mosquito_wings"
    data_file_name = "data_landmark_mosquito_wings.csv"
    metadata_file_name = "meta_landmark_mosquito_wings.csv"
    descr_file_name = "data_landmark_mosquito_wings.rst"

    coords = pd.read_csv(
        _resolve_data_path(dataset_name, data_file_name), index_col=[0, 1]
    )
    meta = pd.read_csv(
        _resolve_data_path(dataset_name, metadata_file_name), index_col=[0]
    )
    fdescr = load_descr(
        descr_module=DESCR_MODULE,
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


def load_landmark_trilobite_cephala(*, as_frame=False):
    """Load and return the trilobite cephala landmark dataset with semilandmarks
    used in [Serra_2023]_.

    The original of this dataset (v1.2) is available at
    `Zenodo <https://zenodo.org/records/17308187>`_.

    This dataset contains 2D landmark and curve semilandmark data for
    300 trilobite cephala specimens, with 16 fixed landmarks and 4 curves.

    ============================   ============
    Specimens                            300
    Landmarks per specimen                16
    Curves per specimen                    4
    Curve points              12, 20, 20, 20
    Landmark dimensionality                2
    Features                            real
    ============================   ============

    Parameters
    ----------
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric).

    Returns
    ----------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        landmarks : {ndarray, dataframe} of shape (300, 16, 2)
            The landmark data. If ``as_frame=True``, returns a pandas
            DataFrame.
        curves : list of list of ndarray
            Curve semilandmark data. Each element is a list of 4 curves
            for one specimen.
        curve_landmarks : list of tuple of (int, int)
            The start and end fixed-landmark indices for each curve,
            used by :func:`~ktch.landmark.combine_landmarks_and_curves`.
        meta : {dict, dataframe}
            Metadata for each specimen.
        DESCR : str
            The full description of the dataset.
        filename : str
            The path to the location of the data.

    References
    --------------------
    .. [Serra_2023] Serra, F., Balseiro, D., Monnet, C., Randolfe, E., Bignon, A., Rustán, J.J., Bault, V., Muñoz, D.F., Vaccari, N.E., Martinetto, M., Crônier, C., Waisfeld, B.G., 2023. A dynamic and collaborative database for morphogeometric information of trilobites. Sci. Data 10, 841. https://doi.org/10.1038/s41597-023-02724-9
    """
    from ktch.io import read_tps

    dataset_name = "landmark_trilobite_cephala"
    data_file_name = "data_landmark_trilobite_cephala.tps"
    metadata_file_name = "meta_landmark_trilobite_cephala.csv"
    descr_file_name = "data_landmark_trilobite_cephala.rst"

    tps_path = str(_resolve_data_path(dataset_name, data_file_name))
    landmarks, curves = read_tps(tps_path, as_frame=as_frame)

    meta = pd.read_csv(
        _resolve_data_path(dataset_name, metadata_file_name), index_col=[0]
    )
    fdescr = load_descr(
        descr_module=DESCR_MODULE,
        descr_file_name=descr_file_name,
    )

    curve_landmarks = [(1, 6), (9, 11), (12, 14), (3, 14)]

    if as_frame:
        # read_tps returns a list of DataFrames when as_frame=True with curves;
        # concatenate into a single DataFrame for consistency.
        landmarks = pd.concat(landmarks)
    else:
        meta = meta.to_dict()

    return Bunch(
        landmarks=landmarks,
        curves=curves,
        curve_landmarks=curve_landmarks,
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
    dataset_name = "outline_mosquito_wings"
    data_file_name = "data_outline_mosquito_wings.csv"
    metadata_file_name = "meta_outline_mosquito_wings.csv"
    descr_file_name = "data_outline_mosquito_wings.rst"

    coords = pd.read_csv(
        _resolve_data_path(dataset_name, data_file_name), index_col=[0, 1]
    )
    meta = pd.read_csv(
        _resolve_data_path(dataset_name, metadata_file_name), index_col=[0]
    )
    fdescr = load_descr(
        descr_module=DESCR_MODULE,
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
    dataset_name = "outline_bottles"
    data_file_name = "data_outline_bottles.csv"
    metadata_file_name = "meta_outline_bottles.csv"
    descr_file_name = "data_outline_bottles.rst"

    coords = pd.read_csv(
        _resolve_data_path(dataset_name, data_file_name), index_col=[0, 1]
    )
    meta = pd.read_csv(
        _resolve_data_path(dataset_name, metadata_file_name), index_col=[0]
    )
    fdescr = load_descr(
        descr_module=DESCR_MODULE,
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

    dataset_name = "outline_bottles"
    if norm:
        data_file_name = "test_coef_nharm_6_norm_true_outline_bottles.csv"
    else:
        data_file_name = "test_coef_nharm_6_norm_false_outline_bottles.csv"
    metadata_file_name = "meta_outline_bottles.csv"
    descr_file_name = "data_outline_bottles.rst"

    coef = pd.read_csv(
        _resolve_data_path(dataset_name, data_file_name), index_col=[0, 1]
    )
    meta = pd.read_csv(
        _resolve_data_path(dataset_name, metadata_file_name), index_col=[0]
    )
    fdescr = load_descr(
        descr_module=DESCR_MODULE,
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


def load_outline_leaf_bending(*, as_frame=False):
    """Load and return the synthetic 3D leaf bending outline dataset.

    This dataset contains 3D outlines of synthetic wheat-like leaves
    with bending deformation.

    ========================   ===========
    Specimens                       60
    Points per specimen            200
    Coordinate dimensionality        3
    Features                      real
    ========================   ===========

    Parameters
    ----------
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `as_frame` is False, `coords` will be a numpy array of shape
        (n_specimens, n_points, 3).

    Returns
    ----------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        coords : {ndarray of shape (60, 200, 3), dataframe}
            The coordinate data. If `as_frame=True`, `coords` will be a pandas
            DataFrame with MultiIndex (specimen_id, coord_id).
        meta : {dict, dataframe}
            Metadata containing bending parameters for each specimen.
        DESCR: str
            The full description of the dataset.
        filename: str
            The path to the location of the data.

    """
    dataset_name = "outline_leaf_bending"
    data_file_name = "data_outline_leaf_bending.csv"
    metadata_file_name = "meta_outline_leaf_bending.csv"
    descr_file_name = "data_outline_leaf_bending.rst"

    coords = pd.read_csv(
        _resolve_data_path(dataset_name, data_file_name), index_col=[0, 1]
    )
    meta = pd.read_csv(
        _resolve_data_path(dataset_name, metadata_file_name), index_col=[0]
    )
    fdescr = load_descr(
        descr_module=DESCR_MODULE,
        descr_file_name=descr_file_name,
    )

    if not as_frame:
        n_specimens = len(meta)
        coords = coords.to_numpy().reshape(n_specimens, -1, 3)
        meta = meta.to_dict()

    return Bunch(
        coords=coords,
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


###########################################################
#
#   Remote dataset functions (using pooch)
#
###########################################################


def _get_default_version():
    """Get the default dataset version based on the package version.

    Returns
    -------
    str
        The major.minor.patch version string (e.g., "0.7.0").
    """
    pkg_version = get_package_version("ktch")
    # Extract major.minor.patch (e.g., "0.7.0" from "0.7.0.dev1")
    parts = pkg_version.split(".")
    if len(parts) >= 3:
        # Handle dev/rc versions: "0.7.0.dev1" -> "0.7.0"
        patch = parts[2].split("dev")[0].split("rc")[0].split("a")[0].split("b")[0]
        return f"{parts[0]}.{parts[1]}.{patch if patch else '0'}"
    return pkg_version


def _fetch_remote_data(dataset_name, version):
    """Fetch remote dataset file using pooch.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset file to fetch.
    version : str
        The dataset version to fetch (e.g., "0.7.0").

    Returns
    -------
    str
        Path to the downloaded file.

    Raises
    ------
    ImportError
        If pooch is not installed.
    ValueError
        If the version is not found in the registry.
    """
    if pooch is None:
        raise ImportError(
            "Missing optional dependency 'pooch' for downloading remote datasets."
            "Please install it using: pip install ktch[data]"
        )

    registry = get_registry(version)
    url = get_url(dataset_name, version)
    known_hash = registry.get(dataset_name)

    if known_hash is None:
        raise ValueError(f"Dataset '{dataset_name}' not found in version '{version}'")

    return pooch.retrieve(
        url=url,
        known_hash=f"sha256:{known_hash}",
        path=pooch.os_cache("ktch-data") / version,
        fname=dataset_name,
    )


def load_image_passiflora_leaves(*, return_paths=False, as_frame=False, version=None):
    """Load and return the Passiflora leaf image dataset.

    This dataset contains leaf images of Passiflora species from GigaDB
    [Chitwood_et_al_2014]_.

    The data is downloaded from a remote server on first use and cached locally.

    ========================   ============
    Specimens                      TBD
    Image format                   PNG
    ========================   ============

    Parameters
    ----------
    return_paths : bool, default=False
        If True, return file paths to the images instead of loading them
        as numpy arrays. This is useful for large datasets or when using
        image loading libraries like PIL or OpenCV.
    as_frame : bool, default=False
        If True, the metadata is returned as a pandas DataFrame.
        Otherwise, it is returned as a dict.
    version : str, optional
        The dataset version to load (e.g., "0.7.0"). If None, uses the
        version corresponding to the installed ktch package.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        images : list of ndarray or list of str
            If `return_paths=False`, list of image arrays with shape
            (height, width, channels). If `return_paths=True`, list of
            file paths to the images.
        meta : dict or DataFrame
            Metadata containing image_id (index), genus, and species for each image.
            If `as_frame=True`, returns a pandas DataFrame.
        DESCR : str
            The full description of the dataset.
        data_dir : str
            Path to the directory containing the extracted data.
        version : str
            The version of the dataset that was loaded.

    Notes
    -----
    This function requires the optional dependency `pooch` for downloading
    the dataset. Install it using: ``pip install ktch[data]``

    When `return_paths=False`, numpy is used to load the images. For more
    control over image loading, use `return_paths=True` and load images
    with your preferred library.

    References
    ----------
    .. [Chitwood_et_al_2016] Chitwood, D.H., Otoni, W.C., 2016. Supporting data
       for "Morphometric analysis of Passiflora leaves: the relationship
       between landmarks of the vasculature and elliptical Fourier
       descriptors of the blade". GigaScience Database.
       https://doi.org/10.5524/100251

    Examples
    --------
    >>> from ktch.datasets import load_image_passiflora_leaves  # doctest: +SKIP
    >>> data = load_image_passiflora_leaves()  # doctest: +SKIP
    >>> len(data.images)  # doctest: +SKIP
    ...
    >>> data.meta['species']  # doctest: +SKIP
    ...
    >>> # Load a specific version
    >>> data = load_image_passiflora_leaves(version="0.7.0")  # doctest: +SKIP
    """
    if version is None:
        version = _get_default_version()

    descr_file_name = "data_image_passiflora_leaves.rst"

    # Fetch and extract the dataset
    archive_path = _fetch_remote_data("image_passiflora_leaves.zip", version)
    data_dir = Path(archive_path).parent / "image_passiflora_leaves"

    if not data_dir.exists():
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            _safe_extractall(zip_ref, data_dir.parent)

    # Load metadata and description
    meta_path = data_dir / "metadata.csv"
    meta = pd.read_csv(meta_path, index_col=0, skipinitialspace=True)

    fdescr = load_descr(
        descr_module=DESCR_MODULE,
        descr_file_name=descr_file_name,
    )

    # Image paths/numpy arrays
    image_paths = [data_dir / "images" / f"{image_id}.png" for image_id in meta.index]

    if return_paths:
        images = [str(p) for p in image_paths]
    else:
        import numpy as np
        from PIL import Image

        images = [np.array(Image.open(p)) for p in image_paths]

    if not as_frame:
        meta = meta.to_dict()

    return Bunch(
        images=images,
        meta=meta,
        DESCR=fdescr,
        data_dir=str(data_dir),
        version=version,
    )
