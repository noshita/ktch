"""TPS file I/O functions."""

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

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

####################################
# TPS dataclass
####################################


@dataclass
class TPSData:
    idx: str
    landmarks: np.ndarray
    image_path: str = None
    scale: float = None
    curves: list[np.ndarray] = None
    comments: str = None

    @property
    def landmarks(self) -> np.ndarray:
        return self._landmarks

    @landmarks.setter
    def landmarks(self, value: npt.ArrayLike) -> None:
        self._landmarks = np.array(value)

    def to_numpy(self):
        if self.curves is None:
            return self.landmarks
        else:
            return self.landmarks, self.curves

    def to_dataframe(self):
        if self.landmarks.shape[1] == 2:
            columns = ["x", "y"]
        elif self.landmarks.shape[1] == 3:
            columns = ["x", "y", "z"]
        else:
            raise ValueError("n_dim must be 2 or 3.")

        df_landmarks = pd.DataFrame(
            self.landmarks,
            columns=columns,
            index=pd.MultiIndex.from_tuples(
                [[self.idx, i] for i in range(len(self.landmarks))],
                name=["specimen_id", "coord_id"],
            ),
        )
        if self.curves is None:
            return df_landmarks
        else:
            return df_landmarks, [
                pd.DataFrame(
                    curve,
                    columns=columns,
                    index=pd.MultiIndex.from_tuples(
                        [[self.idx, i] for i in range(len(curve))],
                        name=["specimen_id", "coord_id"],
                    ),
                )
                for curve in self.curves
            ]


####################################
# TPS I/O functions
####################################


def read_tps(file_path, as_frame=False):
    """Read TPS file.

    Parameters
    ==========
    file_path : str
        Path to the TPS file.
    as_frame : bool, default=False
        If True, return pandas.DataFrame. Otherwise, return numpy.ndarray.

    Returns
    =======
    landmarks : ndarray
        Landmarks.
    semilandmarks: list[ndarray] or ndarray, optional
        Semilandmarks.
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")

    if not path.suffix == ".tps":
        raise ValueError(f"{path} is not a TPS file.")

    tps_data = _read_tps(path)

    if isinstance(tps_data, TPSData):
        if as_frame:
            return tps_data.to_dataframe()
        else:
            return tps_data.to_numpy()
    elif isinstance(tps_data, list):
        semilandmark_flag = sum(
            [False if tps_datum.curves is None else True for tps_datum in tps_data]
        )
        if (semilandmark_flag == 0) or (semilandmark_flag == len(tps_data)):
            pass
        else:
            raise ValueError("Some specimens have semilandmarks and others do not.")
        if as_frame:
            if semilandmark_flag == 0:
                landmarks = [tps_datum.to_dataframe() for tps_datum in tps_data]
                landmarks = pd.concat(landmarks)
                return landmarks
            else:
                landmarks = [tps_datum.to_dataframe()[0] for tps_datum in tps_data]

                semilandmarks = [tps_datum.to_dataframe()[1] for tps_datum in tps_data]
                return landmarks, semilandmarks
        else:
            if semilandmark_flag == 0:
                landmarks = np.array([tps_datum.to_numpy() for tps_datum in tps_data])
                return landmarks
            else:
                landmarks = np.array(
                    [tps_datum.to_numpy()[0] for tps_datum in tps_data]
                )
                semilandmarks = [tps_datum.to_numpy()[1] for tps_datum in tps_data]
                return landmarks, semilandmarks


def write_tps(
    file_path,
    landmarks,
    image_path=None,
    idx=None,
    scale=None,
    semilandmarks=None,
    comments=None,
) -> None:
    """Write TPS file."""

    if isinstance(landmarks, list):
        n_specimens = len(landmarks)
    elif isinstance(landmarks, np.ndarray):
        if landmarks.ndim == 3:
            n_specimens = len(landmarks)
        elif landmarks.ndim == 2:
            n_specimens = 1
        else:
            raise ValueError("")
    elif isinstance(landmarks, pd.DataFrame):
        raise NotImplementedError()
    else:
        raise ValueError("")

    if n_specimens == 1:
        tps_data = TPSData(
            idx=idx,
            landmarks=landmarks,
            image_path=image_path,
            scale=scale,
            comments=comments,
        )
    else:
        if image_path is None:
            image_path_ = [None] * n_specimens

        if idx is None:
            idx_ = [i for i in range(n_specimens)]

        if scale is None:
            scale_ = [None] * n_specimens

        if semilandmarks is None:
            semilandmarks_ = [None] * n_specimens

        if comments is None:
            comments_ = [None] * n_specimens

        tps_data = [
            TPSData(
                idx=idx_[i],
                landmarks=landmarks[i],
                image_path=image_path_[i],
                scale=scale_[i],
                curves=semilandmarks_[i],
                comments=comments_[i],
            )
            for i in range(n_specimens)
        ]

    _write_tps(file_path=file_path, tps_data=tps_data)


######################################
# Regular Express Patterns           #
######################################

PTN_HEAD = re.compile(r"^LM3*\s*=", flags=re.MULTILINE)

PTN_LM = re.compile(
    r"^(?P<LM>LM3*\s*=\s*[0-9]+\s*([\w\.-]+\s+[\w\.-]+\s*[\w\.-]*\s*)+)$",
    flags=re.MULTILINE,
)
PTN_CURVES = re.compile(
    r"^(?P<CURVES>CURVES\s*=\s*[0-9]+\s+(?P<POINTS>(POINTS\s*=\s*[0-9]+\s+[0-9\s\.-]+)+))$",
    flags=re.MULTILINE,
)
PTN_POINTS = re.compile(
    r"^(?P<POINTS>POINTS\s*=\s*[0-9]+\s+([\w\.-]+\s+[\w\.-]+\s*[\w\.-]*\s*)+)",
    flags=re.MULTILINE,
)

PTN_DICT = re.compile(r"^(?P<key>\w+)\s*=\s*(?P<value>.+)$")
PTN_COORD = re.compile(r"^(?P<x>[0-9-\.]*)\s(?P<y>[0-9-\.]*)\s*(?P<z>[0-9-\.]*)?$")

######################################
# Helper functions                   #
######################################


def _read_tps(file_path):
    with open(file_path, "r") as f:
        read_data = f.read()
        specimens = PTN_HEAD.split(read_data)
        specimens = ["LM=" + specimen for specimen in specimens if len(specimen) > 0]

    tps_data = [_read_tps_single(specimen) for specimen in specimens]
    if len(tps_data) == 1:
        return tps_data[0]

    return tps_data


def _read_tps_single(specimen_str: str) -> TPSData:
    m = PTN_LM.search(specimen_str)
    if m is not None:
        key, landmarks = _read_coordinate_values(
            [row for row in m["LM"].splitlines() if len(row) > 0]
        )
    else:
        raise ValueError("")

    filtered_str = PTN_CURVES.sub("", PTN_LM.sub("", specimen_str))
    filtered_list = [row for row in filtered_str.splitlines() if len(row) > 0]

    meta_dict = {
        PTN_DICT.search(row)["key"]: PTN_DICT.search(row)["value"]
        for row in filtered_list
    }

    if "ID" in meta_dict.keys():
        idx = meta_dict["ID"]
    else:
        raise ValueError("")

    if "IMAGE" in meta_dict.keys():
        image_path = meta_dict["IMAGE"]
    else:
        image_path = None

    if "SCALE" in meta_dict.keys():
        scale = meta_dict["SCALE"]
    else:
        scale = None

    if "COMMENTS" in meta_dict.keys():
        comments = meta_dict["COMMENTS"]
    else:
        comments = None

    m = PTN_CURVES.search(specimen_str)
    if m is not None:
        curves = []
        for points in PTN_POINTS.finditer(m["CURVES"]):
            key, val = _read_coordinate_values(
                [row for row in points["POINTS"].splitlines() if len(row) > 0]
            )
            curves.append(val)
    else:
        curves = None

    tps_data = TPSData(
        idx=idx,
        landmarks=landmarks,
        image_path=image_path,
        scale=scale,
        curves=curves,
        comments=comments,
    )

    return tps_data


def _read_coordinate_values(coordinate_list):
    header = coordinate_list[0]
    body = coordinate_list[1:]

    m = PTN_DICT.match(header)

    key = m["key"]
    value = np.array(
        [
            [float(x) for x in PTN_COORD.match(row).groups() if len(x) > 0]
            for row in body
        ]
    )

    return key, value


def _write_tps_single(file_path, tps_data, write_mode="w"):
    with open(file_path, write_mode) as f:
        f.write("LM=" + str(len(tps_data.landmarks)) + "\n")
        f.write(
            "\n".join([" ".join(map(str, row)) for row in tps_data.landmarks.tolist()])
        )
        f.write("\n")
        if tps_data.image_path is not None:
            f.write("IMAGE=" + tps_data.image_path + "\n")

        f.write("ID=" + str(tps_data.idx) + "\n")

        if tps_data.scale is not None:
            f.write("SCALE=" + str(tps_data.scale) + "\n")

        if tps_data.curves is not None:
            f.write("CURVES=" + str(len(tps_data.curves)))
            for curve in tps_data.curves:
                f.write("POINTS=" + str(len(curve)) + "\n")
                f.writelines([" ".join(map(str, row)) for row in curve.tolist()])

        if tps_data.comments is not None:
            f.write("COMMENTS=" + tps_data.comments + "\n")

        f.write("\n")


def _write_tps(file_path, tps_data):
    if isinstance(tps_data, TPSData):
        _write_tps_single(file_path, tps_data)
    elif isinstance(tps_data, list):
        for i, tps_datum in enumerate(tps_data):
            if i == 0:
                mode = "w"
            else:
                mode = "a"
            _write_tps_single(file_path, tps_datum, write_mode=mode)
