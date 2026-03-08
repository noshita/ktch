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
from collections.abc import Sequence
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
    ----------
    file_path : str
        Path to the TPS file.
    as_frame : bool, default=False
        If True, return pandas.DataFrame. Otherwise, return numpy.ndarray.

    Returns
    -------
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
    landmarks_list = _normalize_landmarks_input(landmarks)
    n_specimens = len(landmarks_list)

    image_path_ = _normalize_optional_values(
        image_path, n_specimens=n_specimens, arg_name="image_path", default=None
    )

    if idx is None:
        idx_ = [None] if n_specimens == 1 else [i for i in range(n_specimens)]
    else:
        idx_ = _normalize_optional_values(
            idx, n_specimens=n_specimens, arg_name="idx", default=None
        )

    scale_ = _normalize_optional_values(
        scale, n_specimens=n_specimens, arg_name="scale", default=None
    )

    semilandmarks_ = _normalize_semilandmarks_input(semilandmarks, n_specimens)

    comments_ = _normalize_optional_values(
        comments, n_specimens=n_specimens, arg_name="comments", default=None
    )

    tps_data = [
        TPSData(
            idx=idx_[i],
            landmarks=landmarks_list[i],
            image_path=image_path_[i],
            scale=scale_[i],
            curves=semilandmarks_[i],
            comments=comments_[i],
        )
        for i in range(n_specimens)
    ]

    if n_specimens == 1:
        tps_data = tps_data[0]

    _write_tps(file_path=file_path, tps_data=tps_data)


def _normalize_landmarks_input(landmarks) -> list[np.ndarray]:
    """Normalize landmarks input into a per-specimen list of arrays."""
    if isinstance(landmarks, np.ndarray):
        if landmarks.ndim == 2:
            return [landmarks]
        if landmarks.ndim == 3:
            return [landmarks[i] for i in range(len(landmarks))]
        raise ValueError("landmarks must be a 2D or 3D numpy array.")

    if isinstance(landmarks, list):
        if len(landmarks) == 0:
            raise ValueError("landmarks list cannot be empty.")
        return [np.asarray(lm) for lm in landmarks]

    if isinstance(landmarks, pd.DataFrame):
        raise NotImplementedError()

    raise ValueError(
        "landmarks must be a numpy array or a list of per-specimen arrays."
    )


def _normalize_optional_values(
    values,
    *,
    n_specimens: int,
    arg_name: str,
    default=None,
):
    """Normalize scalar or sequence metadata into n_specimens length list."""
    if values is None:
        return [default] * n_specimens

    if isinstance(values, np.ndarray):
        if values.ndim == 0:
            return [values.item()] * n_specimens
        values_list = values.tolist()
        if len(values_list) != n_specimens:
            raise ValueError(
                f"{arg_name} must have length {n_specimens}, got {len(values_list)}."
            )
        return values_list

    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        values_list = list(values)
        if len(values_list) != n_specimens:
            raise ValueError(
                f"{arg_name} must have length {n_specimens}, got {len(values_list)}."
            )
        return values_list

    return [values] * n_specimens


def _normalize_curve_block(curves):
    """Normalize one specimen's semilandmarks to a list of 2D arrays."""
    if curves is None:
        return None

    if isinstance(curves, np.ndarray):
        if curves.ndim != 2:
            raise ValueError(
                "Each curve must be a 2D array with shape (n_points, n_dim)."
            )
        return [curves]

    if isinstance(curves, Sequence) and not isinstance(curves, (str, bytes, bytearray)):
        curves_list = []
        for curve in curves:
            curve_arr = np.asarray(curve)
            if curve_arr.ndim != 2:
                raise ValueError(
                    "Each curve must be a 2D array with shape (n_points, n_dim)."
                )
            curves_list.append(curve_arr)
        return curves_list

    raise ValueError(
        "semilandmarks must be None, a 2D array, or a sequence of 2D arrays."
    )


def _normalize_semilandmarks_input(semilandmarks, n_specimens: int):
    """Normalize semilandmarks input into per-specimen curve lists."""
    if semilandmarks is None:
        return [None] * n_specimens

    if n_specimens == 1:
        if (
            isinstance(semilandmarks, Sequence)
            and not isinstance(semilandmarks, (str, bytes, bytearray, np.ndarray))
            and len(semilandmarks) == 1
            and (
                semilandmarks[0] is None
                or isinstance(semilandmarks[0], (Sequence, np.ndarray))
            )
        ):
            return [_normalize_curve_block(semilandmarks[0])]
        return [_normalize_curve_block(semilandmarks)]

    if not (
        isinstance(semilandmarks, Sequence)
        and not isinstance(semilandmarks, (str, bytes, bytearray, np.ndarray))
    ):
        raise ValueError(
            "For multiple specimens, semilandmarks must be a sequence with one "
            "entry per specimen."
        )

    semilandmarks_list = list(semilandmarks)
    if len(semilandmarks_list) != n_specimens:
        raise ValueError(
            "For multiple specimens, semilandmarks must have the same length as "
            f"landmarks ({n_specimens}), got {len(semilandmarks_list)}."
        )

    return [_normalize_curve_block(curves) for curves in semilandmarks_list]


######################################
# Regular Express Patterns           #
######################################

PTN_HEAD = re.compile(r"^LM3?\s*=", flags=re.MULTILINE)

PTN_LM = re.compile(
    r"^(?P<LM>LM3?\s*=\s*[0-9]+(?:\s+[-\w\.]+)*)$",
    flags=re.MULTILINE,
)
PTN_CURVES = re.compile(
    r"^(?P<CURVES>CURVES\s*=\s*[0-9]+\s+(?:POINTS\s*=\s*[0-9]+(?:\s+[-0-9\.]+)+\s*)+)(?=^[A-Z]|\Z)",
    flags=re.MULTILINE,
)
PTN_POINTS = re.compile(
    r"^(?P<POINTS>POINTS\s*=\s*[0-9]+(?:\s+[-0-9\.]+)*)$",
    flags=re.MULTILINE,
)

PTN_DICT = re.compile(r"^(?P<key>\w+)\s*=\s*(?P<value>[^\r\n]+)$")
PTN_COORD = re.compile(
    r"^(?P<x>[-0-9\.]+|nan)\s(?P<y>[-0-9\.]+|nan)\s*(?P<z>[-0-9\.]+|nan)?$",
    flags=re.IGNORECASE,
)

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
        raise ValueError("Failed to parse landmark (LM) section in TPS specimen")

    filtered_str = PTN_CURVES.sub("", PTN_LM.sub("", specimen_str))
    filtered_list = [row for row in filtered_str.splitlines() if len(row) > 0]

    meta_dict = {}
    for row in filtered_list:
        m_meta = PTN_DICT.search(row)
        if m_meta is None:
            raise ValueError(f"Invalid metadata line in TPS specimen: {row!r}")
        meta_dict[m_meta["key"]] = m_meta["value"]

    if "ID" in meta_dict.keys():
        idx = meta_dict["ID"]
    else:
        raise ValueError("Missing required 'ID' field in TPS specimen")

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
    if m is None:
        raise ValueError(f"Invalid coordinate header in TPS: {header!r}")

    key = m["key"]
    value_rows = []
    for row in body:
        m_coord = PTN_COORD.match(row)
        if m_coord is None:
            raise ValueError(f"Invalid coordinate row in TPS: {row!r}")
        value_rows.append(
            [float(x) for x in m_coord.groups() if x is not None and len(x) > 0]
        )
    value = np.array(value_rows)

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
            f.write("CURVES=" + str(len(tps_data.curves)) + "\n")
            for curve in tps_data.curves:
                f.write("POINTS=" + str(len(curve)) + "\n")
                curve_lines = [" ".join(map(str, row)) for row in curve.tolist()]
                f.write("\n".join(curve_lines))
                f.write("\n")

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
