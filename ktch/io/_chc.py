"""Chain code file I/O functions."""

# Copyright 2025 Koji Noshita
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

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_DIRECTIONS = np.array(
    [
        [1, 0],  # 0: right
        [1, -1],  # 1: up-right
        [0, -1],  # 2: up
        [-1, -1],  # 3: up-left
        [-1, 0],  # 4: left
        [-1, 1],  # 5: down-left
        [0, 1],  # 6: down
        [1, 1],  # 7: down-right
    ]
)


def _chain_code_area(chain_code):
    """Compute enclosed area in pixels from a chain code.

    Coordinates represent pixel centers. The polygon area from
    the Shoelace formula is converted to pixel count using Pick's theorem:
    ``pixel_count = polygon_area + boundary_points / 2 + 1``.

    Each chain code step (orthogonal or diagonal) has gcd(|dx|, |dy|) = 1,
    so there are no intermediate lattice points on edges, and
    ``boundary_points = len(chain_code)``.

    Parameters
    ----------
    chain_code : np.ndarray
        Chain code sequence with values from 0 to 7.

    Returns
    -------
    area : int
        Enclosed area in pixels.
    """
    steps = _DIRECTIONS[chain_code]
    coords = np.vstack([np.zeros((1, 2)), np.cumsum(steps, axis=0)])
    x, y = coords[:, 0], coords[:, 1]
    polygon_area = 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
    boundary_points = len(chain_code)
    return int(round(polygon_area + boundary_points / 2 + 1))


@dataclass
class ChainCodeData:
    """Chain code data class.

    Chain codes represent 2D contours using directional codes from 0 to 7::

        3 2 1
        4 * 0
        5 6 7

    Parameters
    ----------
    sample_name : str
        Sample name.
    x : float
        X coordinate.
    y : float
        Y coordinate.
    area_per_pixel : float
        Area (mm2) per pixel.
    chain_code : np.ndarray
        Chain code sequence with values from 0 to 7 representing directions.
    area_pixels : int or None, optional
        Area in pixels.
        If None, computed from the chain code using the Shoelace formula.
        If provided and differs from the computed value,
        a warning is issued and the computed value is used.
    """

    sample_name: str
    x: float
    y: float
    area_per_pixel: float
    chain_code: np.ndarray
    area_pixels: int | None = None

    def __post_init__(self):
        if not isinstance(self.chain_code, np.ndarray):
            self.chain_code = np.array(self.chain_code)

        self._validate_chain_code()

        computed_area = _chain_code_area(self.chain_code)
        if self.area_pixels is not None and self.area_pixels != computed_area:
            warnings.warn(
                f"area_pixels ({self.area_pixels}) differs from the value "
                f"computed from chain code ({computed_area}). "
                f"Using computed value.",
                stacklevel=2,
            )
        self.area_pixels = computed_area

    def _validate_chain_code(self):
        """Validate that chain code values are between 0 and 7."""
        if not np.all((self.chain_code >= 0) & (self.chain_code <= 7)):
            invalid_values = self.chain_code[
                (self.chain_code < 0) | (self.chain_code > 7)
            ]
            raise ValueError(
                f"Chain code contains invalid values: {invalid_values}. "
                f"Values must be between 0 and 7 (inclusive)."
            )

    def get_chain_code(self):
        """Get the raw chain code as a numpy array.

        Returns
        -------
        chain_code : np.ndarray
            Raw chain code values (0-7) representing directions.
        """
        return self.chain_code

    def to_numpy(self):
        """Convert chain code to 2D coordinates as a numpy array.

        The chain code is converted to a sequence of 2D coordinates,
        starting from ``(x, y)`` and applying the directional changes
        based on the chain code values. Displacements are scaled by
        ``sqrt(area_per_pixel)``; when ``area_per_pixel <= 0`` (e.g.
        no scale marker was set in SHAPE), pixel units are used
        (scale factor = 1).

        The returned coordinates are in image coordinates where X
        increases rightward and Y increases downward. The starting
        point ``(x, y)`` and displacements share the same coordinate
        system (pixels when ``area_per_pixel <= 0``, physical units
        otherwise).

        Chain codes represent 2D contours using directional codes from 0 to 7::

            3 2 1
            4 * 0
            5 6 7

        Returns
        -------
        coords : np.ndarray
            2D coordinates with shape (n, 2) where n is the number of points.
            The first column is the x-coordinate and the second column is the y-coordinate.
        """
        steps = _DIRECTIONS[self.chain_code]
        coords = np.vstack([np.zeros((1, 2)), np.cumsum(steps, axis=0)])

        if self.area_per_pixel > 0:
            scale_factor = np.sqrt(self.area_per_pixel)
            coords *= scale_factor

        coords[:, 0] += self.x
        coords[:, 1] += self.y

        return coords

    def to_dataframe(self):
        """Convert chain code to 2D coordinates as a pandas DataFrame.

        The chain code is converted to a sequence of 2D coordinates,
        starting from (0, 0) and applying the directional changes
        based on the chain code values. The coordinates are scaled
        using the area_per_pixel value.

        Chain codes represent 2D contours using directional codes from 0 to 7::

            3 2 1
            4 * 0
            5 6 7

        Returns
        -------
        df : pd.DataFrame
            DataFrame with x and y columns for the coordinates and chain_code
            column for the direction codes. The first point has chain_code=-1
            since it has no direction.
        """
        coords = self.to_numpy()

        chain_code_values = np.zeros(len(coords), dtype=int)
        chain_code_values[0] = -1  # First point has no direction
        chain_code_values[1:] = self.chain_code  # Remaining points have directions

        df = pd.DataFrame(
            {
                "x": coords[:, 0],
                "y": coords[:, 1],
                "chain_code": chain_code_values,
            },
            index=pd.MultiIndex.from_tuples(
                [[self.sample_name, i] for i in range(len(coords))],
                name=["specimen_id", "coord_id"],
            ),
        )

        return df


def read_chc(file_path, as_frame=False, as_coordinates=True):
    """Read chain code (.chc) file.

    Chain codes represent 2D contours using directional codes from 0 to 7::

        3 2 1
        4 * 0
        5 6 7

    The chain code file format is:
    [Sample name] [X] [Y] [Area (mm2) per pixel] [Area (pixels)] [Chain code] -1

    Parameters
    ----------
    file_path : str
        Path to the chain code file.
    as_frame : bool, default=False
        If True, return pandas.DataFrame. Otherwise, return numpy.ndarray.
    as_coordinates : bool, default=True
        If True, convert chain codes to 2D coordinates.
        If False, return the raw chain code values.

    Returns
    -------
    chain_codes : list of np.ndarray or pd.DataFrame
        Chain codes or coordinates.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")

    if path.suffix.lower() != ".chc":
        raise ValueError(f"{path} is not a chain code file.")

    chc_data_list = _read_chc(path)

    if len(chc_data_list) == 1:
        if as_frame:
            return chc_data_list[0].to_dataframe()
        else:
            if as_coordinates:
                return chc_data_list[0].to_numpy()
            else:
                return chc_data_list[0].get_chain_code()
    else:
        if as_frame:
            dfs = [chc_data.to_dataframe() for chc_data in chc_data_list]
            return pd.concat(dfs)
        else:
            if as_coordinates:
                return [chc_data.to_numpy() for chc_data in chc_data_list]
            else:
                return [chc_data.get_chain_code() for chc_data in chc_data_list]


def write_chc(
    file_path,
    chain_codes,
    sample_names=None,
    xs=None,
    ys=None,
    area_per_pixels=None,
    area_pixels=None,
):
    """Write chain code to .chc file.

    Chain codes represent 2D contours using directional codes from 0 to 7::

        3 2 1
        4 * 0
        5 6 7

    The chain code file format is:
    [Sample name] [X] [Y] [Area (mm2) per pixel] [Area (pixels)] [Chain code] -1

    Parameters
    ----------
    file_path : str
        Path to the chain code file.
    chain_codes : list of np.ndarray or np.ndarray
        Chain codes with values from 0 to 7 representing directions.
    sample_names : list of str or str, optional
        Sample names.
    xs : list of float or float, optional
        X coordinates.
    ys : list of float or float, optional
        Y coordinates.
    area_per_pixels : list of float or float, optional
        Area (mm2) per pixel.
    area_pixels : list of int or int, optional
        Area in pixels.
    """
    path = Path(file_path)

    if isinstance(chain_codes, np.ndarray):
        if chain_codes.ndim == 1:
            chain_codes = [chain_codes]
        elif chain_codes.ndim == 2:
            chain_codes = [chain_codes[i, :] for i in range(chain_codes.shape[0])]
        else:
            raise ValueError("chain_codes must be a 1D or 2D array.")
    elif not isinstance(chain_codes, list):
        raise ValueError("chain_codes must be a list of numpy arrays or a numpy array.")

    n_samples = len(chain_codes)

    if sample_names is None:
        sample_names = ["Sample"] * n_samples
    elif isinstance(sample_names, str):
        sample_names = [sample_names] * n_samples

    if xs is None:
        xs = [0] * n_samples
    elif isinstance(xs, (int, float)):
        xs = [xs] * n_samples

    if ys is None:
        ys = [0] * n_samples
    elif isinstance(ys, (int, float)):
        ys = [ys] * n_samples

    if area_per_pixels is None:
        area_per_pixels = [1.0] * n_samples
    elif isinstance(area_per_pixels, (int, float)):
        area_per_pixels = [area_per_pixels] * n_samples

    if area_pixels is None:
        area_pixels = [None] * n_samples
    elif isinstance(area_pixels, int):
        area_pixels = [area_pixels] * n_samples

    for name, lst in [
        ("sample_names", sample_names),
        ("xs", xs),
        ("ys", ys),
        ("area_per_pixels", area_per_pixels),
        ("area_pixels", area_pixels),
    ]:
        if len(lst) != n_samples:
            raise ValueError(
                f"Length of {name} ({len(lst)}) does not match "
                f"number of chain codes ({n_samples})."
            )

    chc_data_list = []
    for i in range(n_samples):
        chc_data_list.append(
            ChainCodeData(
                sample_name=sample_names[i],
                x=xs[i],
                y=ys[i],
                area_per_pixel=area_per_pixels[i],
                chain_code=chain_codes[i],
                area_pixels=area_pixels[i],
            )
        )

    _write_chc(path, chc_data_list)


def _read_chc(file_path):
    """Read chain code file.

    Record consists of a header followed by chain code values terminated by ``-1``.
    The header and chain code may appear on a single line
    or be split across multiple lines (as documented in the SHAPE manual).

    Single-line::

        [Sample name] [X] [Y] [Area per pixel] [Area (pixels)] [Chain code...] -1

    Multi-line::

        [Sample name] [X] [Y] [Area per pixel] [Area (pixels)]
        [Chain code...] -1

    Parameters
    ----------
    file_path : str or Path
        Path to the chain code file.

    Returns
    -------
    chc_data_list : list of ChainCodeData
        Chain code data.
    """
    chc_data_list = []

    with open(file_path, "r") as f:
        tokens = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens.extend(line.split())

    # Parse tokens: each record ends with the sentinel "-1"
    pos = 0
    while pos < len(tokens):
        # Need at least header (5 fields) + 1 chain code value + sentinel
        if pos + 6 >= len(tokens):
            remaining = tokens[pos:]
            if remaining:
                raise ValueError(
                    f"Incomplete record at end of {file_path}: {' '.join(remaining)!r}"
                )
            break

        sample_name = tokens[pos]
        x = float(tokens[pos + 1])
        y = float(tokens[pos + 2])
        area_per_pixel = float(tokens[pos + 3])
        area_pixels = int(tokens[pos + 4])
        pos += 5

        # Collect chain code values until sentinel "-1"
        cc_values = []
        found_sentinel = False
        while pos < len(tokens):
            if tokens[pos] == "-1":
                pos += 1
                found_sentinel = True
                break
            cc_values.append(int(tokens[pos]))
            pos += 1

        if not found_sentinel:
            # No sentinel: treat all remaining values as chain code
            pass

        if not cc_values:
            raise ValueError(
                f"Empty chain code for record {sample_name!r} in {file_path}"
            )

        chc_data = ChainCodeData(
            sample_name=sample_name,
            x=x,
            y=y,
            area_per_pixel=area_per_pixel,
            chain_code=np.array(cc_values, dtype=int),
            area_pixels=area_pixels,
        )
        chc_data_list.append(chc_data)

    return chc_data_list


def _write_chc(file_path, chc_data_list):
    """Write chain code data to a file.

    Parameters
    ----------
    file_path : str or Path
        Path to the chain code file.
    chc_data_list : list of ChainCodeData
        Chain code data.
    """
    with open(file_path, "w") as f:
        for chc_data in chc_data_list:
            f.write(
                f"{chc_data.sample_name} {chc_data.x} {chc_data.y} "
                f"{chc_data.area_per_pixel} {chc_data.area_pixels} "
            )

            f.write(" ".join(map(str, chc_data.chain_code.tolist())))

            f.write(" -1\n")
