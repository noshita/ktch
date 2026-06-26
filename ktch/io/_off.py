"""Private OFF (Object File Format) parser."""

# Copyright 2026 Koji Noshita
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

from __future__ import annotations

from os import PathLike

import numpy as np


def _next_significant_line(f) -> str | None:
    """Return the next non-blank, non-comment line, or ``None`` at EOF.

    OFF files may carry ``#`` comment lines and blank lines between the
    header and the counts; skip them so parsing does not depend on layout.
    """
    for line in f:
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            return stripped
    return None


def _read_off(filepath: str | PathLike) -> tuple[np.ndarray, np.ndarray]:
    """Read an ASCII OFF file with triangular faces.

    Parameters
    ----------
    filepath : str or path-like
        Path to an ``.off`` file.

    Returns
    -------
    vertices : ndarray of shape (n_vertices, 3), dtype float64
        Vertex coordinates.
    faces : ndarray of shape (n_faces, 3), dtype int64
        Triangle vertex indices.

    Raises
    ------
    ValueError
        If the file header is not ``OFF`` or the face data contains
        non-triangular faces.
    """
    with open(filepath, encoding="utf-8") as f:
        header = _next_significant_line(f)
        if header != "OFF":
            raise ValueError(
                f"Expected 'OFF' header, got {header!r}. "
                "COFF and NOFF variants are not supported."
            )

        count_line = _next_significant_line(f)
        if count_line is None:
            raise ValueError("Missing vertex/face counts in OFF file.")
        try:
            n_vertices, n_faces = (int(tok) for tok in count_line.split()[:2])
        except ValueError:
            raise ValueError(f"Invalid OFF count line: {count_line!r}")

        if n_vertices <= 0:
            raise ValueError(f"OFF file declares no vertices (got {n_vertices}).")
        if n_faces <= 0:
            raise ValueError(
                f"OFF file declares no faces (got {n_faces}); "
                "a triangular surface mesh is required."
            )

        vertices = np.atleast_2d(np.loadtxt(f, max_rows=n_vertices, dtype=np.float64))
        raw_faces = np.atleast_2d(np.loadtxt(f, max_rows=n_faces, dtype=np.int64))

    # First column is the vertex count per face; must be 3 (triangles).
    if raw_faces.ndim != 2 or raw_faces.shape[1] != 4:
        raise ValueError(
            "Only triangular faces are supported "
            f"(expected 4 columns: count + 3 vertex indices, got shape "
            f"{raw_faces.shape})."
        )

    if not np.all(raw_faces[:, 0] == 3):
        raise ValueError(
            "Non-triangular faces detected. Only triangular meshes are supported."
        )

    faces = raw_faces[:, 1:]

    if vertices.shape != (n_vertices, 3):
        raise ValueError(
            f"Expected {n_vertices} vertices with 3 coordinates, "
            f"got shape {vertices.shape}."
        )
    if faces.shape != (n_faces, 3):
        raise ValueError(
            f"Expected {n_faces} triangular faces, got shape {faces.shape}."
        )

    return vertices, faces
