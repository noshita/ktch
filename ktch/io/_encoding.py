"""Text encoding helpers for reading morphometric files."""

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

from pathlib import Path


def detect_text_encoding(path: str | Path) -> str:
    """Return an encoding for reading a text-based file.

    Tries UTF-8 first and falls back to Latin-1, which maps every byte
    value 1:1 and so never raises. This keeps reading independent of the
    platform locale and still loads files saved by Windows tools (e.g.
    tpsDig) in a Latin-1/CP1252 encoding.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the file to inspect.

    Returns
    -------
    encoding : str
        ``"utf-8"`` if the file decodes as UTF-8, otherwise ``"latin-1"``.
    """
    try:
        with open(path, encoding="utf-8") as f:
            f.read()
    except UnicodeDecodeError:
        return "latin-1"
    return "utf-8"
