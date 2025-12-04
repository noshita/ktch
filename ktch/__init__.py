"""ktch: A Python package for model-based morphometrics"""
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

from importlib.metadata import version

from . import datasets, harmonic, io, landmark, plot

__version__ = version(__name__)

__all__ = [
    "landmark",
    "harmonic",
    "outline",
    "io",
    "datasets",
    "plot",
    "__version__",
]


# for deprecated modules
def __getattr__(name: str):
    if name == "outline":
        from . import outline

        return outline
    raise AttributeError(f"'ktch' has no attribute '{name}'")


def __dir__():
    return list(globals().keys()) + ["outline"]
