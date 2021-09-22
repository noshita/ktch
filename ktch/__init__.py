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

from ._template import TemplateEstimator
from ._template import TemplateClassifier
from ._template import TemplateTransformer

from ._version import __version__

__all__ = ['TemplateEstimator', 'TemplateClassifier', 'TemplateTransformer',
           '__version__']
