# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Initialisation recording, saving and loading
"""


from pyearthtools.utils.initialisation.mixin import InitialisationRecordingMixin
from pyearthtools.utils.initialisation.load import load, save, update_contents
from pyearthtools.utils.initialisation.yaml import Loader, Dumper

from pyearthtools.utils.initialisation.imports import dynamic_import

OVERRIDE_KEY = "_pyearthtools_initialisation"

__all__ = ["InitialisationRecordingMixin", "save", "load", "update_contents", "dynamic_import", "Loader", "Dumper"]
