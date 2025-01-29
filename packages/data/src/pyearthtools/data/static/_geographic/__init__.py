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
Geographic Data Retrieval

Allow retrieval and download of known Geographic Datasets, which are specified in config files.

Will attempt to automatically load into geopandas if installed, or simply return.
"""

from pathlib import Path

DOWNLOAD_DATA: bool = True
DATA_BASEDIRECTORY: Path = Path(__file__).parent.resolve().absolute()

from pyearthtools.data.static._geographic.retrieval import get  # noqa: E402 F401
from pyearthtools.data.static._geographic import retrieval
