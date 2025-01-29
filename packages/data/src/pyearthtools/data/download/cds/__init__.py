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
# Copernicus Data Store Downloaders

## Available

| Name | Description |
| ---- | ----------- |
| `root_cds` | Base class for Copernius access. `_get_from_cds` must be implemented. |
| `cds` | General Copernicus downloader, uses init args to define query. |
| `ERA5` | ERA5 specific downloader. |

"""

from pyearthtools.data.download.cds.cds import cds, root_cds
from pyearthtools.data.download.cds.ERA5 import ERA5
