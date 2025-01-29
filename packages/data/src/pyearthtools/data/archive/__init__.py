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
Provide [Index][pyearthtools.data.ArchiveIndex] for known and widely used archived data sources.

These [Indexes][pyearthtools.data.ArchiveIndex] allow a user to retrieve data with only a date after being initialised.

More archives can be added by wrapping a class with [register_archive][pyearthtools.data.archive.register_archive]

!!! Warning
    `pyearthtools.data` contains no archives itself, and require additional modules to define them.

    Currently the following exist,
    ```
     - NCI
     - UKMO
    ```

!!! Note
    If setup correctly, any registered archive will be automatically imported if detected to be on the appropriate system.
    So, there may be no need to explicity import it.

"""

from pyearthtools.data.archive.extensions import register_archive

from pyearthtools.data.archive.root import set_root, reset_root, config_root


ZARR_IMPORTED = True
try:
    from pyearthtools.data.archive.zarr import ZarrIndex, ZarrTimeIndex  # noqa: F401
except (ImportError, ModuleNotFoundError):
    ZARR_IMPORTED = False


__all__ = [
    "set_root",
    "reset_root",
    "config_root",
    "register_archive",
]

if ZARR_IMPORTED:
    __all__.extend(["ZarrIndex", "ZarrTimeIndex"])
