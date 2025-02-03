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


import os

from pyearthtools.data.archive import zarr

from pyearthtools.data import patterns
from pyearthtools.data.patterns.default import PatternIndex


class ZarrIndex(zarr.ZarrIndex, PatternIndex):
    """
    Zarr archive for use as a pattern for `CachingIndex`.

    If filling in a template archive, ensure `template` = True.

    This will cause any cache checks of existence to return False, and thus generate the data.

    For actual usage, `template` = False.
    """

    def __init__(self, root_dir: os.PathLike, **kwargs):
        root_dir, temp_dir = patterns.utils.parse_root_dir(str(root_dir))
        super().__init__(root_dir, **kwargs, root_dir=root_dir)
        self.temp_dir = temp_dir

    def search(self, *_):
        # Prevents args being passed to underlying search
        return super().search()

    def save(self, data, *_, **kwargs):
        # Prevents args being passed to underlying save
        return super().save(data, **kwargs)


class ZarrTimeIndex(zarr.ZarrTimeIndex, ZarrIndex):
    """Time aware Zarr Pattern archive"""
