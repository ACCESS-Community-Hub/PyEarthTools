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
Himawari 8/9 satellite data
"""

from __future__ import annotations

import datetime
from glob import glob
from pathlib import Path

import pyearthtools.data
from pyearthtools.data import Petdt, TimeDelta
from pyearthtools.data.exceptions import DataNotFoundError
from pyearthtools.data.indexes import ArchiveIndex, decorators, StaticDataIndex, Index
from pyearthtools.data.transforms import Transform, TransformCollection
from pyearthtools.data.archive import register_archive

from site_archive_nci.utilities import check_project

import xarray as xr

@register_archive("AusTopo")
class AusTopo(Index):
    """
    Australian Region Topography 0.01 degree

    https://dx.doi.org/10.25914/60a10aa56dd1b

    Data is covered by Attribution-ShareAlike 4.0 International
    
    """

    @property
    def _desc_(self):
        return {
            "singleline": "Aus",
            "Range": "static",
            "Resolution": "10 minutes",
        }

    def __init__(
        self,
        *,
        transforms: Transform | TransformCollection | None = None,
    ):
        """
        Args:
            transforms (Transform | TransformCollection, optional):
                Base Transforms to apply. Defaults to TransformCollection().
        """
        check_project(project_code="gh70")

        base_transform = (transforms or TransformCollection())
        super().__init__(transforms=base_transform)
        self.record_initialisation()

    def get(self, date_of_interest, **kwargs):
        '''
        Load the topography, but replace the date on the file with the
        date that is being requested, so that even though the underlying data
        is static, it can be requests against any date/time. 
        '''

        file = self.load_file()
        doi = Petdt(date_of_interest)
        xr_time = doi.datetime64()
        file['time'] = [xr_time] 
        file = file.rename({'lat': 'latitude', 'lon': 'longitude'})

        return file

    def load_file(self):
        file = xr.open_dataset('/g/data/gh70/ANUClimate/v2-0/topogrid/dem01/ANUClimate_v2-0_dem01.nc')
        return file

    