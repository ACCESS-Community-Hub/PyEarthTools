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
MODerate resolution Imaging Spectroradiometer
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Literal


import pyearthtools.data
from pyearthtools.data import Petdt
from pyearthtools.data.exceptions import DataNotFoundError
from pyearthtools.data.indexes import ArchiveIndex, decorators
from pyearthtools.data.transforms import Transform, TransformCollection
from pyearthtools.data.archive import register_archive

from site_archive_nci.utilities import check_project


MODIS_REGIONS = ["AU"]
MODIS_RENAME = {"Band1": "lai"}
MODIS_RESOLUTION = ["8-daily", "monthly"]

MODIS_TYPES_RESOLUTION = [(8, "D"), (1, "month")]
MODIS_REGEX = {
    "8-daily": "MOD15A2H.{year_string}_AU_AWRAgrd.nc",
    "monthly": "MOD15A2H.MONTHLY.nc",
}


@register_archive("MODIS")
class MODIS(ArchiveIndex):
    """MODerate resolution Imaging Spectroradiometer

    !!! Note:
        MODIS data exists every 8 days, if data is requested at an invalid day,
        all data will be returned
    """

    @property
    def _desc_(self):
        return {
            "singleline": "MODerate resolution Imaging Spectroradiometer",
            "Range": "2012-2022",
            "Resolution": "8 days",
        }

    @decorators.alias_arguments(resolution=["time", "type", "datatype"], variables="variable")
    @decorators.variable_modifications(variable_keyword="variables")
    @decorators.check_arguments(
        region=MODIS_REGIONS,
        resolution=MODIS_RESOLUTION,
        variables="site_archive_nci.variables.MODIS.surface.valid",
    )
    def __init__(
        self,
        variables: list[str] | str,
        region: Literal[MODIS_REGIONS],
        resolution: Literal[MODIS_RESOLUTION],
        *,
        transforms: Transform | TransformCollection | None = None,
    ):
        """
        Setup MODIS Indexer

        Args:
            variables (list[str] | str):
                Data variables to retrieve
            region (Literal[MODIS_REGIONS]):
                Which Model region subset, currently the only option is 'AU' but there could be more in the future
            resolution (Literal[MODIS_TYPES]):
                Data temporal resolution to retrieve
            transforms (Transform | TransformCollection, optional):
                Base Transforms to apply. Defaults to TransformCollection().
        """
        check_project(project_code="fj4")

        variables = [variables] if isinstance(variables, str) else variables

        self.region = region
        self.resolution = resolution

        self.variables = variables
        base_transform = TransformCollection()

        base_transform += pyearthtools.data.transforms.attributes.Rename(MODIS_RENAME)
        base_transform += pyearthtools.data.transforms.variables.Trim(variables)

        # 8 day timesteps... so strange
        super().__init__(
            transforms=base_transform + (transforms or TransformCollection()),
            data_interval=MODIS_TYPES_RESOLUTION[MODIS_RESOLUTION.index(resolution)],
        )
        self.record_initialisation()

    def filesystem(
        self,
        basetime: str | datetime.datetime | Petdt,
    ) -> Path:
        MODIS_HOME = self.ROOT_DIRECTORIES["MODIS"]

        paths = {}

        basetime = Petdt(basetime)
        basepath = Path(MODIS_HOME.format(region=self.region))

        for variable in self.variables:
            var_path = basepath

            files_in_dir = var_path.iterdir()
            year_string = str(basetime.year)

            relevant_path = None
            for filename in files_in_dir:
                if filename == var_path / MODIS_REGEX[self.resolution].format(year_string=year_string):
                    relevant_path = filename

            if relevant_path is not None:
                relevant_path = var_path / relevant_path

                if relevant_path.exists():
                    paths[variable] = relevant_path
                    continue

            raise DataNotFoundError(
                f"Unable to find data for: basetime: {basetime}, variables: {variable} at {var_path}"
            )
        return paths
