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
Bluelink ReANalysis
"""

from __future__ import annotations

import datetime
from glob import glob
from pathlib import Path
from typing import Any, Literal


import pyearthtools.data

from pyearthtools.data import Petdt, TimeResolution
from pyearthtools.data.exceptions import DataNotFoundError, InvalidIndexError
from pyearthtools.data.indexes import ArchiveIndex, decorators
from pyearthtools.data.transforms import Transform, TransformCollection
from pyearthtools.data.archive import register_archive

from site_archive_nci.utilities import check_project

BRAN_RESOLUTION = ["annual", "daily", "month", "static"]
BRAN_TYPES_RESOLUTION = [(365, "D"), (1, "D"), (31, "D"), None]
BRAN_REGEX = {
    "annual": "{ROOT_DIR}/annual/{variable}_ann_{year}.nc",
    "daily": "{ROOT_DIR}/daily/{variable}_{year}_{month}.nc",
    "month": "{ROOT_DIR}/month/{variable}_mth_{year}_{month}.nc",
    "static": "{ROOT_DIR}/static/{variable}.nc",
}


@register_archive("BRAN")
class BRAN(ArchiveIndex):
    """Index into Bluelink ReANalysis"""

    @property
    def _desc_(self):
        return {
            "singleline": "Bluelink ReANalysis",
            "range": "1993-current",
        }

    @decorators.alias_arguments(
        resolution=["time", "type", "datatype"], depth_value=["depth", "st_ocean"], variables=["variable"]
    )
    @decorators.variable_modifications(variable_keyword="variables")
    @decorators.check_arguments(
        resolution=BRAN_RESOLUTION,
        variables="site_archive_nci.variables.BRAN.{resolution}.valid",
    )
    def __init__(
        self,
        variables: list[str] | str,
        resolution: Literal[BRAN_RESOLUTION],
        *,
        depth_value: Any = None,
        transforms: Transform | TransformCollection | None = None,
    ):
        """
        Setup BRAN Indexer

        Args:
            variables (list[str] | str):
                Data variables to retrieve
            resolution (Literal[BRAN_TYPES]):
                Data resolution to retrieve
            depth_value (Any, optional):
                Depth value to select if data contains levels. Defaults to None.
            transforms (Transform | TransformCollection, optional):
                Base Transforms to apply. Defaults to TransformCollection().
        """
        check_project(project_code="gb6")

        variables = [variables] if isinstance(variables, str) else variables
        self.variables = variables

        self.resolution = resolution

        variables = [var.replace("ocean_", "") for var in variables]
        base_transform = pyearthtools.data.transforms.variables.Trim(variables)

        self.depth_value = depth_value
        if depth_value is not None:
            base_transform += pyearthtools.data.transforms.coordinates.Select(
                {coord: depth_value for coord in ["st_ocean"]}, ignore_missing=True
            )

        super().__init__(
            transforms=base_transform + (transforms or TransformCollection()),
            data_interval=BRAN_TYPES_RESOLUTION[BRAN_RESOLUTION.index(resolution)],
        )
        self.record_initialisation()

    def filesystem(
        self,
        basetime: str | datetime.datetime | Petdt,
    ) -> Path | dict[str, Path]:
        BRAN_HOME = self.ROOT_DIRECTORIES["BRAN"]

        paths = {}

        basetime = Petdt(str(basetime))

        for variable in self.variables:
            if self.resolution == "static":
                var_path = BRAN_REGEX[self.resolution].format(ROOT_DIR=BRAN_HOME, variable=variable)
            else:
                var_path = BRAN_REGEX[self.resolution].format(
                    ROOT_DIR=BRAN_HOME,
                    variable=variable,
                    year=basetime.year,
                    month="%02d" % basetime.month,
                )
            var_path = Path(var_path)

            for file in glob(str(var_path)):
                file = Path(file)
                if file.exists():
                    paths[variable] = file
                    break
            else:
                raise DataNotFoundError(
                    f"Unable to find data for: basetime: {basetime!r}, variables: {variable!r} at {var_path!r}"
                )

        return paths
