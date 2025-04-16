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
Ocean Modelling and Analysis Prediction System
"""

from __future__ import annotations

import datetime
from glob import glob
from pathlib import Path
from typing import Any, Literal


import pyearthtools.data
from pyearthtools.data import Petdt

from pyearthtools.data.exceptions import DataNotFoundError
from pyearthtools.data.indexes import ArchiveIndex, decorators
from pyearthtools.data.transforms import Transform, TransformCollection
from pyearthtools.data.archive import register_archive

from site_archive_nci.utilities import check_project


OceanMaps_TYPES = ["analysis", "forecast"]
OceanMaps_SUBVAR = ["ocean_an00", "ocean_an01", "ocean_an02", "ocean_an_ensemble"]


OceanMaps_REGEX = "{ROOT_DIR}/version_{version}/{datatype}/{variable}/"
OceanMaps_SUBVAR_FORMAT = "{sub_var}_{date}12_{variable}*"


@register_archive("OceanMaps")
class OceanMaps(ArchiveIndex):
    """Index into Ocean Modelling and Analysis Prediction System"""

    @property
    def _desc_(self):
        return {
            "singleline": "Ocean Modelling and Analysis Prediction System",
            "Resolution": "1 day",
        }

    @decorators.alias_arguments(variables=["variable"])
    @decorators.variable_modifications(variable_keyword="variables")
    @decorators.check_arguments(
        datatype=OceanMaps_TYPES,
        sub_var=OceanMaps_SUBVAR,
        variables="site_archive_nci.variables.OceanMaps.{datatype}.valid",
    )
    def __init__(
        self,
        variables: list[str] | str,
        datatype: Literal[OceanMaps_TYPES],
        sub_var: Literal[OceanMaps_SUBVAR],
        *,
        depth_value: Any = None,
        version: str = "3.4",
        transforms: Transform | TransformCollection | None = None,
    ):
        """
        Setup BRAN Indexer

        Args:
            variables (list[str] | str):
                Data variables to retrieve
            datatype (Literal[OceanMaps_TYPES]):
                Data type to retrieve, must be ["analysis", "forecast"]
            sub_var (Literal[OceanMaps_SUBVAR]):
                OceanMaps identifier, must be "ocean_an00", "ocean_an01", "ocean_an02", "ocean_an_ensemble"]
            depth_value (Any, optional):
                Depth value to select if data contains levels. Defaults to None.
            version (str, optional):
                OceanMaps Version. Defaults to "3.4".
            transforms (Transform | TransformCollection, optional):
                Base Transforms to apply. Defaults to TransformCollection().
        """
        check_project(project_code="rr6")

        variables = [variables] if isinstance(variables, str) else variables
        self.variables = variables

        self.datatype = datatype
        self.sub_var = sub_var

        self.version = version

        base_transform = pyearthtools.data.transforms.variables.Trim(variables)

        self.depth_value = depth_value
        if depth_value is not None:
            base_transform += pyearthtools.data.transforms.coordinates.Select(
                {coord: depth_value for coord in ["st_ocean"]}, ignore_missing=True
            )
        super().__init__(transforms=base_transform + (transforms or TransformCollection()), data_interval=(1, "D"))
        self.record_initialisation()

    def filesystem(
        self,
        basetime: str | datetime.datetime | Petdt,
    ) -> Path | dict[str, Path]:
        OceanMaps_HOME = self.ROOT_DIRECTORIES["OceanMaps"]

        paths = {}

        basetime = Petdt(str(basetime))
        basetime -= datetime.timedelta(days=1)

        for variable in self.variables:
            basepath = OceanMaps_REGEX.format(
                ROOT_DIR=OceanMaps_HOME,
                version=self.version,
                datatype=self.datatype,
                variable=variable,
            )
            var_path = Path(basepath) / OceanMaps_SUBVAR_FORMAT.format(
                sub_var=self.sub_var,
                date=str(basetime).replace("-", ""),
                variable=variable,
            )

            for file in glob(str(var_path)):
                file = Path(file)
                if file.exists():
                    paths[variable] = file
                    break
            else:
                raise DataNotFoundError(
                    f"Unable to find data for: basetime: {basetime}, variables: {variable} at {var_path}"
                )

        return paths
