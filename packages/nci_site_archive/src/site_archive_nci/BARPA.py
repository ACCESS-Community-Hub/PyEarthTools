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
Bureau of Meteorology Atmospheric Regional Projections for Australia (BARPA)
"""

from __future__ import annotations
from pathlib import Path
from typing import Type

from pyearthtools.data import Petdt, DataNotFoundError
from pyearthtools.data.indexes import ArchiveIndex, decorators, VariableDefault
from pyearthtools.data.transforms import Transform, TransformCollection
from pyearthtools.data.archive import register_archive

from site_archive_nci.utilities import check_project

"""
Structure order

order:
  - activity
  - product
  - domain
  - institution
  - gcm_model_name
  - experiment_name
  - ensemble_member
  - rcm_model_name
  - version
  - frequency
  - variable

"""

BARPA_DIR_STRUCTURE = "{project}/{MIP}/{activity}/{domain}/{institution}/{driving_source}/{experiment}/{variant}/{source}/{version_realisation}/{frequency}/"

VARIABLE_DEFAULT = Type[VariableDefault]


@register_archive("BARPA", sample_kwargs={"variables": "CAPE", "driving_source": "ERA5", "frequency": "1hr"})
class BARPA(ArchiveIndex):
    """Index into Bureau of Meteorology Atmospheric Regional Projections for Australia"""

    @decorators.alias_arguments(variables=["variable"])
    @decorators.variable_modifications(variable_keyword="variables")
    @decorators.check_arguments(struc="site_archive_nci.structure.BARPA.struc")
    def __init__(
        self,
        variables: list[str] | str,
        driving_source: str,
        frequency: str,
        *,
        project: str | VARIABLE_DEFAULT = VariableDefault,
        MIP: str | VARIABLE_DEFAULT = VariableDefault,
        activity: str | VARIABLE_DEFAULT = VariableDefault,
        domain: str | VARIABLE_DEFAULT = VariableDefault,
        institution: str | VARIABLE_DEFAULT = VariableDefault,
        experiment: str | VARIABLE_DEFAULT = VariableDefault,
        variant: str | VARIABLE_DEFAULT = VariableDefault,
        source: str | VARIABLE_DEFAULT = VariableDefault,
        version_realisation: str | VARIABLE_DEFAULT = VariableDefault,
        version: str | VARIABLE_DEFAULT = "v20231001",  # VariableDefault,
        transforms: Transform | TransformCollection | None = None,
    ):
        """
        Bureau of Meteorology Atmospheric Regional Projections for Australia (BARPA)

        High resolution Climate simulation in the Australia Region.

        All arguments with `VariableDefault` as default might not have to be given,
        If based upon on the structure only one option is available, that will be picked.
        Otherwise an error will be raised.

        Args:
            variables (list[str] | str):
                Variables to retireve.
                Based upon https://docs.google.com/spreadsheets/d/1qUauozwXkq7r1g-L4ALMIkCNINIhhCPx/pyearthtools#gid=1672965248
            driving_source (str):
                Global Coupled Model. The models selected are:
                    ERA5, ACCESS-CM2, ACCESS-ESM1-5, NorESM2-MM, EC-Earth3, CESM2, CMCC-ESM2, MPI-ESM1-2-HR
                Must be only one.
            frequency (str):
                Temporal Frequency. 1hr (1-hourly), 3hr, 6hr, day (daily), mon (monthly), fx
            transforms (Transform | TransformCollection, optional):
                Transforms to apply to the data. Defaults to TransformCollection().

            project (str | VARIABLE_DEFAULT, optional):
                nature of data or project_id is output or CORDEX for data for CORDEX-CMIP6.
            MIP (str | VARIABLE_DEFAULT, optional):
                MIP-era is the cycle of CMIP defines experiment and data specifications. BARPS uses CMIP6.
            activity (str | VARIABLE_DEFAULT, optional):
                DD for dynamical downscaling.
            domain (str | VARIABLE_DEFAULT, optional):
                Spatial domain and grid resolution of the data, namely AUS-15, AUS-04.
            institution (str | VARIABLE_DEFAULT, optional):
                RCM-institution is BOM
            experiment (str | VARIABLE_DEFAULT, optional):
                Evaluation (for ERA5), historical or ssp126, ssp370 for CMIP6 experiments.
            variant (str | VARIABLE_DEFAULT, optional):
                Labels the ensemble member of the CMIP6 simulation that produced forcing data.
            source (str | VARIABLE_DEFAULT, optional):
                Either BARPA-R or BARPA-C.
            version_realisation (str | VARIABLE_DEFAULT, optional):
                Identifies the modelling version (TBC on identifying data version)
            version (str | VARIABLE_DEFAULT, optional):
                Denotes the date of data generation or date of data release
        """

        check_project(project_code="py18")

        variables = [variables] if isinstance(variables, str) else variables
        self.dir = Path(BARPA_DIR_STRUCTURE.format(**locals()))

        self.variables = variables
        self.version = str(version)

        super().__init__(transforms=(transforms or TransformCollection()))
        self.record_initialisation()

    def filesystem(
        self,
        querytime: str | Petdt,
    ) -> Path | dict[str, str]:
        BARPA_HOME = Path(self.ROOT_DIRECTORIES["BARPA"])

        discovered_paths = {}

        querytime_year = Petdt(querytime).at_resolution("year")

        for variable in self.variables:
            dir_path = BARPA_HOME / self.dir / variable / self.version
            paths = list(dir_path.glob(f"*{querytime_year.year}01-{querytime_year.year}12*.nc"))
            if len(paths) == 0:
                raise DataNotFoundError(f"Could not find data at {dir_path!r} at time {querytime!r}")
            discovered_paths[variable] = paths[0]
        return discovered_paths
