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
Structure order is taken from the CORDEX-CMIP6 archiving specs:
https://zenodo.org/records/15047096

order:
  - project_id
  - activity_id
  - domain_id
  - institution_id
  - driving_source_id
  - driving_experiment_id
  - driving_variant_label
  - source_id
  - version_realisation
  - frequency
  - variables
  - version
"""

BARPA_DIR_STRUCTURE = "{nature}/{project_id}/{activity_id}/{domain_id}/{institution_id}/{driving_source_id}/{driving_experiment_id}/{driving_variant_label}/{source_id}/{version_realisation}/{frequency}/"

VARIABLE_DEFAULT = Type[VariableDefault]


@register_archive("BARPA", sample_kwargs={"variables": "CAPE", "driving_source_id": "ERA5", "frequency": "1hr"})
class BARPA(ArchiveIndex):
    """Index into Bureau of Meteorology Atmospheric Regional Projections for Australia"""

    @decorators.alias_arguments(variables=["variable"])
    @decorators.variable_modifications(variable_keyword="variables")
    @decorators.check_arguments(struc="site_archive_nci.structure.BARPA.struc")
    def __init__(
        self,
        variables: list[str] | str,
        driving_source_id: str,
        frequency: str,
        driving_experiment_id: str,
        *,
        nature: str | VARIABLE_DEFAULT = VariableDefault,
        project_id: str | VARIABLE_DEFAULT = VariableDefault,
        activity_id: str | VARIABLE_DEFAULT = VariableDefault,
        domain_id: str | VARIABLE_DEFAULT = VariableDefault,
        institution_id: str | VARIABLE_DEFAULT = VariableDefault,
        driving_variant_label: str | VARIABLE_DEFAULT = VariableDefault,
        source_id: str | VARIABLE_DEFAULT = VariableDefault,
        version_realisation: str | VARIABLE_DEFAULT = VariableDefault,
        version: str | VARIABLE_DEFAULT = "latest",  # VariableDefault,
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
                Based upon https://opus.nci.org.au/spaces/NDP/pages/338002650/BARPA+Parameter+Descriptions
            driving_source_id (str):
                Global Coupled Model. The models selected are:
                    ERA5, ACCESS-CM2, ACCESS-ESM1-5, NorESM2-MM, EC-Earth3, CESM2, CMCC-ESM2, MPI-ESM1-2-HR
                Must be only one.
            frequency (str):
                Temporal Frequency. 1hr (1-hourly), 3hr, 6hr, day (daily), mon (monthly), fx
            transforms (Transform | TransformCollection, optional):
                Transforms to apply to the data. Defaults to TransformCollection().
            project_id (str | VARIABLE_DEFAULT, optional):
                nature of data or project_id is output or CORDEX-CMIP6.
            activity_id (str | VARIABLE_DEFAULT, optional):
                DD for dynamical downscaling.
            domain_id (str | VARIABLE_DEFAULT, optional):
                Spatial domain and grid resolution of the data, namely AUS-15, AUST-15, AUST-04, AUS-20i.
            institution_id (str | VARIABLE_DEFAULT, optional):
                RCM-institution is BOM
            driving_experiment_id (str | VARIABLE_DEFAULT, optional):
                Evaluation (for ERA5), historical or ssp126, ssp370, ssp585 (only ACCESS-ESM-1-5, EC-Earth3) for CMIP6 experiments.
            driving_variant_label (str | VARIABLE_DEFAULT, optional):
                Labels the ensemble member of the CMIP6 simulation that produced forcing data.
            source_id (str | VARIABLE_DEFAULT, optional):
                Either BARPA-R or BARPA-C.
            version_realisation (str | VARIABLE_DEFAULT, optional):
                Identifies the modelling version (v1-r1)
            version (str | VARIABLE_DEFAULT, optional):
                Denotes the date of data generation or date of data release
        """

        check_project(project_code="py18")

        variables = [variables] if isinstance(variables, str) else variables
        self.dir = Path(BARPA_DIR_STRUCTURE.format(**locals()))

        self.variables = variables
        self.version = str(version)
        self.source_id = source_id

        super().__init__(transforms=(transforms or TransformCollection()))
        self.record_initialisation()

    def filesystem(
        self,
        querytime: str | Petdt,
    ) -> Path | dict[str, str]:
        BARPA_HOME = Path(self.ROOT_DIRECTORIES["BARPA"])

        discovered_paths = {}

        for variable in self.variables:
            dir_path = BARPA_HOME / self.dir / variable / self.version
            if self.source_id == "BARPA-R":
                querytime_year = Petdt(querytime).at_resolution("year")
                filetmpl = f"*{querytime_year.year}01-{querytime_year.year}12*.nc"
            elif self.source_id == "BARPA-C":
                querytime_year = Petdt(querytime).at_resolution("month")
                filetmpl = (
                    f"*{querytime_year.year}{querytime_year.month:02}-{querytime_year.year}{querytime_year.month:02}.nc"
                )
            else:
                raise DataNotFoundError(f"Could not find source_id of {self.source_id}")

            paths = list(dir_path.glob(filetmpl))
            if len(paths) == 0:
                raise DataNotFoundError(f"Could not find data at {dir_path!r} at time {querytime!r}")
            discovered_paths[variable] = paths[0]
        return discovered_paths
