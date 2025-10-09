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
Bureau of Meteorology Atmospheric Regional Projections for Australia (BARRA_V2)
"""

from __future__ import annotations


import pyearthtools.data
from pyearthtools.data.indexes import Structured, VARIABLE_DEFAULT, VariableDefault, decorators
from pyearthtools.data.transforms import Transform, TransformCollection
from pyearthtools.data.archive import register_archive

from site_archive_nci.utilities import check_project

from site_archive_nci.ancilliary.BARRA_V2 import variable_rename, coarse_variables

temporal_resolution = {
    "fx": None,
    "mon": (1, "month"),
    "3hr": (3, "hour"),
    "1hr": (1, "hour"),
    "day": (1, "day"),
    "20min": (20, "minute"),
}


@register_archive("BARRA_V2", sample_kwargs={"variables": "CAPE", "frequency": "1hr"})
class BARRA_V2(Structured):
    """Bureau of Meteorology Atmospheric high-resolution Regional Reanalysis for Australia, BARRA Version 2"""

    @property
    def _desc_(self):
        return {
            "singleline": "Bureau of Meteorology Atmospheric high-resolution Regional Reanalysis for Australia, BARRA Version 2",
            "Documentation": "https://dx.doi.org/10.25914/1x6g-2v48",
        }

    DIR_STRUCTURE = "{nature}/{activity_id}/{domain_id}/{institution_id}/{driving_source_id}/{driving_experiment_id}/{driving_variant_label}/{source_id}/{version_realisation}/{frequency}/"
    GLOB_TEMPLATE = "{variable}/{version}/{variable}_*%Y%m-%Y%m.nc"

    @decorators.alias_arguments(variables=["variable"])
    @decorators.variable_modifications(variable_keyword="variables")
    @decorators.check_arguments(struc="site_archive_nci.structure.BARRA_V2.struc")
    def __init__(
        self,
        variables: list[str] | str,
        frequency: str,
        *,
        nature: str | VARIABLE_DEFAULT = VariableDefault,
        activity_id: str | VARIABLE_DEFAULT = VariableDefault,
        domain_id: str | VARIABLE_DEFAULT = VariableDefault,
        institution_id: str | VARIABLE_DEFAULT = VariableDefault,
        driving_source_id: str | VARIABLE_DEFAULT = VariableDefault,
        driving_experiment_id: str | VARIABLE_DEFAULT = VariableDefault,
        driving_variant_label: str | VARIABLE_DEFAULT = VariableDefault,
        source_id: str | VARIABLE_DEFAULT = VariableDefault,
        version_realisation: str | VARIABLE_DEFAULT = VariableDefault,
        version: str | VARIABLE_DEFAULT = "latest",
        transforms: Transform | TransformCollection | None = None,
    ):
        """
        Bureau of Meteorology Atmospheric high-resolution Regional Reanalysis for Australia (BARRA_V2)

        BARRA2 provides the Bureau's higher resolution regional atmospheric reanalysis
        over Australia and surrounding regions, spanning 1979-present day time period.
        When completed, it replaces the first version of BARRA (Su et al.,
        doi: 10.5194/gmd-14-4357-2021; 10.5194/gmd-12-2049-2019).

        All arguments with `VariableDefault` as default might not have to be given,
        If based upon on the structure only one option is available, that will be picked.
        Otherwise an error will be raised.

        Args:
            variables (list[str] | str):
                Variables to retrieve.
                Mostly based on https://opus.nci.org.au/spaces/NDP/pages/338002591/BARRA2+Parameter+Descriptions and
                structure order is taken from the CORDEX-CMIP6 archiving specs: https://zenodo.org/records/15047096
            frequency (str):
                Temporal Frequency. '1hr' (1-hourly), '3hr', '6hr', 'day' (daily), 'mon' (monthly), 'fx'
            transforms (Transform | TransformCollection, optional):
                Transforms to apply to the data. Defaults to TransformCollection().

            nature (str | VARIABLE_DEFAULT, optional):
                'output'
            activity_id (str | VARIABLE_DEFAULT, optional):
                'reanalysis'
            domain_id (str | VARIABLE_DEFAULT, optional):
                Spatial domain and grid resolution of the data, namely 'AUS-11', AUST-11, 'AUS-22', AUST-22, 'AUST-04'.
            institution_id (str | VARIABLE_DEFAULT, optional):
                'BOM', RCM-institution
            driving_source_id (str| VARIABLE_DEFAULT, optional):
                'ERA5', global model that drives BARRA2 at the lateral boundary
            driving_experiment_id (str | VARIABLE_DEFAULT, optional):
                'historical'
            driving_variant_label (str | VARIABLE_DEFAULT, optional):
                labels the nature of ERA5 data used, either 'hres' or 'eda'
            source_id (str | VARIABLE_DEFAULT, optional):
                BARRA-R2, BARRA-RE2, or BARRA-C2
            version_realisation (str | VARIABLE_DEFAULT, optional):
                identifies the modelling version of BARRA2 (TBC on identifying data version)
            version (str | VARIABLE_DEFAULT, optional):
                Denotes the date of data generation or date of data release.
                Defaults to 'latest'
        """

        check_project(project_code="ob53")

        if frequency == "fx":
            self.GLOB_TEMPLATE = "{variable}/{version}/{variable}_*.nc"

        transforms = transforms or TransformCollection()
        transforms += pyearthtools.data.transforms.variables.Drop("time_bnds")

        variables = [variables] if isinstance(variables, str) else variables
        new_vars = []

        for var in variables:
            if var in coarse_variables[frequency]:
                new_vars.extend(coarse_variables[frequency][var])
            else:
                new_vars.append(var)
        variables = new_vars

        preprocess = pyearthtools.data.transforms.dimensions.Expand(["pressure", "depth", "height"], missing="skip")
        preprocess += pyearthtools.data.transforms.attributes.Rename(
            {var: variable_rename[var] for var in variables if var in variable_rename}
        )

        super().__init__(
            variables=variables,
            data_interval=temporal_resolution[frequency],
            transforms=transforms,
            preprocess_transforms=preprocess,
            round=frequency == "mon",
            config_vars=dict(
                frequency=frequency,
                nature=nature,
                activity_id=activity_id,
                domain_id=domain_id,
                institution_id=institution_id,
                driving_source_id=driving_source_id,
                driving_experiment_id=driving_experiment_id,
                driving_variant_label=driving_variant_label,
                source_id=source_id,
                version_realisation=version_realisation,
                version=version,
            ),
        )
        self.record_initialisation()
