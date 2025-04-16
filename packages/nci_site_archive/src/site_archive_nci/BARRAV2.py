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

    DIR_STRUCTURE = "{nature}/{activity}/{domain}/{institution}/{driving_source}/{experiment}/{variant}/{source}/{version_realisation}/{frequency}/"
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
        activity: str | VARIABLE_DEFAULT = VariableDefault,
        domain: str | VARIABLE_DEFAULT = VariableDefault,
        institution: str | VARIABLE_DEFAULT = VariableDefault,
        driving_source: str | VARIABLE_DEFAULT = VariableDefault,
        experiment: str | VARIABLE_DEFAULT = VariableDefault,
        variant: str | VARIABLE_DEFAULT = VariableDefault,
        source: str | VARIABLE_DEFAULT = VariableDefault,
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
                Mostly based on https://docs.google.com/spreadsheets/d/1qUauozwXkq7r1g-L4ALMIkCNINIhhCPx/pyearthtools#gid=1672965248
            frequency (str):
                Temporal Frequency. '1hr' (1-hourly), '3hr', '6hr', 'day' (daily), 'mon' (monthly), 'fx'
            transforms (Transform | TransformCollection, optional):
                Transforms to apply to the data. Defaults to TransformCollection().

            nature (str | VARIABLE_DEFAULT, optional):
                'output'
            activity (str | VARIABLE_DEFAULT, optional):
                'reanalysis'
            domain (str | VARIABLE_DEFAULT, optional):
                Spatial domain and grid resolution of the data, namely 'AUS-11', 'AUS-22', 'AUS-04'.
            institution (str | VARIABLE_DEFAULT, optional):
                'BOM', RCM-institution
            driving_source (str| VARIABLE_DEFAULT, optional):
                'ERA5', global model that drives BARRA2 at the lateral boundary
            experiment (str | VARIABLE_DEFAULT, optional):
                'historical'
            variant (str | VARIABLE_DEFAULT, optional):
                labels the nature of ERA5 data used, either 'hres' or 'eda'
            source (str | VARIABLE_DEFAULT, optional):
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
                activity=activity,
                domain=domain,
                institution=institution,
                driving_source=driving_source,
                experiment=experiment,
                variant=variant,
                source=source,
                version_realisation=version_realisation,
                version=version,
            ),
        )
        self.record_initialisation()
