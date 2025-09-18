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
Australian Gridded Climate Data (AGCD)
"""

from __future__ import annotations

import datetime
from pathlib import Path


import pyearthtools.data
from pyearthtools.data import Petdt
from pyearthtools.data.exceptions import DataNotFoundError
from pyearthtools.data.indexes import ArchiveIndex, decorators
from pyearthtools.data.indexes.utilities import spellcheck

from pyearthtools.data.transforms import Transform, TransformCollection
from pyearthtools.data.archive import register_archive

from site_archive_nci.utilities import check_project

AGCD_VARIABLES = ["tmax", "tmin", "precip", "vapourpres_h09"]  # vapourpres_h15

AGCD_var_path = "site_archive_nci.variables.AGCD.{variable}.valid"
AGCD_RENAME = {"vapourpres": "vapourpres_09"}


@register_archive("AGCD")
class AGCD(ArchiveIndex):
    """Index into Australian Gridded Climate Data (AGCD)"""

    @property
    def _desc_(self):
        return {
            "singleline": "Australian Gridded Climate Data (AGCD)",
        }

    @decorators.variable_modifications(variable_keyword="variables")
    @decorators.check_arguments(variables=AGCD_VARIABLES, resolution=["day", "month"])
    def __init__(
        self,
        variables: list[str] | str,
        resolution: str,
        *,
        sub_var: str | dict = {"precip": "total", "default": "mean"},
        transforms: Transform | TransformCollection | None = None,
    ):
        """Setup AGCD Indexer

        Args:
            variables (list[str] | str):
                Data variables to retrieve
            resolution (str):
                Temporal resolution of data. Must be in ['day', 'month']
            sub_var (str | dict, optional):
                Record of data for each variable, 'default' can be set. Defaults to {'precip': 'total', 'default': 'mean'}.
            transforms (Transform | TransformCollection, optional):
                Base Transforms to apply.
                Defaults to TransformCollection().

        Raises:
            KeyError:
                If `sub_var` does not contain a default and a given variable
        """

        check_project(project_code="zv2")

        variables = [variables] if isinstance(variables, str) else variables

        if not isinstance(sub_var, dict):
            sub_var = {var: sub_var for var in variables}

        for var in variables:
            if var not in sub_var and "default" not in sub_var:
                raise KeyError("If 'sub_var' is a dict, it most contain entries for all variables")

            valid_args = spellcheck.open_static(AGCD_var_path.format(variable=var))
            spellcheck.check_prompt(
                sub_var[var] if var in sub_var else sub_var["default"],
                valid_args,
                f"subvar for {var}",
            )

        self.sub_var = sub_var
        self.resolution = resolution

        self.variables = variables
        base_transform = TransformCollection()

        base_transform += pyearthtools.data.transforms.attributes.Rename(AGCD_RENAME)
        base_transform += pyearthtools.data.transforms.variables.Trim(variables)

        super().__init__(
            transforms=base_transform + (transforms or TransformCollection()),
            data_interval=(
                1,
                "month" if "month" in sub_var[list(sub_var.keys())[0]] else "day",
            ),
        )
        self.record_initialisation()

    def filesystem(
        self,
        basetime: str | datetime.datetime | Petdt,
    ) -> Path:
        AGCD_HOME = self.ROOT_DIRECTORIES["AGCD"]

        paths = {}

        basetime = Petdt(basetime)

        for variable in self.variables:
            sub_var = self.sub_var[variable]
            sub_var = "/".join(
                [
                    sub_var.split("/")[0],
                    "r005",
                    f"01{self.resolution}" * sub_var.split("/")[1:],
                ]
            )

            basepath = Path(AGCD_HOME) / variable / sub_var

            path = list(basepath.glob(f"agcd*{variable}*_{basetime.year}.nc"))

            if len(path) == 0:
                raise DataNotFoundError(
                    f"Could not find data at {basepath}/agcd*{variable}*_{basetime.year}.nc for {variable} at time {basetime}"
                )
            paths[variable] = path[0]

        return paths
