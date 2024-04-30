"""
Australian Gridded Climate Data (AGCD)
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Literal


from edit.data import EDITDatetime, transform
from edit.data.exceptions import DataNotFoundError
from edit.data.indexes import ArchiveIndex, decorators
from edit.data.indexes.utilities import spellcheck

from edit.data.transform import Transform, TransformCollection
from edit.data.archive import register_archive

from edit_archive_NCI.utilities import check_project

AGCD_VARIABLES = ["tmax", "tmin", "precip", "vapourpres_h09"]  # vapourpres_h15

AGCD_var_path = "edit_archive_NCI.variables.AGCD.{variable}.valid"
AGCD_RENAME = {"vapourpres": "vapourpres_09"}


@register_archive('AGCD')
class AGCD(ArchiveIndex):
    """Index into Australian Gridded Climate Data (AGCD)"""

    @property
    def _desc_(self):
        return {
            "singleline": "Australian Gridded Climate Data (AGCD)",
        }

    @decorators.check_arguments(variables=AGCD_VARIABLES, resolution=["day", "month"])
    def __init__(
        self,
        variables: list[str] | str,
        resolution: str,
        *,
        sub_var: str | dict = {"precip": "total", "default": "mean"},
        transforms: Transform | TransformCollection = TransformCollection(),
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

        self.make_catalog()
        check_project(project_code='zv2')

        variables = [variables] if isinstance(variables, str) else variables

        if not isinstance(sub_var, dict):
            sub_var = {var: sub_var for var in variables}

        for var in variables:
            if var not in sub_var and "default" not in sub_var:
                raise KeyError(f"If 'sub_var' is a dict, it most contain entries for all variables")

            valid_args = spellcheck.open_file(AGCD_var_path.format(variable=var))
            spellcheck.check_prompt(
                sub_var[var] if var in sub_var else sub_var["default"],
                valid_args,
                f"subvar for {var}",
            )

        self.sub_var = sub_var
        self.resolution = resolution

        self.variables = variables
        base_transform = TransformCollection()

        base_transform += transform.variables.rename_variables(AGCD_RENAME)
        base_transform += transform.variables.variable_trim(variables)

        super().__init__(
            transforms=base_transform + transforms,
            data_interval=(
                1,
                "month" if "month" in sub_var[list(sub_var.keys())[0]] else "day",
            ),
        )

    def filesystem(
        self,
        basetime: str | datetime.datetime | EDITDatetime,
    ) -> Path:
        AGCD_HOME = self.ROOT_DIRECTORIES["AGCD"]

        paths = {}

        basetime = EDITDatetime(basetime)

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
