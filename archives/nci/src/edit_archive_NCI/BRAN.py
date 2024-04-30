"""
Bluelink ReANalysis
"""

from __future__ import annotations

import datetime
from glob import glob
from pathlib import Path
from typing import Any, Literal


from edit.data import EDITDatetime, transform
from edit.data.exceptions import DataNotFoundError, InvalidIndexError
from edit.data.indexes import ArchiveIndex, decorators
from edit.data.transform import Transform, TransformCollection
from edit.data.archive import register_archive

from edit_archive_NCI.utilities import check_project

BRAN_RESOLUTION = ["annual", "daily", "month", "static"]
BRAN_TYPES_RESOLUTION = [(365, "D"), (1, "D"), (31, "D"), None]
BRAN_REGEX = {
    "annual": "{ROOT_DIR}/annual/{variable}_ann_{year}.nc",
    "daily": "{ROOT_DIR}/daily/{variable}_{year}_{month}.nc",
    "month": "{ROOT_DIR}/month/{variable}_mth_{year}_{month}.nc",
    "static": "{ROOT_DIR}/static/{variable}.nc",
}

@register_archive('BRAN')
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
    @decorators.check_arguments(
        resolution=BRAN_RESOLUTION,
        variables="edit_archive_NCI.variables.BRAN.{resolution}.valid",
    )
    def __init__(
        self,
        variables: list[str] | str,
        resolution: Literal[BRAN_RESOLUTION],
        *,
        depth_value: Any = None,
        transforms: Transform | TransformCollection = TransformCollection(),
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
        self.make_catalog()
        check_project(project_code='gb6')

        variables = [variables] if isinstance(variables, str) else variables
        self.variables = variables

        self.resolution = resolution

        variables = [var.replace("ocean_", "") for var in variables]
        base_transform = transform.variables.variable_trim(variables)

        self.depth_value = depth_value
        if depth_value is not None:
            base_transform += transform.coordinates.select(
                {coord: depth_value for coord in ["st_ocean"]}, ignore_missing=True
            )

        super().__init__(
            transforms=base_transform + transforms,
            data_interval=BRAN_TYPES_RESOLUTION[BRAN_RESOLUTION.index(resolution)],
        )

    def filesystem(
        self,
        basetime: str | datetime.datetime | EDITDatetime,
    ) -> Path:
        BRAN_HOME = self.ROOT_DIRECTORIES["BRAN"]

        paths = {}

        basetime = EDITDatetime(str(basetime))
        basetime.set_components(["minute", "second"], False)

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
                    f"Unable to find data for: basetime: {basetime}, variables: {variable} at {var_path}"
                )

        return paths
