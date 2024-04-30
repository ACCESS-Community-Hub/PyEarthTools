"""
ECWMF ReAnalysis v5
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal


from edit.data import EDITDatetime, transform
from edit.data.exceptions import DataNotFoundError
from edit.data.indexes import ArchiveIndex, decorators
from edit.data.transform import Transform, TransformCollection
from edit.data.archive import register_archive

from edit_archive_NCI.utilities import check_project

ERA5_LEVELS = ["single", "pressure"]
ERA_PROD = ["monthly-averaged", "monthly-averaged-by-hour", "reanalysis"]
ERA_RES_RESOLUTION = [(1, "month"), (1, "month"), (1, "hour")]

ERA5_RENAME = {"t2m": "2t", "u10": "10u", "v10": "10v"}

#TODO allow level to be inferred from each var

@register_archive('ERA5')
class ERA5(ArchiveIndex):
    """ECWMF ReAnalysis v5"""

    @property
    def _desc_(self):
        return {
            "singleline": "ECWMF ReAnalysis v5",
            "range": "1970-current",
            "Documentation": "https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation",
        }

    @decorators.alias_arguments(
            level_value=["pressure"], 
            variables=["variable"],
            product=['resolution'],
            )
    @decorators.check_arguments(
        level=ERA5_LEVELS,
        product=ERA_PROD,
        variables="edit_archive_NCI.variables.ERA5.{level}.{product}.valid",
    )
    def __init__(
        self,
        variables: list[str] | str,
        *,
        level: Literal[ERA5_LEVELS],
        product: Literal[ERA_PROD] = "reanalysis",
        level_value: Any = None,
        transforms: Transform | TransformCollection = TransformCollection(),
    ):
        """
        Setup ERA5 Indexer

        Args:
            variables (list[str] | str):
                Data variables to retrieve
            level (Literal[ERA5_LEVELS]):
                Model level of data, must be either "single", "pressure"
            resolution (Literal[ERA_RES], optional):
                Resolution of data, must be one of 'monthly-averaged','monthly-averaged-by-hour', 'reanalysis'. Defaults to 'reanalysis'.
            level_value: (int, optional):
                Level value to select if data contains levels. Defaults to None.
            transforms (Transform | TransformCollection, optional): 
                Base Transforms to apply.
                Defaults to TransformCollection().
        """
        self.make_catalog()
        check_project(project_code='rt52')

        variables = [variables] if isinstance(variables, str) else variables

        self.level = level
        self.resolution = product

        if level_value and not level == "pressure":
            raise KeyError(f"Pressure level cannot be set if level != 'pressure'")

        self.variables = variables
        base_transform = TransformCollection()

        base_transform += transform.variables.rename_variables(ERA5_RENAME)
        base_transform += transform.variables.variable_trim(variables)

        self.level_value = level_value

        if level_value:
            base_transform += transform.coordinates.select(
                {coord: level_value for coord in ["level"]}, ignore_missing=True
            )

        super().__init__(
            transforms=base_transform + transforms,
            data_interval=ERA_RES_RESOLUTION[ERA_PROD.index(product)],
        )

    def filesystem(
        self,
        querytime: str | EDITDatetime,
    ) -> Path:
        ERA5_HOME = self.ROOT_DIRECTORIES["ERA5"]

        paths = {}

        querytime = EDITDatetime(querytime)
        basepath = Path(ERA5_HOME.format(level=self.level, resolution=self.resolution))

        for variable in self.variables:
            var_path = basepath / variable / str(querytime.year)

            files_in_dir = var_path.iterdir()
            start_of_month_string = querytime.replace(day=1).strftime("%Y%m%d")

            relevant_path = None
            for filename in files_in_dir:
                if start_of_month_string in str(filename):
                    relevant_path = filename

            if relevant_path is not None:
                relevant_path = var_path / relevant_path

                if relevant_path.exists():
                    paths[variable] = relevant_path
                    continue

            raise DataNotFoundError(
                f"Unable to find data for: basetime: {querytime}, variables: {variable} at {var_path}"
            )
        return paths
