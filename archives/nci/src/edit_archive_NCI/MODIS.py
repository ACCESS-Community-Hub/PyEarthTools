"""
MODerate resolution Imaging Spectroradiometer
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Literal


from edit.data import EDITDatetime, transform
from edit.data.exceptions import DataNotFoundError
from edit.data.indexes import ArchiveIndex, decorators
from edit.data.transform import Transform, TransformCollection
from edit.data.archive import register_archive

from edit_archive_NCI.utilities import check_project


MODIS_REGIONS = ["AU"]
MODIS_RENAME = {"Band1": "lai"}
MODIS_RESOLUTION = ["8-daily", "monthly"]

MODIS_TYPES_RESOLUTION = [(8, "D"), (1, "month")]
MODIS_REGEX = {
    "8-daily": "MOD15A2H.{year_string}_AU_AWRAgrd.nc",
    "monthly": "MOD15A2H.MONTHLY.nc",
}

@register_archive('MODIS')
class MODIS(ArchiveIndex):
    """MODerate resolution Imaging Spectroradiometer

    !!! Note:
        MODIS data exists every 8 days, if data is requested at an invalid day,
        all data will be returned
    """

    @property
    def _desc_(self):
        return {
            "singleline": "MODerate resolution Imaging Spectroradiometer",
            "Range": "2012-2022",
            "Resolution": "8 days",
        }

    @decorators.alias_arguments(resolution=["time", "type", "datatype"], variables="variable")
    @decorators.check_arguments(
        region=MODIS_REGIONS,
        resolution=MODIS_RESOLUTION,
        variables="edit_archive_NCI.variables.MODIS.surface.valid",
    )
    def __init__(
        self,
        variables: list[str] | str,
        region: Literal[MODIS_REGIONS],
        resolution: Literal[MODIS_RESOLUTION],
        *,
        transforms: Transform | TransformCollection = TransformCollection(),
    ):
        """
        Setup MODIS Indexer

        Args:
            variables (list[str] | str):
                Data variables to retrieve
            region (Literal[MODIS_REGIONS]):
                Which Model region subset, currently the only option is 'AU' but there could be more in the future
            resolution (Literal[MODIS_TYPES]):
                Data temporal resolution to retrieve
            transforms (Transform | TransformCollection, optional):
                Base Transforms to apply. Defaults to TransformCollection().
        """
        self.make_catalog()
        check_project(project_code='fj4')

        variables = [variables] if isinstance(variables, str) else variables

        self.region = region
        self.resolution = resolution

        self.variables = variables
        base_transform = TransformCollection()

        base_transform += transform.variables.rename_variables(MODIS_RENAME)
        base_transform += transform.variables.variable_trim(variables)

        # 8 day timesteps... so strange
        super().__init__(
            transforms=base_transform + transforms,
            data_interval=MODIS_TYPES_RESOLUTION[MODIS_RESOLUTION.index(resolution)],
        )

    def filesystem(
        self,
        basetime: str | datetime.datetime | EDITDatetime,
    ) -> Path:
        MODIS_HOME = self.ROOT_DIRECTORIES["MODIS"]

        paths = {}

        basetime = EDITDatetime(basetime)
        basepath = Path(MODIS_HOME.format(region=self.region))

        for variable in self.variables:
            var_path = basepath

            files_in_dir = var_path.iterdir()
            year_string = str(basetime.year)

            relevant_path = None
            for filename in files_in_dir:
                if filename == var_path / MODIS_REGEX[self.resolution].format(year_string=year_string):
                    relevant_path = filename

            if relevant_path is not None:
                relevant_path = var_path / relevant_path

                if relevant_path.exists():
                    paths[variable] = relevant_path
                    continue

            raise DataNotFoundError(
                f"Unable to find data for: basetime: {basetime}, variables: {variable} at {var_path}"
            )
        return paths
