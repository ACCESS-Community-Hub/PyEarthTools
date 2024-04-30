"""
Himiwari 8/9 satellite data
"""

from __future__ import annotations

import datetime
from glob import glob
from pathlib import Path


from edit.data import EDITDatetime, transform
from edit.data.exceptions import DataNotFoundError
from edit.data.indexes import ArchiveIndex, decorators
from edit.data.transform import Transform, TransformCollection
from edit.data.archive import register_archive

from edit_archive_NCI.utilities import check_project

SATELLITE_PATTERN = "{ROOT_DIR}/{FILE_DATE}/{FILE}"
FILE_REGEX = "*{date_info}*{time_info}*.nc"

@register_archive('Himiwari')
class Himiwari(ArchiveIndex):
    """Index into Himiwari 8/9 satellite data"""

    @property
    def _desc_(self):
        return {
            "singleline": "Himiwari 8/9 satellite data",
            "Range": "2019-current",
            "Resolution": "10 minutes",
        }

    @decorators.alias_arguments(variables=["variable"])
    def __init__(
        self,
        variables: list[str] | str | None = None,
        *,
        file_regex: str = FILE_REGEX,
        data_interval: tuple[int, str] = (10, "m"),
        transforms: Transform | TransformCollection = TransformCollection(),
    ):
        """
        Setup Satellite Indexer

        Args:
            variables (list[str], str | None, optional):
                Which variables to retrieve, can be None to get all. Defaults to None.
            file_regex (str, optional):
                File Regular expression, use date_info & time_info as keys. Defaults to  "*{date_info}*{time_info}*.nc".
            data_interval (tuple[int, str], optional):
                Override for data resolution. Defaults to (10, "m").
            transforms (Transform | TransformCollection, optional):
                Base Transforms to apply. Defaults to TransformCollection().
        """
        self.make_catalog()
        check_project(project_code='rv74')

        variables = [variables] if isinstance(variables, str) else variables

        self.variables = variables
        self.file_regex = file_regex

        base_transform = transform.variables.variable_trim(variables) + transforms
        super().__init__(transforms=base_transform, data_interval=data_interval or (10, "m"))

    def filesystem(
        self,
        basetime: str | datetime.datetime | EDITDatetime,
    ):
        root_dir = self.ROOT_DIRECTORIES["HIMIWARI"]
        basetime = EDITDatetime(basetime)

        offset = datetime.timedelta(days=1)
        check_dates = [basetime - offset, basetime, basetime + offset]

        for dates in check_dates:
            basepath = Path(root_dir) / dates.strftime("%Y/%m/%d")
            file_search = basepath / self.file_regex.format(
                date_info=basetime.strftime("%Y%m%d"),
                time_info=basetime.strftime("%H%M"),
            )

            resolved_names = [Path(p) for p in glob(str(file_search))]

            for file in resolved_names:
                if file.exists():
                    return file

        raise DataNotFoundError(
            f"Unable to find data for: basetime: {basetime} at {root_dir}\nAttempted to use {resolved_names}"
        )
