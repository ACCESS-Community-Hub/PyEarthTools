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
Himawari 8/9 satellite data
"""

from __future__ import annotations

import datetime
from glob import glob
from pathlib import Path


import pyearthtools.data
from pyearthtools.data import Petdt, TimeDelta
from pyearthtools.data.exceptions import DataNotFoundError
from pyearthtools.data.indexes import ArchiveIndex, decorators
from pyearthtools.data.transforms import Transform, TransformCollection
from pyearthtools.data.archive import register_archive

from site_archive_nci.utilities import check_project

SATELLITE_PATTERN = "{ROOT_DIR}/{FILE_DATE}/{FILE}"
FILE_REGEX = "*{date_info}*{time_info}*.nc"


@register_archive("Himawari")
class Himawari(ArchiveIndex):
    """Index into Himawari 8/9 satellite data"""

    @property
    def _desc_(self):
        return {
            "singleline": "Himawari 8/9 satellite data",
            "Range": "2019-current",
            "Resolution": "10 minutes",
        }

    @decorators.alias_arguments(variables=["variable"])
    @decorators.variable_modifications(variable_keyword="variables")
    def __init__(
        self,
        variables: list[str] | str | None = None,
        *,
        file_regex: str = FILE_REGEX,
        data_interval: tuple[int, str] = (10, "m"),
        transforms: Transform | TransformCollection | None = None,
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
        check_project(project_code="rv74")

        variables = [variables] if isinstance(variables, str) else variables

        self.variables = variables
        self.file_regex = file_regex

        base_transform = pyearthtools.data.transforms.variables.Trim(variables) + (transforms or TransformCollection())
        super().__init__(transforms=base_transform, data_interval=data_interval or (10, "m"))
        self.record_initialisation()

    def filesystem(
        self,
        basetime: str | datetime.datetime | Petdt,
    ):
        root_dir = self.ROOT_DIRECTORIES["Himawari"]
        basetime = Petdt(basetime)

        offset = TimeDelta(1, "day")
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
