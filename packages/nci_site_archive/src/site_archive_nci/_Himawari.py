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
import os


import pyearthtools.data
from pyearthtools.data import Petdt, TimeDelta
from pyearthtools.data.exceptions import DataNotFoundError
from pyearthtools.data.indexes import ArchiveIndex, decorators
from pyearthtools.data.transforms import Transform, TransformCollection
from pyearthtools.data.archive import register_archive

from site_archive_nci.utilities import check_project

SATELLITE_PATTERN = "{ROOT_DIR}/{FILE_DATE}/{FILE}"
FILE_REGEX = "*{date_info}*{time_info}*.nc"

VALID_BANDS = [
    "BRF_B01",
    "BRF_B02",
    "BRF_B03",
    "BRF_B04",
    "BRF_B05",
    "BRF_B06",
    "CREFL_B01",
    "CREFL_B02",
    "CREFL_B03",
    "CREFL_B04",
    "CREFL_B05",
    "CREFL_B06",
    "OBS_B01",
    "OBS_B02",
    "OBS_B03",
    "OBS_B04",
    "OBS_B05",
    "OBS_B06",
    "OBS_B07",
    "OBS_B08",
    "OBS_B09",
    "OBS_B10",
    "OBS_B11",
    "OBS_B12",
    "OBS_B13",
    "OBS_B14",
    "OBS_B15",
    "OBS_B16",
]

RESOLUTIONS = {"B03": 500, "B01": 1000, "B02": 1000}

ANC_FILENAME = "/g/data/ra22/satellite-products/arc/obs/himawari-ahi/fldk/latest/ancillary"


# 'GEOM_SOLAR-PRJ_GEOS141_1000-HIMAWARI9-AHI.nc
# 'GEOM_SOLAR-PRJ_GEOS141_2000-HIMAWARI9-AHI.nc
# 'GEOM_SOLAR-PRJ_GEOS141_500-HIMAWARI9-AHI.nc


def check_resolution(bands: list[str]):
    # default_res = 2000
    band_numbers = [b.split("_")[1] for b in bands]
    res = [RESOLUTIONS.get(b, 2000) for b in band_numbers]
    return len(set(res))


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


@register_archive("HimawariChannels")
class HimawariChannels(ArchiveIndex):
    """Index into Himawari 8/9 satellite data"""

    @property
    def _desc_(self):
        return {
            "singleline": "Himawari 8/9 satellite data",
            "Range": "2015-current",
            "Resolution": "10 minutes",
        }

    def __init__(
        self,
        *,
        bands: list[str] | None = None,
        transforms: Transform | TransformCollection | None = None,
    ):
        """
        Setup Satellite Indexer

        Args:
            bands: Which variables to retrieve, can be None to get all. Defaults to None. e.g. 'OBS_B08'
            transforms (Transform | TransformCollection, optional):
                Base Transforms to apply. Defaults to TransformCollection().
        """
        check_project(project_code="ra22")
        data_interval = (10, "m")

        if not all([b in VALID_BANDS for b in bands]):
            raise ValueError("Not all specified bands are valid, please review the requested bands")

        if check_resolution(bands) != 1:
            raise ValueError(
                "Bands specified are at incompatible resolutions, please request each resolution set separately"
            )

        self.bands = bands
        base_transform = pyearthtools.data.transforms.variables.Trim(bands) + (transforms or TransformCollection())
        super().__init__(transforms=base_transform, data_interval=data_interval)
        self.record_initialisation()

    def filesystem(
        self,
        basetime: str | datetime.datetime | Petdt,
    ):
        """
        The resolution on disk is per ten minutes, so match the basetime based on that (i.e. no need to match by hour)
        """
        root_dir = self.ROOT_DIRECTORIES["HimawariChannels"]
        basetime = Petdt(basetime)
        # lastbit = f"{basetime.hour}{basetime.minute}"

        segment = f"{basetime.year}/{basetime.month:02}/{basetime.day:02}/{basetime.hour:02}{basetime.minute:02}"
        files = os.listdir(root_dir + segment)
        files_that_match_bands = [f for f in files if any([b in f for b in self.bands])]
        files_that_match_bands = [f"{root_dir}/{segment}/{f}" for f in files_that_match_bands]

        # import pudb; pudb.set_trace()

        if not files_that_match_bands:
            raise DataNotFoundError(f"Unable to find data for: basetime: {basetime} at {root_dir}")

        return files_that_match_bands
