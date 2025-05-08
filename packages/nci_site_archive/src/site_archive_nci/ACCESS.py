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
Australian Community Climate and Earth-System Simulator
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Literal
import warnings
import functools

import xarray as xr

import pyearthtools.data
from pyearthtools.data.exceptions import DataNotFoundError
from pyearthtools.data.warnings import IndexWarning
from pyearthtools.data.indexes import ArchiveIndex, ForecastIndex, DataFileSystemIndex, decorators
from pyearthtools.data.time import Petdt, TimeDelta
from pyearthtools.data.transforms import Transform, TransformCollection

from pyearthtools.data.archive import register_archive

from site_archive_nci.utilities import check_project

ACCESS_REGIONS = ["g", "bn", "ad", "sy", "vt", "ph", "nq", "dn"]
ACCESS_REGIONS = [*ACCESS_REGIONS, *(x.upper() for x in ACCESS_REGIONS)]
FORECAST_INTERVAL = 6
ACCESS_VARIABLE_TYPES = ["ml", "sfc", "pl"]
ACCESS_DATATYPES = ["an", "fc", "fcmm"]

VALID_DATATYPES = Literal["an", "fc", "fcmm"]


def rounder(time: Petdt, interval: int) -> str:
    hour = time.hour
    return "%02d00" % ((hour // interval) * interval,)


class ACCESS_UI_MIXIN:
    # -----------------------
    # User friendly Methods
    # -----------------------

    @staticmethod
    @functools.wraps("ACCESS")
    def forecast(*args, **kwargs):
        kwargs["datatype"] = kwargs.pop("datatype", "fc")
        return ACCESS_Forecast(*args, **kwargs)

    @staticmethod
    @functools.wraps("ACCESS")
    def analysis(*args, **kwargs):
        # if 'datatype' in kwargs:
        #     raise TypeError("ACCESS.analysis got an unexpected argument 'datatype'")
        kwargs["datatype"] = "an"
        return ACCESS_Analysis(*args, **kwargs)


@register_archive("ACCESS", sample_kwargs=dict(variable="sfc/temp_scrn", datatype="an", region="g"))
class ACCESS(DataFileSystemIndex, ACCESS_UI_MIXIN):
    """Index into Australian Community Climate and Earth-System Simulator"""

    @property
    def _desc_(self):
        return {
            "singleline": "Australian Community Climate and Earth-System Simulator",
            "Documentation": "http://www.bom.gov.au/nwp/doc/access/NWPData.shtml",
        }

    @decorators.alias_arguments(variables=["variable"])
    def __new__(
        cls,
        variables: list[str] | str,
        region: str,
        *,
        datatype: VALID_DATATYPES,
        **kwargs,
    ):
        if datatype in ["fc", "fcmm"]:
            cls = ACCESS_Forecast
        else:
            cls = ACCESS_Analysis

        return super().__new__(cls)

    @decorators.alias_arguments(variables=["variable"], level_value=["level"])
    @decorators.check_arguments(
        region=ACCESS_REGIONS,
        datatype=ACCESS_DATATYPES,
        variables="site_archive_nci.variables.ACCESS.{datatype}.valid",
    )
    def __init__(
        self,
        variables: list[str] | str,
        region: str,
        *,
        datatype: str,
        level_value: Any = None,
        transforms: Transform | TransformCollection | None = None,
        **kwargs,
    ):
        """
        Setup ACCESS Index Class

        Args:
            variables (list[str] | str):
                Variables to retrieve
            region (str):
                ACCESS Region Code - ['g','bn','ad','sy','vt','ph','nq','dn']
            datatype (str):
                ACCESS Datatype Code - ['an', 'fc', 'fcmm']
            level_value: (int, optional):
                Level value to select if data contains levels. Defaults to None.
            transforms (Transform | TransformCollection, optional):
                Base Transforms to apply. Defaults to TransformCollection().
        """
        check_project(project_code="wr45")
        variables = [variables] if isinstance(variables, str) else variables

        region = region.lower()

        self.variables = variables
        self.region = region.lower()
        self.datatype = datatype

        split_variables = [var.split("/")[-1] for var in variables]

        base_transform = pyearthtools.data.transforms.variables.Trim(split_variables)

        self.level_value = level_value
        if level_value is not None:
            base_transform += pyearthtools.data.transforms.coordinates.Select(
                {coord: level_value for coord in ["theta_lvl", "lvl", "rho_lvl"]},
                ignore_missing=True,
            )

        super().__init__(transforms=base_transform + (transforms or TransformCollection()), **kwargs)
        self.record_initialisation()

    def load(self, *args, **kwargs) -> Any:
        """Load access data, accounting for coord issues"""
        e = None
        try:
            new_kwargs = dict(kwargs)
            new_kwargs["compat"] = "override"
            # new_kwargs['coords'] = 'all'
            return super().load(*args, **new_kwargs)
        except Exception as excep:
            try:
                return super().load(*args, **kwargs)
            except Exception:
                e = excep
        raise e

    def filesystem(
        self,
        basetime: str | Petdt,
    ):
        paths = {}

        ACCESS_HOME = self.ROOT_DIRECTORIES["ACCESS"]
        basetime = Petdt(basetime)

        basepath = Path(ACCESS_HOME.format(region=self.region))

        interval = FORECAST_INTERVAL if self.region == "g" or self.datatype in ["fc", "fcmm"] else 1
        if not int(basetime.hour) % interval == 0:
            warnings.warn(
                f"Data exists at {interval} hourly intervals, {basetime} is thus invalid. Rounding down...",
                IndexWarning,
            )

        for variable in self.variables:
            var_path = (
                basepath
                / basetime.strftime("%Y%m%d")
                / rounder(
                    basetime,
                    interval,
                )
                / self.datatype
            )

            relevant_files = var_path.rglob(f"{variable}.nc")

            for relevant_path in relevant_files:
                if relevant_path.exists():
                    paths[variable.split("/")[-1]] = relevant_path
                    break
            else:
                raise DataNotFoundError(
                    f"Unable to find data for: basetime: {basetime!r}, region: {self.region!r}, "
                    f"datatype: {self.datatype!r}, variable:{variable!r} at {basepath!r}"
                )

        return paths


class ACCESS_Analysis(ACCESS, ArchiveIndex):
    @functools.wraps(ACCESS.__init__)
    @decorators.alias_arguments(variables=["variable"])
    @decorators.variable_modifications(variable_keyword="variables", remove_variables=False)
    def __init__(self, variables: list[str] | str, region: str, **kwargs):
        kwargs["data_interval"] = (6, "h") if region.lower() == "g" else (1, "h")
        super().__init__(variables, region=region, **kwargs)

    def series(self, *args, **kwargs) -> Any:
        """Load access data, accounting for coord issues"""
        e = None
        try:
            new_kwargs = dict(kwargs)
            new_kwargs["compat"] = "override"
            new_kwargs["coords"] = "all"
            return super().series(*args, **new_kwargs)
        except Exception as excep:
            try:
                return super().series(*args, **kwargs)
            except Exception:
                e = excep
        raise e


class ACCESS_Forecast(ACCESS, ForecastIndex):
    @decorators.alias_arguments(variables=["variable"])
    def __init__(
        self,
        variables: list[str] | str,
        region: str,
        datatype: Literal["fc", "fcmm"] = "fc",
        *,
        forecast_leadtime: datetime.timedelta | int | tuple | None = None,
        transforms: Transform | TransformCollection | None = None,
        **kwargs,
    ):
        """
        Setup ACCESS Analysis Class

        Args:
            variables (list[str] | str):
                Variables to retrieve
            region (str):
                ACCESS Region
            datatype (Literal["fc", "fcmm"], optional):
                forecast product to get from. Defaults to "fc".
            forecast_leadtime (datetime.timedelta | int | tuple, optional):
                Minimum forecast lead time in minutes. Use pd.timedelta Notation (int, str). Defaults to None.
            transforms (Transform | TransformCollection, optional):
                Base Transforms to apply. Defaults to TransformCollection().
        """
        kwargs["data_interval"] = (6, "h")
        super().__init__(variables, region=region, datatype=datatype, transforms=transforms, **kwargs)
        self.record_initialisation()

        self.forecast_leadtime = forecast_leadtime if forecast_leadtime is None else TimeDelta(forecast_leadtime)

    def get(self, querytime: Petdt, *, select_time: Petdt | None = None, **kwargs) -> xr.Dataset:
        querytime = Petdt(querytime)
        if self.forecast_leadtime:
            querytime = querytime - self.forecast_leadtime
        data = super().get(querytime, **kwargs)

        if select_time:
            data = data.sel(time=select_time)
        return data

    def filesystem(self, basetime: str | Petdt):
        basetime = Petdt(basetime)
        if self.forecast_leadtime:
            basetime = basetime - self.forecast_leadtime
        return super().filesystem(basetime)
