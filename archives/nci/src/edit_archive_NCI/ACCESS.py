"""
Australian Community Climate and Earth-System Simulator
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Literal
import warnings

import xarray as xr

from edit.data import transform
from edit.data.exceptions import DataNotFoundError, InvalidIndexError
from edit.data.warnings import IndexWarning
from edit.data.indexes import ArchiveIndex, ForecastIndex, DataIndex, decorators
from edit.data.time import EDITDatetime, TimeDelta
from edit.data.transform import Transform, TransformCollection

from edit.data.archive import register_archive

from edit_archive_NCI.utilities import check_project

ACCESS_REGIONS = ["g", "bn", "ad", "sy", "vt", "ph", "nq", "dn"]
FORECAST_INTERVAL = 6
ACCESS_VARIABLE_TYPES = ["ml", "sfc", "pl"]
ACCESS_DATATYPES = ["an", "fc", "fcmm"]

VALID_DATATYPES = Literal["an", "fc", "fcmm"]


def rounder(time: EDITDatetime, interval: int) -> str:
    hour = time.hour
    return "%02d00" % ((hour // interval) * interval,)


@register_archive('ACCESS')
class ACCESS(DataIndex):
    """Index into Australian Community Climate and Earth-System Simulator"""

    @property
    def _desc_(self):
        return {
            "singleline": "Australian Community Climate and Earth-System Simulator",
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

    @decorators.alias_arguments(variables=["variable"])
    @decorators.check_arguments(
        region=ACCESS_REGIONS,
        datatype=ACCESS_DATATYPES,
        variables="edit_archive_NCI.variables.ACCESS.{datatype}.valid",
    )
    def __init__(
        self,
        variables: list[str] | str,
        region: str,
        *,
        datatype: str,
        level_value: Any = None,
        transforms: Transform | TransformCollection = TransformCollection(),
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
        check_project(project_code='wr45')
        variables = [variables] if isinstance(variables, str) else variables

        region = region.lower()

        self.variables = variables
        self.region = region
        self.datatype = datatype

        split_variables = [var.split("/")[-1] for var in variables]

        base_transform = transform.variables.variable_trim(split_variables)

        self.level_value = level_value
        if level_value is not None:
            base_transform += transform.coordinates.select(
                {coord: level_value for coord in ["theta_lvl", "lvl", "rho_lvl"]},
                ignore_missing=True,
            )

        super().__init__(transforms=base_transform + transforms, **kwargs)
        self.make_catalog()

    def filesystem(
        self,
        basetime: str | datetime.datetime,
    ):
        paths = {}

        ACCESS_HOME = self.ROOT_DIRECTORIES["ACCESS"]
        basetime = EDITDatetime(basetime)

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
                    f"Unable to find data for: basetime: {basetime}, region: {self.region}, "
                    f"datatype: {self.datatype}, variable:{variable} at {basepath}"
                )

        return paths

    # -----------------------
    # Static Methods
    # -----------------------

    @staticmethod
    def forecast(*args, **kwargs):
        kwargs["datatype"] = kwargs.pop("datatype", "fc")
        return ACCESS_Forecast(*args, **kwargs)

    @staticmethod
    def analysis(*args, **kwargs):
        # if 'datatype' in kwargs:
        #     raise TypeError("ACCESS.analysis got an unexpected argument 'datatype'")
        kwargs["datatype"] = "an"
        return ACCESS_Analysis(*args, **kwargs)


class ACCESS_Analysis(ACCESS, ArchiveIndex):
    @decorators.alias_arguments(variables=["variable"])
    def __init__(self, variables: list[str] | str, region: str, **kwargs):
        kwargs["data_interval"] = (6, "h") if region.lower() == "g" else (1, "h")
        super().__init__(variables, region, **kwargs)


class ACCESS_Forecast(ACCESS, ForecastIndex):
    @decorators.alias_arguments(variables=["variable"])
    def __init__(
        self,
        variables: list[str] | str,
        region: str,
        datatype: Literal["fc", "fcmm"] = "fc",
        *,
        forecast_leadtime: datetime.timedelta | int | tuple = None,
        transforms: Transform | TransformCollection = TransformCollection(),
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
        super().__init__(variables, region, datatype=datatype, transforms=transforms, **kwargs)
        self.make_catalog()

        self.forecast_leadtime = forecast_leadtime if forecast_leadtime is None else TimeDelta(forecast_leadtime)

    def get(self, querytime: EDITDatetime, *, select_time: EDITDatetime = None, **kwargs) -> xr.Dataset:
        querytime = EDITDatetime(querytime)
        if self.forecast_leadtime:
            querytime = querytime - self.forecast_leadtime
        data = super().get(querytime, **kwargs)

        if select_time:
            data = data.sel(time=select_time)
        return data

    def filesystem(self, basetime: str | EDITDatetime):
        basetime = EDITDatetime(basetime)
        if self.forecast_leadtime:
            basetime = basetime - self.forecast_leadtime
        return super().filesystem(basetime)
