"""
Bureau of meteorology Atmospheric high-resolution Regional Reanalysis for Australia
"""

from __future__ import annotations

import datetime
import functools
from pathlib import Path
from typing import Any, Literal


from edit.data import transform, DataNotFoundError

from edit.data.indexes import (
    DataIndex,
    ArchiveIndex,
    ForecastIndex,
    StaticDataIndex,
    decorators,
)
from edit.data.time import EDITDatetime
from edit.data.transform import Transform, TransformCollection
from edit.data.archive import register_archive

from edit_archive_NCI.utilities import check_project


BARRA_REGIONS = ["R", "AD", "PH", "SY", "TA"]
BARRA_TYPES = ["forecast", "static", "analysis"]


def rounder(time: datetime.datetime, interval: int) -> str:
    hour = time.hour
    return "%02d00" % ((hour // interval) * interval,)

@register_archive('BARRA')
class BARRA(DataIndex):
    """Index into Bureau of meteorology Atmospheric high-resolution Regional Reanalysis for Australia"""

    @property
    def _desc_(self):
        return {
            "singleline": "Bureau of meteorology Atmospheric high-resolution Regional Reanalysis for Australia",
            "range": "1990-2019",
        }

    def __new__(
        cls,
        variables: str | list[str],
        region: Literal[BARRA_REGIONS],
        datatype: Literal[BARRA_TYPES],
        **kwargs,
    ):
        if datatype == "static":
            cls = BARRA_Static
        elif datatype == "forecast":
            cls = BARRA_Forecast
        else:
            cls = BARRA_Analysis

        return object().__new__(cls)

    @decorators.alias_arguments(variables=["variable"])
    @decorators.check_arguments(
        region=BARRA_REGIONS,
        datatype=BARRA_TYPES,
        variables="edit_archive_NCI.variables.BARRA.{datatype}.valid",
    )
    def __init__(
        self,
        variables: str | list[str],
        region: Literal[BARRA_REGIONS] | str,
        *,
        datatype: Literal[BARRA_TYPES],
        version: str = "v1",
        pressure: float = None,
        transforms: Transform | TransformCollection = TransformCollection(),
        **kwargs,
    ):
        """
        Index into BARRA

        Args:
            variables (str | list[str]):
                Variables to retrieve
            region (Literal[BARRA_REGIONS]):
                BARRA Model to retrieve. Must be one of ['R','AD','PH','SY','TA']
            datatype (Literal[BARRA_TYPES]):
                BARRA Datatype. Must be one of ['forecast','static','analysis']
            version (str, optional):
                BARRA Version. Defaults to 'v1'.
            pressure (float, optional):
                Pressure Level to select. Defaults to None.
            transforms (Transform, optional):
                Base Transforms to apply. Defaults to TransformCollection().

        Raises:
            IndexError:
                If datatype = 'analysis' and region is not 'R", as only R has an analysis product
        """
        check_project(project_code='cj37')

        variables = variables if isinstance(variables, (list, tuple)) else [variables]
        self.region = region

        self.datatype = datatype

        if datatype == "analysis" and not region == "R":
            raise IndexError(f"Only BARRA-R contains an analysis product. It is suggested to use datatype='forecast'")

        self.version = version
        self.variables = variables
        variables = [var.split("/")[-1] for var in variables]

        base_transform = TransformCollection()
        base_transform += transform.variables.variable_trim(variables)

        self.pressure = pressure
        if pressure is not None:
            base_transform += transform.coordinates.select(
                {coord: pressure for coord in ["pressure"]}, ignore_missing=True
            )
        super().__init__(transforms=base_transform + transforms, **kwargs)

    # -------------------
    # Static Type Methods
    # -------------------
    @staticmethod
    def forecast(*args, **kwargs):
        """BARRA Forecast"""
        # if 'datatype' in kwargs:
        #     raise TypeError("BARRA.forecast got an unexpected argument 'datatype'")
        kwargs["datatype"] = "forecast"
        return BARRA_Forecast(*args, **kwargs)

    @staticmethod
    def static(*args, **kwargs):
        """BARRA Static"""
        # if 'datatype' in kwargs:
        #     raise TypeError("BARRA.static got an unexpected argument 'datatype'")
        kwargs["datatype"] = "static"

        return BARRA_Static(*args, **kwargs)

    @staticmethod
    def analysis(*args, **kwargs):
        """BARRA Analysis"""
        # if 'datatype' in kwargs:
        #     raise TypeError("BARRA.analysis got an unexpected argument 'datatype'")
        kwargs["datatype"] = "analysis"
        return BARRA_Analysis(*args, **kwargs)

    def filesystem(self, querytime: EDITDatetime) -> Path:
        querytime = EDITDatetime(querytime)

        BARRA_HOME = self.ROOT_DIRECTORIES["BARRA"]
        basepath = Path(BARRA_HOME.format(region=self.region, version=self.version, datatype=self.datatype))

        paths = {}
        for variable in self.variables:
            var_path = basepath / variable / querytime.strftime("%Y/%m/")
            last_var = variable.split("/")[-1]
            path = list(var_path.glob(f"{last_var}*{querytime.strftime('%Y%m%dT%H%M')}*.nc"))
            if len(path) == 0:
                raise DataNotFoundError(f"Could not find data at {basepath} for {variable} at time {querytime}")
            paths[variable] = path[0]
        return paths


class BARRA_Analysis(BARRA, ArchiveIndex):
    """Index into BARRA"""

    @functools.wraps(BARRA.__init__)
    def __init__(
        self,
        variables: str | list[str],
        region: Any | str,
        *,
        datatype: Literal[BARRA_TYPES] = "analysis",
        version: str = "v1",
        pressure: float = None,
        transforms: Transform | TransformCollection = TransformCollection(),
        **kwargs,
    ):
        kwargs.update(data_interval=(6, "h") if region == "g" else (1, "h"))

        super().__init__(
            variables,
            region,
            datatype=datatype,
            version=version,
            pressure=pressure,
            transforms=transforms,
            **kwargs,
        )
        self.make_catalog()


class BARRA_Forecast(BARRA, ForecastIndex):
    @functools.wraps(BARRA.__init__)
    def __init__(
        self,
        variables: str | list[str],
        region: Any | str,
        *,
        datatype: Literal[BARRA_TYPES] = "forecast",
        version: str = "v1",
        pressure: float = None,
        transforms: Transform | TransformCollection = TransformCollection(),
        **kwargs,
    ):
        super().__init__(
            variables,
            region,
            datatype=datatype,
            version=version,
            pressure=pressure,
            transforms=transforms,
            **kwargs,
        )
        self.make_catalog()

    def filesystem(self, querytime: EDITDatetime) -> Path:
        querytime = EDITDatetime(querytime)

        BARRA_HOME = self.ROOT_DIRECTORIES["BARRA"]
        basepath = Path(BARRA_HOME.format(region=self.region, version=self.version, datatype=self.datatype))

        paths = {}
        for variable in self.variables:
            var_path = basepath / variable / querytime.strftime("%Y/%m/")
            last_var = variable.split("/")[-1]
            path = list(var_path.glob(f"{last_var}*{querytime.strftime('%Y%m%d')}T{rounder(querytime, 6)}*.nc"))

            if len(path) == 0:
                raise DataNotFoundError(f"Could not find data at {basepath} for {variable} at time {querytime}")
            paths[variable] = path[0]
        return paths


class BARRA_Static(BARRA, StaticDataIndex):
    """Static BARRA File Indexer"""

    @functools.wraps(BARRA.__init__)
    def __init__(
        self,
        variables: str | list[str],
        region: Any | str,
        *,
        datatype: Literal[BARRA_TYPES] = "static",
        version: str = "v1",
        pressure: float = None,
        transforms: Transform | TransformCollection = TransformCollection(),
        **kwargs,
    ):
        if not datatype == "static":
            raise ValueError(f"BARRA Static cannot accept datatype's not 'static', {datatype}")

        super().__init__(
            variables,
            region,
            datatype=datatype,
            version=version,
            pressure=pressure,
            transforms=transforms,
            **kwargs,
        )

        self.make_catalog()

    def filesystem(self) -> Path:
        BARRA_HOME = self.ROOT_DIRECTORIES["BARRA"]
        basepath = Path(BARRA_HOME.format(region=self.region, version=self.version, datatype=self.datatype))

        paths = {}
        for variable in self.variables:
            path = list(basepath.glob(f"{variable}*.nc"))
            if len(path) == 0:
                raise DataNotFoundError(f"Could not find data at {basepath} for {variable}")
            paths[variable] = path[0]
        return paths
