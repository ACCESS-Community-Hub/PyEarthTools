# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
ERA5 Copernicus Data Storage Index
"""

from __future__ import annotations
from typing import Literal, Union, Optional
from pathlib import Path
import warnings


import edit.data
from edit.data import EDITDatetime
from edit.data.warnings import EDITDataWarning

from edit.data.indexes import decorators
from edit.data.transforms import Transform, TransformCollection

from edit.data.download.cds.cds import root_cds, as_list
from edit.data.download.cds._ERA5 import (
    ERA5_LEVELS,
    ERA_PRESSURE_NAME_CHANGE,
    ERA_SINGLE_NAME_CHANGE,
)


cds_type = [
    "ensemble_mean",
    "ensemble_members",
    "ensemble_spread",
    "reanalysis",
    "monthly_averaged_ensemble_members",
    "monthly_averaged_ensemble_members_by_hour_of_day",
    "monthly_averaged_reanalysis",
    "monthly_averaged_reanalysis_by_hour_of_day",
]
ERA_TYPE = Literal[
    "ensemble_mean",
    "ensemble_members",
    "ensemble_spread",
    "reanalysis",
    "monthly_averaged_ensemble_members",
    "monthly_averaged_ensemble_members_by_hour_of_day",
    "monthly_averaged_reanalysis",
    "monthly_averaged_reanalysis_by_hour_of_day",
]
ERA_ALL_NAMES = dict(**ERA_SINGLE_NAME_CHANGE, **ERA_PRESSURE_NAME_CHANGE)
LEVEL_TYPE = Union[int, float, list[int], list[float], None]

CDS_RESOLUTION = {"ensemble": (6, "hour"), "reanalysis": (1, "hour")}


def convert_vars(variables: list[str]) -> list[str]:
    """Convert variables to dataset names"""
    update_dict = dict(**ERA_PRESSURE_NAME_CHANGE, **ERA_SINGLE_NAME_CHANGE)
    return [update_dict[var] for var in set(as_list(variables)).intersection(set(update_dict.keys()))]


def get_from_shortname(variables: list[str]) -> list[str]:
    """Convert from variable short name"""

    def invert_dict(dictionary: dict[str, str]) -> dict[str, str]:
        return {val: key for key, val in dictionary.items()}

    pressure_short_name = invert_dict(ERA_PRESSURE_NAME_CHANGE)
    single_short_name = invert_dict(ERA_SINGLE_NAME_CHANGE)

    update_dict = dict(**pressure_short_name, **single_short_name)
    return [update_dict[var] if var in update_dict else var for var in variables]


class ERA5(root_cds):
    """
    ERA5 Access through Copernicus Data Store
    """

    @property
    def _desc_(self):
        return {
            "singleline": "Copernicus Data Store of ECWMF ReAnalysis v5",
            "range": "1940-current",
            "Documentation": "https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-complete",
        }

    @decorators.alias_arguments(
        variables=["variable"],
        level=["levels", "level_value"],
    )
    @decorators.variable_modifications("variables")
    @decorators.check_arguments(
        product=cds_type,
        variables="edit.data.download.cds.variables.ERA5.valid",
        level=ERA5_LEVELS,
    )
    def __init__(
        self,
        variables: list[str] | str,
        *,
        level: LEVEL_TYPE = None,
        cache: str | Path | None = "temp",
        product: ERA_TYPE = "reanalysis",
        transforms: Optional[Transform | TransformCollection] = None,
        download_transforms: Optional[Transform | TransformCollection] = None,
        **kwargs,
    ):
        """
        Access ERA5 through Copernicus

        This will cache data if a path is given. If no `cache` is given,
        data will be downloaded each time.

        Args:
            variables (list[str] | str):
                Variables to retrieve, can be shortname or longname
            level (LEVEL_TYPE, optional):
                Level value to subset pressure levels on. Defaults to None.
            cache (str | Path | None, optional):
                Location to cache data to. Defaults to 'temp'.
            product (CDS_TYPE, optional):
                ERA5 product to retrieve. Defaults to 'reanalysis'.
            transforms (Transform | TransformCollection, optional):
                Base Transforms to apply. Defaults to TransformCollection().
            download_transforms (Transform | TransformCollection, optional):
                Transforms to apply only just once downloaded. Defaults to TransformCollection().
        """
        variables = get_from_shortname(as_list(variables))

        # Remove level select if all single level
        if len(as_list(set(variables).intersection(set(ERA_PRESSURE_NAME_CHANGE.keys())))) == 0:
            level = None

        if level is None and len(as_list(set(variables).intersection(set(ERA_PRESSURE_NAME_CHANGE.keys())))) > 0:
            level = ERA5_LEVELS[1:]
            warnings.warn(
                "As `level` was None, this will now request all levels, which may take a long time.",
                EDITDataWarning,
            )

        self._level = level
        base_transform = TransformCollection()
        base_transform += edit.data.transforms.attributes.Rename({var: ERA_ALL_NAMES[var] for var in variables})
        _download_transforms = download_transforms or TransformCollection()

        # Select on level if needed
        if level is not None and len(as_list(level)) > 1:
            base_transform += edit.data.transforms.coordinates.Select(
                {coord: level for coord in ["level"]},
                ignore_missing=False,
            )
        base_transform += transforms or TransformCollection()

        if level is not None and len(as_list(level)) == 1:
            # warnings.warn(
            #     f"Only getting one level will create a dataset with no level value, which will break things.",
            #     EDITDataWarning
            #     )
            _download_transforms += edit.data.transforms.coordinates.Assign(level=as_list(level))

        kwargs.pop("pattern_kwargs", None)

        super().__init__(
            cache=cache,
            pattern_kwargs=dict(
                variables=convert_vars(as_list(variables)),
                directory_resolution="day",
            ),
            transforms=base_transform,
            data_interval=kwargs.pop(
                "data_interval",
                CDS_RESOLUTION["ensemble" if "ensemble" in product else "reanalysis"],
            ),
            download_transforms=_download_transforms,
            quiet=kwargs.pop("quiet", True),
            **kwargs,
        )
        self.record_initialisation()

        self._variables = variables
        self._product = product

    def _get_from_cds(self, querytime: EDITDatetime | str) -> tuple[str, dict] | list[tuple[str, dict]]:
        """
        Format cds query for data as needed

        Args:
            querytime (EDITDatetime | str):
                Datetime to get data for

        Returns:
            (tuple[str, dict] | list[tuple[str, dict]]):
                Tuple for request or list of tupled requests
        """
        querytime = EDITDatetime(querytime).at_resolution("hour")

        base_dict = {
            "product_type": as_list(self._product),
            "format": "netcdf",
            "year": querytime.year,
            "month": querytime.month,
            "day": querytime.day,
            "time": querytime.strftime("%H:%M"),
        }
        return_list = []

        pressure_request_dictionary = {
            "variable": as_list(set(self._variables).intersection(set(ERA_PRESSURE_NAME_CHANGE.keys()))),
            "pressure_level": [str(x) for x in as_list(self._level)],
        }
        pressure_request_dictionary.update(base_dict)

        if len(pressure_request_dictionary["variable"]) > 0:
            return_list.append(
                (
                    "reanalysis-era5-pressure-levels" + ("-monthly-means" if "monthly" in self._product else ""),
                    pressure_request_dictionary,
                )
            )

        single_request_dictionary = {
            "variable": as_list(set(self._variables).intersection(set(ERA_SINGLE_NAME_CHANGE.keys()))),
        }
        if "geopotential_at_surface" in single_request_dictionary["variable"]:
            single_request_dictionary["variable"].remove("geopotential_at_surface")
            single_request_dictionary["variable"].append("geopotential")

        single_request_dictionary.update(base_dict)

        if len(single_request_dictionary["variable"]) > 0:
            return_list.append(
                (
                    "reanalysis-era5-single-levels" + ("-monthly-means" if "monthly" in self._product else ""),
                    single_request_dictionary,
                )
            )

        return return_list

    @classmethod
    def sample(cls):
        return ERA5("2m_temperature")
