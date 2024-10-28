# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations
import functools

import xarray as xr
from os import PathLike

import edit.data
from edit.data.time import EDITDatetime

from edit.data.indexes import AdvancedTimeDataIndex, decorators, CachingIndex
from edit.data.transforms.transform import Transform, TransformCollection

from edit.data.patterns.expanded_date import ExpandedDateVariable

from edit.data.download.arco.variables.ERA5 import (
    ERA5_LEVELS,
    ERA_NAME_CHANGE,
)

ROOT_ARCO_DS = None


def convert_vars(variables: list[str]) -> list[str]:
    """Convert variables to dataset names"""
    update_dict = dict(ERA_NAME_CHANGE)
    return [
        update_dict[var]
        for var in set([variables] if not isinstance(variables, list) else variables).intersection(
            set(update_dict.keys())
        )
    ]


def get_from_shortname(variables: list[str]) -> list[str]:
    """Convert from variable short name"""

    def invert_dict(dictionary: dict[str, str]) -> dict[str, str]:
        return {val: key for key, val in dictionary.items()}

    short_name = invert_dict(ERA_NAME_CHANGE)

    return [short_name[var] if var in short_name else var for var in variables]


class ARCOERA5(AdvancedTimeDataIndex):
    """
    Analysis-Ready, Cloud Optimized ERA5

    https://github.com/google-research/arco-era5

    Carver, Robert W, and Merose, Alex. (2023):
    ARCO-ERA5: An Analysis-Ready Cloud-Optimized Reanalysis Dataset.
    22nd Conf. on AI for Env. Science, Denver, CO, Amer. Meteo. Soc, 4A.1,
    https://ams.confex.com/ams/103ANNUAL/meetingapp.cgi/Paper/415842
    """

    _desc_ = {
        "singleline": "Analysis-Ready, Cloud Optimized ERA5",
        "link": "https://github.com/google-research/arco-era5",
    }

    _ds: xr.Dataset

    @decorators.alias_arguments(
        variables=["variable"],
        level=["levels", "level_value"],
    )
    @decorators.check_arguments(
        variables="edit.data.download.arco.variables.ERA5.valid",
        level=ERA5_LEVELS,
    )
    @decorators.variable_modifications("variables")
    def __init__(
        self,
        variables: str | list[str],
        level: int | list[int] | None = None,
        transforms: Transform | TransformCollection | None = None,
        **kwargs,
    ):
        """
        Analysis-Ready, Cloud Optimized ERA5 integrated within `EDIT`.

        Allows for access to a cloud ERA5 archive.

        Args:
            variables (str | list[str]):
                Variables to retrieve, can be either short_name or long_name
            level (int | list[int] | None, optional):
                Pressure levels to select. Defaults to None.
            transforms (Transform | TransformCollection | None, optional):
                Transforms to apply to dataset. Defaults to None.
        """
        super().__init__(transforms or TransformCollection(), data_interval="1 hour")
        self.record_initialisation()

        variables = get_from_shortname([variables] if not isinstance(variables, list) else variables)
        self.variables = variables
        self.level = level
        self._kwargs = kwargs

        self._get_zarr_file()

    def _get_zarr_file(self):
        global ROOT_ARCO_DS

        if ROOT_ARCO_DS is None:
            ds = xr.open_zarr(
                "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
                chunks=self._kwargs.pop("chunks", "auto"),
                storage_options=dict(token="anon"),
                **self._kwargs,
            )
            ROOT_ARCO_DS = ds
        else:
            ds = ROOT_ARCO_DS

        self._ds = ds[self.variables]

        if self.level is not None:
            self._ds = edit.data.transform.coordinates.Select(level=self.level, ignore_missing=True)(self._ds)

    @property
    def full_ds(self) -> xr.Dataset:
        """Get full ARCO ds"""
        assert ROOT_ARCO_DS is not None
        return ROOT_ARCO_DS

    @property
    def dataset(self) -> xr.Dataset:
        """Get full dataset for this obj"""
        return self._ds

    def get(self, time: str):
        """Get timestep from dataset"""
        return self._ds.sel(time=EDITDatetime(time).datetime64())

    @classmethod
    def sample(cls):
        return ARCOERA5("2m_temperature")


# class CachingARCOERA5(CachingIndex):
#     def __init__(self,
#         cache: PathLike,
#         variables: str | list[str],
#         level: int | list[int] | None = None,
#         transforms: Transform | TransformCollection | None = None,
#         **kwargs,
#         ):
#         super().__init__(cache = str(cache), pattern=ExpandedDateVariable, data_interval = '1 hour', **kwargs)
#         self._arco = ARCOERA5(variables, level=level, transforms = transforms)

#     def _generate(self, querytime) -> xr.Dataset:
#         return self._arco[querytime]

#     def get(self, querytime) -> xr.Dataset:
#         self.cleanup()
#         self.save_record()

#         if self.cache is None and self.pattern_type is None:
#             return self._generate(querytime)

#         dataset = self._generate(querytime)

#         return xr.apply_ufunc(
#             functools.partial(self.pattern.save, querytime = querytime),
#             dataset,
#             input_core_dims=[['latitude', 'longitude']],
#         )
