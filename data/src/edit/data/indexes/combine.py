# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

from __future__ import annotations

import xarray as xr


from edit.data.indexes import Index, AdvancedTimeDataIndex
from edit.data import EDITDatetime
from edit.data.operations import SpatialInterpolation, TemporalInterpolation
from edit.data.transform.transform import Transform, TransformCollection


class InterpolationIndex(AdvancedTimeDataIndex):
    def __init__(
        self,
        *ind,
        indexes: Index | dict = None,
        transforms: Transform | TransformCollection = TransformCollection(),
        data_interval: tuple[int, str] | int = None,
        **kwargs,
    ):
        super().__init__(transforms, data_interval, **kwargs)

    def get(self, *args, **kwargs):
        return super().get(*args, **kwargs)

    def retrieve(
        self,
        querytime: str | EDITDatetime,
        *,
        aggregation: str = None,
        select: bool = True,
        use_simple: bool = False,
        **kwargs,
    ) -> xr.Dataset:
        return super().retrieve(
            querytime,
            aggregation=aggregation,
            select=select,
            use_simple=use_simple,
            **kwargs,
        )
