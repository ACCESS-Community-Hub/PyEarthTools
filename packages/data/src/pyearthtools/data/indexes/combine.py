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


from __future__ import annotations

import xarray as xr


from pyearthtools.data.indexes import Index, AdvancedTimeDataIndex
from pyearthtools.data import pyearthtoolsDatetime
from pyearthtools.data.operations import SpatialInterpolation, TemporalInterpolation
from pyearthtools.data.transforms.transform import Transform, TransformCollection


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
        querytime: str | pyearthtoolsDatetime,
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
