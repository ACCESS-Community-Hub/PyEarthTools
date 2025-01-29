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


import logging
from typing import Any, Union

import numpy as np
import xarray as xr

import pyearthtools.data
from pyearthtools.data import Transform

LOG = logging.getLogger("pyearthtools.pipeline")


class AddCoordinates(Transform):
    """
    Add coordinates as variable to a dataset

    Use [DropDataset][pyearthtools.pipeline.operations.select.DropDataset] to remove it
    if an earlier step in the pipeline is sensitive to variable names.

    """

    def __init__(self, coordinates: Union[str, list[str]], *extra_coords: str):
        """
        Add coordinates to dataset.

        Args:
            coordinates (str | list[str]):
                Coordinate/s to add
            *extra_coords (str):
                Args form of coordinates
        """
        super().__init__()
        self.record_initialisation()

        coordinates = [coordinates] if isinstance(coordinates, str) else coordinates
        coordinates = [*coordinates, *extra_coords]
        self.coordinates = coordinates

    def apply(self, data: xr.Dataset):
        dims = list(data.dims)
        rebuild_encoding = pyearthtools.data.transforms.attributes.set_encoding(
            reference=data
        ) + pyearthtools.data.transforms.attributes.set_attributes(reference=data)

        for coord in self.coordinates:
            if coord in data:
                new_dims = {}
                for key in (key for key in dims if key not in [coord]):
                    new_dims[key] = np.atleast_1d(data[key].values)

                axis = [list(dims).index(key) for key in new_dims.keys()]
                data[f"var_{coord}"] = data[coord].expand_dims(new_dims, axis=axis)
            else:
                LOG.warn(f"{coord} not found in dataset, which has coords: {list(data.coords)}")
        return rebuild_encoding(data)

    @property
    def _info_(self) -> Any | dict:
        return dict(coordinates=self.coordinates)
