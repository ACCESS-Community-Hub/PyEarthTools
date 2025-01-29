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


from typing import Any, Literal, Union, Optional

import xarray as xr


import pyearthtools.data
from pyearthtools.pipeline.operation import Operation


class SelectDataset(Operation):
    """
    Operation to select a given set of variables from a [Dataset][xarray.Dataset]
    """

    _override_interface = "Serial"

    def __init__(
        self,
        variables,
        operation: Literal["apply", "undo"] = "apply",
    ):
        """Select variables from dataset

        Args:
            variables ():
                Variables to select
            operation (Literal['apply', 'undo'], optional):
                Operation to run on. Defaults to 'apply'.
        """

        super().__init__(
            operation=operation,
            split_tuples=True,
            recognised_types=(xr.Dataset),
        )
        self.record_initialisation()
        self.variables = variables

    def apply_func(self, sample: xr.Dataset):
        return pyearthtools.data.transforms.variables.trim(self.variables)(sample)


class DropDataset(Operation):
    """
    DataOperation to drop a given set of variables from a [Dataset][xarray.Dataset]

    Can be used to remove variables when undoing, if one was added as a pipeline step.
    """

    _override_interface = "Serial"

    def __init__(
        self,
        variables,
        operation: Literal["apply", "undo"] = "apply",
    ):
        """Drop variables from dataset

        Args:
            variables ():
                Variables to drop
            operation (Literal['apply', 'undo'], optional):
                Operation to run on. Defaults to 'apply'.
        """

        super().__init__(
            operation=operation,
            split_tuples=True,
            recognised_types=(xr.Dataset),
        )
        self.record_initialisation()
        self.variables = variables

    def apply_func(self, sample: xr.Dataset):
        return pyearthtools.data.transforms.variables.drop(self.variables)(sample)


class SliceDataset(Operation):
    """
    Select a slice of an xarray object

    Examples
        >>> Slicer(slices = {'time': (0,10,2)}) # == .sel(time = slice(0,10,2))

    """

    _override_interface = "Serial"

    def __init__(self, slices: Optional[dict[str, tuple[Any, ...]]] = None, **kwargs: tuple):
        """
        Setup dataset slicer

        Args:
            slices (Optional[dict[str, tuple[Any, ...]]], optional):
                Slice dictionary, must be key of dim in ds, and slice notation as value. Defaults to None.
            kwargs (tuple, optional):
                Keyword argument form of `slices`.
        """
        if slices is None:
            slices = {}

        super().__init__(
            operation="apply",
            split_tuples=True,
            recognised_types=(xr.Dataset, xr.DataArray),
        )
        self.record_initialisation()
        slices.update(kwargs)

        self.slices = {key: slice(*value) for key, value in slices.items()}

    def apply_func(self, data: Union[xr.Dataset, xr.DataArray]):
        return data.sel(self.slices)
