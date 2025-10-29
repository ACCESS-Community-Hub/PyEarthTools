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


from typing import TypeVar, Optional

import xarray as xr

from pyearthtools.pipeline.operation import Operation

T = TypeVar("T", xr.Dataset, xr.DataArray)


class AlignDataVariableDimensionsToDatasetCoords(Operation):
    """
    Sometimes, the data variables within a dataset may not be ordered consistently.

    For example a Dataset may have the coordinate dimensions of time, lat and lon.
    Data variable (a) may in indedex by time, lat and lon while variable (b) is indexed by time, lon, lat.

    Such a structure can't be easily converted to numpy due to collisions in the dimensionality.

    This operator will align all of the data variables to the same ordering as present in the Dataset.
    """

    def apply_func(self, data: xr.Dataset) -> xr.Dataset:
        dataset_ordering = list(data.coords)

        data = data.transpose(*dataset_ordering)
        return data

    def undo_func(self, data: xr.Dataset) -> xr.Dataset:
        # TODO: Record all the original orderings and transpose them back, I guess

        return data


class Sort(Operation):
    """
    Sort Variables of an `xarray` object

    Examples
    >>> Sort(order = ['a','b'])
    """

    _override_interface = "Serial"

    def __init__(self, order: Optional[list[str]] = None, strict: bool = False):
        """

        Sort `xarray` variables

        Args:
            order:
                Order to set vars to, if not given sort alphabetically,
                or add others to the end.
                Cannot be None if `strict` is `True`.
                Defaults to None.
            strict:
                Forces all variables to be listed in `order`, and no extras given.
                Defaults to False.
        """
        if order is None:
            order = []

        self.order = list(order)
        self.strict = strict

        super().__init__(
            split_tuples=True,
            operation="apply",
            recognised_types=(xr.Dataset, xr.DataArray),
        )
        self.record_initialisation()

    def apply_func(self, data: xr.Dataset) -> xr.Dataset:
        """Sort an `xarray` object data variables into the given order

        Args:
            data (T):
                `xarray` object to sort.

        Returns:
            (T):
                Sorted dataset
        """
        current_data_vars = list(data.data_vars)
        order = self.order

        if self.strict:

            diff = set(current_data_vars).symmetric_difference(set(order))
            extra_vars = set(current_data_vars) - set(order)
            missing_vars = set(order) - set(current_data_vars)
            if not len(diff) == 0:
                raise RuntimeError(
                    f"When sorting, the data passed {('contained extra: '+ str(extra_vars)) if extra_vars else ''}{' and/or' if extra_vars and missing_vars else ''}{(' missed: '+ str(missing_vars)) if missing_vars else ''}"
                )

        # The default (empty) ordering should be a default-sort-ordering sort
        if order is None or len(order) == 0:
            order = [str(index) for index in current_data_vars]
            order.sort()
            self.order = list(order)

        if not len(order) == len(current_data_vars) or not order == current_data_vars:
            add_to = list([str(index) for index in current_data_vars])
            for var in order:
                if var in add_to:
                    add_to.remove(var)
            order.extend(add_to)
            self.order = list(order)

        order = list(order)
        filtered_order: list = [ord for ord in order if ord in current_data_vars]
        filtered_order = [n for n in filtered_order if n is not None]

        new_data = data[[filtered_order.pop(0)]]

        for key in filtered_order:
            new_data[key] = data[key]

        return new_data
