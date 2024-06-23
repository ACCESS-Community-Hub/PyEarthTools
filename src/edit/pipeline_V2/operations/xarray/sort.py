# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from typing import TypeVar, Optional

import xarray as xr

from edit.pipeline_V2.operation import Operation

T = TypeVar("T", xr.Dataset, xr.DataArray)


class Sort(Operation):
    """
    Sort Variables of an `xarray` object

    Examples
    >>> Sort(order = ['a','b'])
    """
    _override_interface = 'Serial'

    def __init__(self, order: Optional[list[str]], safe: bool = False):
        """

        Sort `xarray` variables

        Args:
            order (list[str], optional):
                Order to set vars to, if not given sort alphabetically,
                or add others alphabetically to the end.
                Cannot be None if `safe` is `True`.
                Defaults to None.
            safe (bool, optional):
                Forces all variables to be listed in `order`, and no extras given.
                Defaults to False.
        """
        if order is None:
            order = []

        self.order = list(order)
        self.safe = safe

        super().__init__(
            split_tuples=True,
            operation="apply",
            recognised_types=(xr.Dataset, xr.DataArray),
        )
        self.record_initialisation()

    def apply_func(self, data: T) -> T:
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

        if self.safe:
            diff = set(current_data_vars).difference(set(order)).union(set(order).difference(set(current_data_vars)))
            extra_vars = set(current_data_vars) - set(order)
            missing_vars = set(order) - set(current_data_vars)
            if not len(diff) == 0:
                raise RuntimeError(
                    f"When sorting, the data passed {('contained extra: '+ str(extra_vars)) if extra_vars else ''}{' and/or' if extra_vars and missing_vars else ''}{( 'missed: '+ str(missing_vars)) if missing_vars else ''}"
                )

        if order is None or len(order) == 0:
            order = [str(index) for index in current_data_vars]
            order.sort()
            self.order = list(order)

        if not len(order) == len(current_data_vars) or not order == current_data_vars:
            add_to = list(set([str(index) for index in current_data_vars]).difference(set(order)))
            add_to.sort()
            order.extend(add_to)
            self.order = list(order)

        order = list(order)
        filtered_order: list = [ord for ord in order if ord in current_data_vars]
        while None in filtered_order:
            filtered_order.remove(None)

        new_data = data[[filtered_order.pop(0)]]

        for key in filtered_order:
            new_data[key] = data[key]

        return new_data
