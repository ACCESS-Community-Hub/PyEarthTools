
from __future__ import annotations

import numpy as np
import xarray as xr

from edit.training.data.templates import (
    DataIterator,
    DataStep,
    DataOperation,
)
from edit.training.data.sequential import SequentialIterator


@SequentialIterator
class xarraySorter(DataOperation):
    """
    Sort Variables of an xarray object

    !!! Example
        ```python
        xarraySorter(PipelineStep, order = ['a','b'])

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialxarraySorter = xarraySorter(order = ['a','b'])
        partialxarraySorter(PipelineStep)
        ```
    """
    def __init__(self, index: DataStep, order : list[str] = None):
        """Initialise sorter

        Args:
            index (DataStep): 
                Underlying DataStep to get data for
            order (list[str], optional): 
                Order to set vars to, if not given sort alphabetically, or add others alphabetically to the end. Defaults to None.
        """        
        self.order = order
        super().__init__(index, apply_func=self.sort, undo_func=self.sort, split_tuples=True, recognised_types=(xr.Dataset, xr.DataArray))

    
    def sort(self, data : xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
        """Sort an xarray object data variables into the given order

        Args:
            data (xr.Dataset): 
                Dataset to sort

        Returns:
            (xr.Dataset | xr.DataArray): 
                Sorted dataset
        """
        current_data_vars = data.data_vars
        order = self.order

        if order is None:
            order = [str(index) for index in current_data_vars]
            order.sort()
            self.order = list(order)

        if not len(order) == len(current_data_vars):
            add_to = list(set([str(index) for index in current_data_vars]).difference(set(order)))
            add_to.sort()
            order = [*order, *add_to]
            self.order = list(order)


        order = list(order)
        filtered_order: list = [ord for ord in order if ord in current_data_vars]
        while None in filtered_order:
            filtered_order.pop(None)

        new_data = data[filtered_order.pop(0)].to_dataset()
        for key in filtered_order:
            new_data[key] = data[key]
        return new_data

    @property
    def _info_(self):
        return dict(order = self.order)