# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from typing import Any, Literal, Union, Optional

import xarray as xr


import edit.data
from edit.pipeline.operation import Operation


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
        return edit.data.transforms.variables.trim(self.variables)(sample)


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
        return edit.data.transforms.variables.drop(self.variables)(sample)


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
