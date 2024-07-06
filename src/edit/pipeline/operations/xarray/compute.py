# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from typing import TypeVar

import xarray as xr

from edit.pipeline.operation import Operation

T = TypeVar("T", xr.Dataset, xr.DataArray)


class Compute(Operation):
    """Compute xarray object"""

    _override_interface = "Serial"

    def __init__(self):
        super().__init__(
            split_tuples=True,
            recognised_types=(xr.Dataset, xr.DataArray),
            operation="apply",
        )
        self.record_initialisation()

    def apply_func(self, sample: T) -> T:
        if hasattr(sample, 'compute'):
            return sample.compute()
        return sample
