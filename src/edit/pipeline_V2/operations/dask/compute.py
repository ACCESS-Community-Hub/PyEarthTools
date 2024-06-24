# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

# type: ignore[reportPrivateImportUsage]

from typing import TypeVar

import dask.array as da

from dask.delayed import Delayed

from edit.pipeline_V2.operation import Operation

T = TypeVar("T", da.Array, Delayed)


class Compute(Operation):
    """
    Compute dask array or delayed object

    If dask array, will convert it to a full numpy array
    """

    _override_interface = "Serial"

    def __init__(self):
        super().__init__(
            split_tuples=True,
            recognised_types=(da.Array, Delayed),
            operation="apply",
        )
        self.record_initialisation()

    def apply_func(self, sample: T) -> T:
        return sample.compute()
