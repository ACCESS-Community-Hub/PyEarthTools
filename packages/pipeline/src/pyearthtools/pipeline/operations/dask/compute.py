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


# type: ignore[reportPrivateImportUsage]

from typing import TypeVar

import dask.array as da

from dask.delayed import Delayed

from pyearthtools.pipeline.operations.dask.dask import DaskOperation

T = TypeVar("T", da.Array, Delayed)


class Compute(DaskOperation):
    """
    Compute dask array or delayed object

    If dask array, will convert it to a full numpy array
    """

    _override_interface = "Serial"

    def __init__(self):
        super().__init__(
            split_tuples=True,
            operation="apply",
        )
        self.record_initialisation()

    def apply_func(self, sample: T) -> T:
        if not hasattr(sample, "compute"):
            return sample
        return sample.compute()
