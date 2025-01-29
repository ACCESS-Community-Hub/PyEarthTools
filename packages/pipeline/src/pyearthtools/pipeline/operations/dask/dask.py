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

"""
Dask specific operation
"""

from typing import Type, Union, Optional

import functools
import numpy as np

import pyearthtools.utils

from pyearthtools.pipeline.operation import Operation


class DaskOperation(Operation):
    """
    Override for Operations with `dask`.

    If set `_numpy_counterpart` use a counterpart numpy class if data given is a numpy array.
    Can be str specifying path after `pyearthtools.pipeline.operations.numpy` or full class.
    """

    _override_interface = ["Serial"]

    # Numpy counterpart function, used if apply or undo hit a np object
    _numpy_counterpart: Optional[Union[str, Type[Operation]]] = None

    def _add_np_to_types(self):
        """Add numpy arrays to recognised types"""
        for func_name in ["apply", "undo"]:
            types = list(self.recognised_types.get(func_name, []))
            if np.ndarray not in types:
                types.append(np.ndarray)
            self.recognised_types[func_name] = types

    def apply(self, sample):
        """Run the `apply_func` on sample, splitting tuples if needed"""
        if not self._operation["apply"]:
            return sample
        if isinstance(sample, np.ndarray) and self._numpy_counterpart is not None:
            self._add_np_to_types()

            if isinstance(self._numpy_counterpart, str):
                self._numpy_counterpart = pyearthtools.utils.dynamic_import(
                    f"pyearthtools.pipeline.operations.numpy.{self._numpy_counterpart}"
                )
            with pyearthtools.utils.context.ChangeValue(
                self, "apply_func", functools.partial(self._numpy_counterpart.apply_func, self)
            ):
                sample = super().apply(sample)
            return sample
        return super().apply(sample)

    def undo(self, sample):
        """Run the `undo_func` on sample, splitting tuples if needed"""
        if not self._operation["undo"]:
            return sample
        if isinstance(sample, np.ndarray) and self._numpy_counterpart is not None:
            self._add_np_to_types()

            if isinstance(self._numpy_counterpart, str):
                self._numpy_counterpart = pyearthtools.utils.dynamic_import(
                    f"pyearthtools.pipeline.operations.numpy.{self._numpy_counterpart}"
                )
            with pyearthtools.utils.context.ChangeValue(
                self, "undo_func", functools.partial(self._numpy_counterpart.undo_func, self)
            ):
                sample = super().undo(sample)
            return sample
        return super().undo(sample)
