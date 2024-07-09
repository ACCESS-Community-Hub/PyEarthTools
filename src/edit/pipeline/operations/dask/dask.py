# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

# type: ignore[reportPrivateImportUsage]

"""
Dask specific operation
"""

from typing import Type, Union, Optional

import functools
import numpy as np

import edit.utils

from edit.pipeline.operation import Operation


class DaskOperation(Operation):
    """
    Override for Operations with `dask`.

    If set `_numpy_counterpart` use a counterpart numpy class if data given is a numpy array.
    Can be str specifying path after `edit.pipeline.operations.numpy` or full class.
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
                self._numpy_counterpart = edit.utils.dynamic_import(
                    f"edit.pipeline.operations.numpy.{self._numpy_counterpart}"
                )
            with edit.utils.context.ChangeValue(
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
                self._numpy_counterpart = edit.utils.dynamic_import(
                    f"edit.pipeline.operations.numpy.{self._numpy_counterpart}"
                )
            with edit.utils.context.ChangeValue(
                self, "undo_func", functools.partial(self._numpy_counterpart.undo_func, self)
            ):
                sample = super().undo(sample)
            return sample
        return super().undo(sample)
