# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from typing import Any, TypeVar, Union, Optional

import xarray as xr

import pyearthtools.data
from pyearthtools.pipeline.operation import Operation

T = TypeVar("T", xr.Dataset, xr.DataArray)
TRANSFORM_TYPE = Union[pyearthtools.data.Transform, pyearthtools.data.TransformCollection]


class Transforms(Operation):
    """
    Run `pyearthtools.data.Transforms` within a `Pipeline`.
    """

    _override_interface = "Serial"

    def __init__(
        self,
        transforms: Optional[TRANSFORM_TYPE] = None,
        apply: Optional[TRANSFORM_TYPE] = None,
        undo: Optional[TRANSFORM_TYPE] = None,
    ):
        """
        Run `Transforms`

        If `transforms` given will run on both functions first, and then if also given `apply` and `undo` respectively.

        Args:
            transforms (Optional[TRANSFORM_TYPE], optional):
                Transforms to run on both `apply` and `undo`. Defaults to None.
            apply (Optional[TRANSFORM_TYPE], optional):
                Transforms to run on `apply`. Defaults to None.
            undo (Optional[TRANSFORM_TYPE], optional):
                Transforms to run on `undo`. Defaults to None.
        """
        super().__init__(split_tuples=True, recursively_split_tuples=True)
        self.record_initialisation()

        self._transforms = pyearthtools.data.TransformCollection() + (transforms if transforms is not None else [])
        self._apply_transforms = pyearthtools.data.TransformCollection() + (apply if apply is not None else [])
        self._undo_transforms = pyearthtools.data.TransformCollection() + (undo if undo is not None else [])

    def apply_func(self, sample: T) -> T:
        sample = self._transforms(sample)
        return self._apply_transforms(sample)

    def undo_func(self, sample: T) -> T:
        sample = self._transforms(sample)
        return self._undo_transforms(sample)
