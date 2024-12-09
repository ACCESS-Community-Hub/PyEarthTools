# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

# type: ignore[reportPrivateImportUsage]

from typing import Optional, Any

import dask.array as da

from pyearthtools.pipeline.branching.join import Joiner
from pyearthtools.pipeline.operations.dask.dask import DaskOperation


class Stack(Joiner, DaskOperation):
    """
    Stack a tuple of da.Array's

    Currently cannot undo this operation
    """

    _override_interface = ["Serial"]
    _numpy_counterpart = "join.Stack"

    def __init__(self, axis: Optional[int] = None):
        super().__init__()
        self.record_initialisation()
        self.axis = axis

    def join(self, sample: tuple[Any, ...]) -> da.Array:
        """Join sample"""
        return da.stack(sample, self.axis)  # type: ignore

    def unjoin(self, sample: Any) -> tuple:
        return super().unjoin(sample)


class VStack(Joiner, DaskOperation):
    """
    Vertically Stack a tuple of da.Array's

    Currently cannot undo this operation
    """

    _override_interface = ["Serial"]
    _numpy_counterpart = "join.VStack"

    def __init__(self):
        super().__init__()
        self.record_initialisation()

    def join(self, sample: tuple[Any, ...]) -> da.Array:
        """Join sample"""
        return da.vstack(
            sample,
        )  # type: ignore

    def unjoin(self, sample: Any) -> tuple:
        return super().unjoin(sample)


class HStack(Joiner, DaskOperation):
    """
    Horizontally Stack a tuple of da.Array's

    Currently cannot undo this operation
    """

    _override_interface = ["Serial"]
    _numpy_counterpart = "join.HStack"

    def __init__(self):
        super().__init__()
        self.record_initialisation()

    def join(self, sample: tuple[Any, ...]) -> da.Array:
        """Join sample"""
        return da.hstack(
            sample,
        )  # type: ignore

    def unjoin(self, sample: Any) -> tuple:
        return super().unjoin(sample)


class Concatenate(Joiner, DaskOperation):
    """
    Concatenate a tuple of da.Array's

    Currently cannot undo this operation
    """

    _override_interface = ["Serial"]
    _numpy_counterpart = "join.Concatenate"

    def __init__(self, axis: Optional[int] = None):
        super().__init__()
        self.record_initialisation()
        self.axis = axis

    def join(self, sample: tuple[Any, ...]) -> da.Array:
        """Join sample"""
        return da.concatenate(sample, self.axis)  # type: ignore

    def unjoin(self, sample: Any) -> tuple:
        return super().unjoin(sample)
