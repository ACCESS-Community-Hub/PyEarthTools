# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

#type: ignore[reportPrivateImportUsage]

from typing import Optional, Any

import dask.array as da

from edit.pipeline_V2.branching.join import Joiner


class Stack(Joiner):
    """
    Stack a tuple of da.Array's

    Currently cannot undo this operation
    """
    _override_interface = ['Serial']

    def __init__(self, axis: Optional[int] = None):
        super().__init__()
        self.record_initialisation()
        self.axis = axis

    def join(self, sample: tuple[Any, ...]) -> da.Array:
        """Join sample"""
        return da.stack(sample, self.axis)  # type: ignore

    def unjoin(self, sample: Any) -> tuple:
        return super().unjoin(sample)


class VStack(Joiner):
    """
    Vertically Stack a tuple of da.Array's

    Currently cannot undo this operation
    """
    _override_interface = ['Serial']

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


class HStack(Joiner):
    """
    Horizontally Stack a tuple of da.Array's

    Currently cannot undo this operation
    """
    _override_interface = ['Serial']

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


class Concatenate(Joiner):
    """
    Concatenate a tuple of da.Array's

    Currently cannot undo this operation
    """
    _override_interface = ['Serial']

    def __init__(self, axis: Optional[int] = None):
        super().__init__()
        self.record_initialisation()
        self.axis = axis

    def join(self, sample: tuple[Any, ...]) -> da.Array:
        """Join sample"""
        return da.concatenate(sample, self.axis)  # type: ignore

    def unjoin(self, sample: Any) -> tuple:
        return super().unjoin(sample)
