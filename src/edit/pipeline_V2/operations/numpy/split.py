# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from typing import Optional

import numpy as np


from edit.pipeline_V2.branching.split import Spliter


class OnAxis(Spliter):
    """
    Split across an axis in a numpy array
    """

    def __init__(self, axis: int, axis_size: Optional[int] = None):
        """Split over a numpy array axis

        Args:
            axis (int):
                Axis number to iterate over
            axis_size (int | None, optional):
                Expected size of the axis, can be found automatically. Defaults to None.
        """
        super().__init__(
            recognised_types=np.ndarray,
            recursively_split_tuples=True,
        )
        self.record_initialisation()

        self.axis = axis
        self.axis_size = axis_size

    def split(self, sample: np.ndarray) -> tuple[np.ndarray]:
        """Combine all elements of axis on batch dimension"""
        self.axis_size = self.axis_size or sample.shape[self.axis]
        sample = np.moveaxis(sample, self.axis, 0)
        return tuple(d for d in sample)

    def join(self, sample: tuple[np.ndarray]) -> np.ndarray:
        """Join `sample` together, recovering initial shape"""
        if self.axis_size is None:
            raise RuntimeError(f"`axis_size` not set.")

        data = np.concatenate(sample, axis=0)
        shape = data.shape
        data = data.reshape((self.axis_size, shape[0] // self.axis_size, *shape[1:]))
        data = np.moveaxis(data, 0, self.axis)
        return data


class OnSlice(Spliter):
    """
    Split across slices on axis

    Examples:

    """

    def __init__(self, *slices: tuple[int, ...], axis: int):
        """
        Setup slicing operation

        Args:
            slices (tuple[int, ...]):
                Each tuple is converted into a slice. So must follow slice notation
            axis (int):
                Axis number to slice over
        """
        super().__init__(
            recognised_types=np.ndarray,
        )

        self.record_initialisation()
        self._slices = tuple(slice(*x) for x in slices)
        self.axis = axis

    def split(self, sample: np.ndarray) -> tuple[np.ndarray]:
        samples = []
        sample = np.moveaxis(sample, self.axis, 0)

        for sli in self._slices:
            sli_samp = sample[sli]
            samples.append(np.moveaxis(sli_samp, 0, self.axis))

        return tuple(samples)

    def join(self, sample: tuple[np.ndarray]) -> np.ndarray:
        """Join `sample` together"""

        data = np.stack(sample, axis=0)
        data = np.moveaxis(data, 0, self.axis)
        return data


class Vsplit(Spliter):
    """
    vsplit on numpy arrays

    """

    def __init__(
        self,
    ):
        """
        Setup slicing operation
        """

        super().__init__(
            recognised_types=np.ndarray,
        )

        self.record_initialisation()

    def split(self, sample: np.ndarray) -> tuple[np.ndarray]:
        return np.vsplit(sample)  # type: ignore

    def join(self, sample: tuple[np.ndarray]) -> np.ndarray:
        """Join `sample` together"""
        return np.vstack(sample)


class Hsplit(Spliter):
    """
    hsplit on numpy arrays

    """

    def __init__(
        self,
    ):
        """
        Setup slicing operation
        """

        super().__init__(
            recognised_types=np.ndarray,
        )

        self.record_initialisation()

    def split(self, sample: np.ndarray) -> tuple[np.ndarray]:
        return np.hsplit(sample)  # type: ignore

    def join(self, sample: tuple[np.ndarray]) -> np.ndarray:
        """Join `sample` together"""
        return np.hstack(sample)
