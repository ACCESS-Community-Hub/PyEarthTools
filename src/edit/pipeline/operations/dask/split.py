# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

# type: ignore[reportPrivateImportUsage]

from typing import Optional

import dask.array as da

from edit.pipeline.branching.split import Spliter


class OnAxis(Spliter):
    """
    Split across an axis in a dask array
    """

    _override_interface = ["Serial"]

    def __init__(self, axis: int, axis_size: Optional[int] = None):
        """Split over a dask array axis

        Args:
            axis (int):
                Axis number to iterate over
            axis_size (int | None, optional):
                Expected size of the axis, can be found automatically. Defaults to None.
        """
        super().__init__(
            recognised_types=da.Array,
            recursively_split_tuples=True,
        )
        self.record_initialisation()

        self.axis = axis
        self.axis_size = axis_size

    def split(self, sample: da.Array) -> tuple[da.Array]:
        """Combine all elements of axis on batch dimension"""
        self.axis_size = self.axis_size or sample.shape[self.axis]
        sample = da.moveaxis(sample, self.axis, 0)
        return tuple(d for d in sample)

    def join(self, sample: tuple[da.Array]) -> da.Array:
        """Join `sample` together, recovering initial shape"""
        if self.axis_size is None:
            raise RuntimeError(f"`axis_size` not set.")

        data = da.concatenate(sample, axis=0)
        shape = data.shape
        data = data.reshape((self.axis_size, shape[0] // self.axis_size, *shape[1:]))
        data = da.moveaxis(data, 0, self.axis)
        return data


class OnSlice(Spliter):
    """
    Split across slices on axis
    """

    _override_interface = ["Serial"]

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
            recognised_types=da.Array,
        )

        self.record_initialisation()
        self._slices = tuple(slice(*x) for x in slices)
        self.axis = axis

    def split(self, sample: da.Array) -> tuple[da.Array]:
        samples = []
        sample = da.moveaxis(sample, self.axis, 0)

        for sli in self._slices:
            sli_samp = sample[sli]
            samples.append(da.moveaxis(sli_samp, 0, self.axis))

        return tuple(samples)

    def join(self, sample: tuple[da.Array]) -> da.Array:
        """Join `sample` together"""

        data = da.stack(sample, axis=0)
        data = da.moveaxis(data, 0, self.axis)
        return data


class Vsplit(Spliter):
    """
    vsplit on dask arrays

    """

    _override_interface = ["Serial"]

    def __init__(
        self,
    ):
        """
        Setup slicing operation
        """

        super().__init__(
            recognised_types=da.Array,
        )

        self.record_initialisation()

    def split(self, sample: da.Array) -> tuple[da.Array]:
        return da.vsplit(sample)  # type: ignore

    def join(self, sample: tuple[da.Array]) -> da.Array:
        """Join `sample` together"""
        return da.vstack(sample)


class Hsplit(Spliter):
    """
    hsplit on dask arrays

    """

    _override_interface = ["Serial"]

    def __init__(
        self,
    ):
        """
        Setup slicing operation
        """

        super().__init__(
            recognised_types=da.Array,
        )

        self.record_initialisation()

    def split(self, sample: da.Array) -> tuple[da.Array]:
        return da.hsplit(sample)  # type: ignore

    def join(self, sample: tuple[da.Array]) -> da.Array:
        """Join `sample` together"""
        return da.hstack(sample)
