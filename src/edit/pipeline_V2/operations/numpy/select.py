# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from typing import Any, Optional

import numpy as np

from edit.pipeline_V2.operation import Operation


class Select(Operation):
    """
    Operation to select an element from a given array
    """

    _override_interface = ["Delayed", "Serial"]
    _interface_kwargs = {"Delayed": {"name": "Select"}}

    def __init__(
        self,
        array_index: tuple[Optional[int], ...],
        tuple_index: Optional[int] = None,
    ):
        """Select data from a given index

        Args:
            array_index (tuple[Optional[int],...]):
                Tuple of indexes from which to select data. Can use None to specify not to select
            tuple_index (Optional[int], optional):
                Choice of which tuple element to apply selection to, if tuples passed. Defaults to None.

        Examples
            >>> incoming_data = np.zeros((10,5,2))
            >>> select = Select([0])
            >>> select.apply_func(incoming_data).shape
            (5,2)
            >>> select = Select([0, None, 0])
            >>> select.apply_func(incoming_data).shape
            (5)
        """

        super().__init__(
            operation="apply",
            split_tuples=False,
            recognised_types=(np.ndarray, tuple, list),
        )

        self.record_initialisation()

        self.array_index = array_index
        self.tuple_index = tuple_index

    def _index(self, data, array_index):
        shape = data.shape
        for i, index in enumerate(reversed(array_index)):
            if index is None:
                pass
            selected_data = np.take(data, indices=index, axis=-(i + 1))
            if len(selected_data.shape) < len(shape):
                selected_data = np.expand_dims(selected_data, axis=-(i + 1))
            data = selected_data
        return data

    def apply_func(self, data):
        array_index = self.array_index

        if isinstance(data, tuple):
            data = list(data)
            if self.tuple_index is None:
                return tuple(map(lambda x: self._index(x, array_index), data))

            data[self.tuple_index] = self._index(data[self.tuple_index], array_index)
            data = tuple(data)
            return data

        return self._index(data, array_index)


class Slice(Operation):
    """
    Slice a chunk of a numpy array

    Examples:
        >>> Slice((0,10,2)) # == slice(0,10,2)
        >>> incoming_data = np.zeros((10,5,4))
        >>> Slice((0,10,2), (1, 3)).apply_func(incoming_data).shape
        (5,2,4)
        >>> Slice((1, 3)).apply_func(incoming_data).shape
        (2,5,4)
        >>> Slice((1, 3), reverse_slice = True).apply_func(incoming_data).shape
        (10,5,2)
    """

    _override_interface = ["Delayed", "Serial"]
    _interface_kwargs = {"Delayed": {"name": "Slice"}}

    def __init__(self, *slices: tuple[Optional[int], ...], reverse_slice: bool = False):
        """
        Setup slicing operation

        Args:
            slices (tuple[Optional[int], ...]):
                Each tuple is converted into a slice. So must follow slice notation
            reverse_slice (bool, optional):
                Whether to slice offset towards last axis. Defaults to False.
        """
        super().__init__(
            operation="apply",
            split_tuples=True,
            recursively_split_tuples=True,
            recognised_types=np.ndarray,
        )

        self.record_initialisation()
        self._slices = tuple(slice(*x) for x in slices)
        self.reverse_slice = reverse_slice

    def apply_func(self, sample: np.ndarray) -> np.ndarray:
        if self.reverse_slice:
            base_slices = [slice(None, None, None)] * (len(sample.shape) - len(self._slices))
            base_slices.extend(self._slices)
            return sample[tuple(base_slices)]

        return sample[self._slices]
