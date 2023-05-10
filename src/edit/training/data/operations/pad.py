from __future__ import annotations

import numpy as np
import warnings

from edit.training.data.templates import (
    DataStep,
    DataOperation,
)
from edit.training.data.sequential import SequentialIterator
from edit.training.data.warnings import PipelineWarning

class _UndoPadderOper:
    def __init__(self, padding_type: str = 'edge') -> None:
        self.shape = None
        self.padding_type = padding_type

    def pad(self, array, target_shape):
        if len(array.shape) == 1:
            array = array.reshape((*array.shape, 1))

        return np.pad(
            array,
            [(0, target_shape[i] - array.shape[i]) for i in range(len(array.shape))],
            self.padding_type,
        )

    def apply(self, data : np.ndarray) -> np.ndarray:
        self.shape = data.shape
        return data

    def undo(self, data: np.ndarray) -> np.ndarray:
        if self.shape is None:
            raise RuntimeError(f"Shape not set, therefore cannot undo")
        return self.pad(data, self.shape)

@SequentialIterator
class UndoPadder(DataOperation):
    """
    DataOperation to Pad Data Samples on the undo to the original shape

    !!! Example
        ```python
        Padder(PipelineStep)

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialPadder = Padder()
        partialPadder(PipelineStep)
        ```

    !!! Warning
        If use this with [PatchingDataIndex][edit.training.data.operations.PatchingDataIndex], set `seperate_patch` to True
    """

    def __init__(self, index: DataStep, padding_type: str = 'edge') -> None:
        """DataOperation to pad data as it is undone back to the incoming data shape

        Args:
            index (DataStep): 
                Underlying index to retrieve data from
            padding_type (str, optional):
                Padding type, must be one of [np.pad][numpy.pad] options. Defaults to 'edge'
        """        
        super().__init__(index, apply_func=self._record_data_shape, undo_func=self._pad_undo_func, recognised_types=(tuple, list, np.ndarray))

        self._padders = []

        self._info_ = dict(padding_type = padding_type)
        warnings.warn(f"Padding configured on the undo operation. This will alter the behaviour significantly.", PipelineWarning)

    def _get_padders(self, number: int) -> tuple[_UndoPadderOper]:
        """
        Retrieve a set number of _UndoPadderOper's, creating new ones if needed
        """
        return_values = []
        for i in range(number):
            if i < len(self._padders):
                return_values.append(self._padders[i])
            else:
                self._padders.append(_UndoPadderOper())
                return_values.append(self._padders[-1])

        return return_values

    def _record_data_shape(self, data : tuple[np.ndarray] | np.ndarray):
        if isinstance(data, tuple):
            padders = self._get_padders(len(data))
            return tuple(padders[i].apply(data_item) for i,data_item in enumerate(data))
        return self._get_padders(1)[0].apply(data)


    def _pad_undo_func(self, data: tuple[np.ndarray] | np.ndarray):
        if isinstance(data, tuple):
            padders = self._get_padders(len(data))
            return tuple(padders[i].undo(data_item) for i,data_item in enumerate(data))
        return self._get_padders(1)[0].undo(data)