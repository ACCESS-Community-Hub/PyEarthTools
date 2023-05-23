import numpy as np
import xarray as xr
from edit.training.data.templates import (
    DataStep,
    DataOperation,
)
from edit.training.data.sequential import Sequential, SequentialIterator


@SequentialIterator
class Select(DataOperation):

    def __init__(self,
        index: DataStep,
        array_index: tuple,
        array_choice: int = None,
        **kwargs,
        ):

        super().__init__(index, self._apply_select, undo_func=None, recognised_types=[np.ndarray, tuple, list], **kwargs)
        self.array_index = array_index
        self.array_choice = array_choice

        self.__doc__ = f"Index Data in position {array_choice} according to {array_index}"
        self._info_ = dict(array_index = array_index, array_choice = array_choice)

    def _index(self, data, array_index):
        shape = data.shape
        for i, index in enumerate(reversed(array_index)):
            selected_data = np.take(data, indices=index, axis=-(i+1))
            if len(selected_data.shape)< len(shape):
                selected_data = np.expand_dims(selected_data, axis=-(i+1))
            data = selected_data
        return data

    def _apply_select(self, data):

        array_index = self.array_index

        if isinstance(data, tuple):
            data = list(data)
            if self.array_choice is None:
                return tuple(map(lambda x: self._index(x, array_index), data))
            
            data[self.array_choice] = self._index(data[self.array_choice], array_index)
            data = tuple(data)
            return data

        return self._index(data, array_index)


@SequentialIterator
class SelectDataset(DataOperation):

    def __init__(self,
        index: DataStep,
        variables : str | list[str],
        tuple_index: int = None,
        **kwargs,
        ):

        super().__init__(index, self._apply_select, undo_func=None, recognised_types=[xr.Dataset, tuple, list], **kwargs)
        self.variables = variables
        self.tuple_index = tuple_index

        self.__doc__ = f"Select Data Variables"
        self._info_ = dict(variables = variables, tuple_index = tuple_index)

    def _index(self, data: xr.Dataset):
        data =  data[self.variables]
        if isinstance(data, xr.DataArray):
            data = data.to_dataset()
        return data

    def _apply_select(self, data):

        if isinstance(data, tuple):
            data = list(data)
            if self.tuple_index is None:
                return tuple(map(lambda x: self._index(x), data))
            
            data[self.tuple_index] = self._index(data[self.tuple_index])
            data = tuple(data)
            return data            

        return self._index(data)

        

