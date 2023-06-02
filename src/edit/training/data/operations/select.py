import numpy as np
import xarray as xr
from edit.training.data.templates import (
    DataStep,
    DataOperation,
)
from edit.training.data.sequential import Sequential, SequentialIterator


@SequentialIterator
class Select(DataOperation):
    """
    DataOperation to select an element from a given array

    !!! Example
        ```python
        Select(PipelineStep, array_index = [0])

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialSelect = Select(array_index = [0])
        partialSelect(PipelineStep)
        ```
    
    ??? Warning
        If using this with [PatchingDataIndex][edit.training.data.operations.patch] issues may arise with invalid shapes for .
    """

    def __init__(self,
        index: DataStep,
        array_index: tuple,
        tuple_index: int = None,
        **kwargs,
        ):
        """Select data from a given index

        Args:
            index (DataStep): 
                Underlying DataStep to get data
            array_index (tuple): 
                Tuple of indexes from which to select data. Can use None to specify not to select
            tuple_index (int, optional): 
                Choice of which tuple element to apply selection to, if tuples passed. Defaults to None.

        Examples
            >>> incoming_data = np.zeros((10,5,2))
            >>> select = Select(None, [0])
            >>> select.apply_func(incoming_data).shape
            (5,2)            
            >>> select = Select(None, [0, None, 0])
            >>> select.apply_func(incoming_data).shape
            (5)  
        """

        super().__init__(index, self._apply_select, undo_func=None, recognised_types=[np.ndarray, tuple, list], **kwargs)
        self.array_index = array_index
        self.tuple_index = tuple_index

        self.__doc__ = f"Index Data in position {tuple_index} according to :, {array_index}"
        self._info_ = dict(array_index = array_index, tuple_index = tuple_index)

    def _index(self, data, array_index):
        shape = data.shape
        for i, index in enumerate(reversed(array_index)):
            if index is None:
                pass
            selected_data = np.take(data, indices=index, axis=-(i+1))
            if len(selected_data.shape)< len(shape):
                selected_data = np.expand_dims(selected_data, axis=-(i+1))
            data = selected_data
        return data

    def _apply_select(self, data):

        array_index = self.array_index

        if isinstance(data, tuple):
            data = list(data)
            if self.tuple_index is None:
                return tuple(map(lambda x: self._index(x, array_index), data))
            
            data[self.tuple_index] = self._index(data[self.tuple_index], array_index)
            data = tuple(data)
            return data

        return self._index(data, array_index)


@SequentialIterator
class SelectDataset(DataOperation):
    """
    DataOperation to select a given set of variables from an [Dataset][xarray.Dataset]

    !!! Example
        ```python
        SelectDataset(PipelineStep, variables = 'var_1')

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialSelectDataset = SelectDataset(variables = 'var_1')
        partialSelectDataset(PipelineStep)
        ```
    """
    def __init__(self,
        index: DataStep,
        variables : str | list[str],
        tuple_index: int = None,
        **kwargs,
        ):
        """_summary_

        Args:
            index (DataStep): 
                Underlying DataStep to get data
            variables (str | list[str]): 
                List of variables to select
            tuple_index (int, optional): 
                Choice of which tuple element to apply selection to, if tuples passed. Defaults to None.

        """

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

        

