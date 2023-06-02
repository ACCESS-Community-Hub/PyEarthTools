from __future__ import annotations
import math

import einops
import numpy as np
import xarray as xr

from edit.training.data.templates import (
    DataStep,
    DataOperation,
)
from edit.training.data.sequential import Sequential, SequentialIterator


@SequentialIterator
class Rearrange(DataOperation):
    """
    DataOperation to rearrange data using einops
    

    !!! Example
        ```python
        Rearrange(PipelineStep, rearrange = 't c h w -> h w t c')

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialRearrange = Rearrange(rearrange = 't c h w -> h w t c')
        partialRearrange(PipelineStep)
        ```
    """
    def __init__(
        self,
        index: DataStep,
        rearrange: str,
        *rearrange_args,
        skip: bool = False,
        reverse_rearrange: str = None,
        rearrange_kwargs: dict = {},
        **kwargs,
    ) -> None:
        """Using Einops rearrange, rearrange data.
        
        !!! Warning
            This will occur on each iteration, and on `__getitem__`,
            so it is best to leave patches code out if using [PatchingDataIndex][edit.training.data.operations.PatchingDataIndex].
            
            ```
            'p t c h w' == 't c h w'
            ```

            As this will attempt to add the patch dim if the first attempt fails

        Args:
            index (DataStep): 
                Underlying DataStep to use to retrieve data
            rearrange (str): 
                String entry to einops.rearrange
            skip (bool, optional): 
                Whether to skip data that cannot be rearranged. Defaults to False.
            reverse_rearrange (str, optional):
                Override for reverse operation, if not given flip rearrange. Defaults to None.
            *rearrange_args (Any, optional):
                Extra arguments to be passed to the einops.rearrange call
            rearrange_kwargs (dict, optional):
                Extra keyword arguments to be passed to the einops.rearrange call. Defaults to {}.
        """        
        super().__init__(index, self._apply_rearrange, self._undo_rearrange, split_tuples=True, recognised_types=[np.ndarray], **kwargs)
        self.pattern = rearrange
        self.reverse_pattern = reverse_rearrange
        self.rearrange_args = rearrange_args
        self.rearrange_kwargs = rearrange_kwargs

        self.skip = skip
        self.__doc__ = f"Rearrange Data according to {rearrange}"
        self._info_ = dict(rearrange = rearrange)

    def __rearrange(
        self, data: tuple[np.ndarray] | np.ndarray, pattern: str, catch=True
    ):
        try:
            return einops.rearrange(data, pattern, *self.rearrange_args, **self.rearrange_kwargs)
        except einops.EinopsError as excep:
            if not catch:
                if self.skip:
                    return data
                raise excep
            pattern = "->".join(["p " + side for side in pattern.split("->")])
            return self.__rearrange(data, pattern, catch=False)

    def _apply_rearrange(self, data: np.ndarray):
        return self.__rearrange(data, self.pattern)

    def _undo_rearrange(self, data: np.ndarray):
        if self.reverse_pattern:
            pattern = self.reverse_pattern
        else:
            pattern = self.pattern.split("->")
            pattern.reverse()
            pattern = "->".join(pattern)
        return self.__rearrange(data, pattern)


@SequentialIterator
class Squish(DataOperation):
    """
    DataOperation to Squish One Dimensional axis at 'axis' location    

    !!! Example
        ```python
        Squish(PipelineStep, axis = 1)

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialSquish = Squish(axis = 1)
        partialSquish(PipelineStep)
        ```

    !!! Warning
        If using this with [PatchingDataIndex][edit.training.data.operations.PatchingDataIndex], as patch dim only exists on `__getitem__` calls, axis indexing may be off.
        Either use negative indexing, or two squish operators, one for `__getitem__` with `apply_iterator` = False,
        and one for `__iter__` with `apply_get` = False
    """

    def __init__(self, index: DataStep, axis: int, **kwargs) -> None:
        """Squish Dimension of Data        
        
        Args:
            index (DataStep): 
                Underlying DataStep to use to retrieve data
            axis (int): 
                Axis to squish at
        """        
        super().__init__(index, self._apply_squish, self._apply_expand, **kwargs)
        self.axis = axis
        self._info_ = dict(axis = axis)

    @property
    def __doc__(self):
        return f"""
        Squish One Dimensional axis {self.axis!r}
        """

    def _apply_squish(self, data):
        if isinstance(data, tuple):
            return tuple(map(self._apply_squish, data))
        try:
            data = np.squeeze(data, self.axis)
        except ValueError as e:
            e.args = (*e.args, f"Shape {data.shape}")
            raise e
        return data

    def _apply_expand(self, data):
        if isinstance(data, tuple):
            return tuple(map(self._apply_expand, data))
        return np.expand_dims(data, self.axis)


@SequentialIterator
class Expand(DataOperation):
    """
    DataOperation to Expand One Dimensional axis at 'axis' location    

    !!! Example
        ```python
        Expand(PipelineStep, axis = 1)

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialExpand = Expand(axis = 1)
        partialExpand(PipelineStep)
        ```

    !!! Warning
        If using this with [PatchingDataIndex][edit.training.data.operations.PatchingDataIndex], as patch dim only exists on `__getitem__` calls, axis indexing may be off.
        Either use negative indexing, or two squish operators, one for `__getitem__` with `apply_iterator` = False,
        and one for `__iter__` with `apply_get` = False
    """

    def __init__(self, index: DataStep, axis: int, **kwargs) -> None:
        """Expand Dimension of Data
        

        Args:
            index (DataStep): 
                Underlying DataStep to use to retrieve data
            axis (int): 
                Axis to expand at
        """        
        super().__init__(index, self._apply_expand, self._apply_squish, **kwargs)
        self.axis = axis
        self._info_ = dict(axis = axis)

    @property
    def __doc__(self):
        return f"""
        Expand One Dimensional axis {self.axis!r}
        """

    def _apply_squish(self, data):
        if isinstance(data, tuple):
            return tuple(map(self._apply_squish, data))
        try:
            data = np.squeeze(data, self.axis)
        except ValueError as e:
            e.args = (*e.args, f"Shape {data.shape}")
            raise e
        return data

    def _apply_expand(self, data):
        if isinstance(data, tuple):
            return tuple(map(self._apply_expand, data))
        return np.expand_dims(data, self.axis)

class Flattener:
    def __init__(self, flatten_dims, shape_attempt: tuple = None) -> None:
        self._unflattenshape = None
        self._fillshape = None
        self.shape_attempt = shape_attempt

        if isinstance(flatten_dims, int) and flatten_dims < 1:
            raise ValueError(f"'flatten_dims' cannot be smaller than 1.")
        self.flatten_dims = flatten_dims

    def _prod_shape(self, shape):
        if isinstance(shape, np.ndarray):
            shape = shape.shape
        return math.prod(shape)

    def _configure_shape_attempt(self):
        if not self._fillshape or not self.shape_attempt:
            return self.shape_attempt
        if not '...' in self.shape_attempt:
            return self.shape_attempt
        
        shape_attempt = list(self.shape_attempt)
        if not len(shape_attempt) == len(self._fillshape):
            raise IndexError(f"Shapes must be the same length, not {shape_attempt} and {self._unflattenshape}")
        
        while '...' in shape_attempt:
            shape_attempt[shape_attempt.index('...')] = self._fillshape[shape_attempt.index('...')]
        
        return tuple(shape_attempt)

    def apply(self, data : np.ndarray) -> np.ndarray:
        #if self._unflattenshape is None:
        self._unflattenshape = data.shape
        if self._fillshape is None:
            self._fillshape = data.shape

        if not self.flatten_dims:
            self.flatten_dims = len(data.shape)
    
        self._unflattenshape = self._unflattenshape[-1 * self.flatten_dims:]
        return data.reshape((*data.shape[: -1 * self.flatten_dims], self._prod_shape(self._unflattenshape)))

    def undo(self, data: np.ndarray) -> np.ndarray:
        if self._unflattenshape is None:
            raise RuntimeError(f"Shape not set, therefore cannot undo")
        
        def _unflatten(data, shape):
            while len(data.shape) > len(shape):
                shape = (data[-len(shape)], *shape)
            return data.reshape(shape)
        
        data_shape = data.shape
        parsed_shape = data_shape[: -1 * min(1,(self.flatten_dims-1))] if len(data_shape) > 1 else data_shape
        attempts = [(*parsed_shape, *self._unflattenshape), ]

        if self.shape_attempt:
            shape_attempt = self._configure_shape_attempt()
            attempts.append((*parsed_shape, *shape_attempt[-1 * self.flatten_dims:]))

        for attemp in attempts:
            try:
                return _unflatten(data, attemp)
            except ValueError:
                continue
        raise ValueError(f"Unable to unflatten array of shape: {data.shape} with any of {attempts}")

@SequentialIterator
class Flatten(DataOperation):
    """
    DataOperation to Flatten Data Samples into a one dimensional array    

    !!! Example
        ```python
        Flatten(PipelineStep)

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialFlatten = Flatten()
        partialFlatten(PipelineStep)
        ```

    !!! Warning
        If use this with [PatchingDataIndex][edit.training.data.operations.PatchingDataIndex], set `seperate_patch` to True
    """

    def __init__(self, index: DataStep, flatten_dims: int = None, *, shape_attempt: tuple = None,) -> None:
        """DataOperation to flatten incoming data

        Args:
            index (DataStep): 
                Underlying index to retrieve data from
            flatten_dims (int, optional): 
                Number of dimensions to flatten, counting from the end. If None, flatten all, with size being stored from first use. 
                Is used for negative indexing, so for last three dims `flatten_dims` == 3, Defaults to None.
            shape_attempt (tuple, optional):
                Reshape value to try if discovered shape fails. Used if data coming to be undone is different. 
                Can have `'...'` as wildcards to get from discovered, Defaults to None.

        Examples:
            >>> incoming_data = np.zeros((5,4,3,2))
            >>> flattener = Flatten([], flatten_dims = 2)
            >>> flattener.apply_func(incoming_data).shape   
            (5, 4, 6)
            >>> flattener = Flatten([], flatten_dims = 3)
            >>> flattener.apply_func(incoming_data).shape 
            (5, 24)
            >>> flattener = Flatten([], flatten_dims = None)
            >>> flattener.apply_func(incoming_data).shape 
            (120)
                    
        ??? tip "shape_attempt Advanced Use"
            If using a model which does not return a full sample, say an XGBoost model only returning the centre value, set `shape_attempt`.

            If incoming data is of shape `(1, 1, 3, 3)`, and data for undoing is `(1, 1, 1, 1)` aka `(1)`, set `shape_attempt` to `('...','...', 1, 1)`

            
            ```python title="Spatial Size Change"
            incoming_data = np.zeros((1,1,3,3))
            flattener = Flatten([], shape_attempt = (1,1,1,1))
            flattener.apply_func(incoming_data).shape   #(9,)

            undo_data = np.zeros((1))
            flattener.undo_func(undo_data).shape        #(1,1,1,1)
            ```  

           
            If incoming data is of shape `(8, 1, 3, 3)`, and data for undoing is `(2, 1, 1, 1)` aka `(2)`, set `shape_attempt` to `(2,'...',1,1)`

            ```python title=" Channel or Time Size Change also"
            incoming_data = np.zeros((8,1,3,3))
            flattener = Flatten([], shape_attempt = (2,1,1,1))
            flattener.apply_func(incoming_data).shape   #(72,)

            undo_data = np.zeros((2))
            flattener.undo_func(undo_data).shape        #(2,1,1,1)
            ```  
        """        
        super().__init__(index, apply_func=self._apply_flattening, undo_func=self._undo_flattening)

        self.shape_attempt = shape_attempt
        self.flatten_dims = flatten_dims
        self._flatteners = []


        self._info_ = dict(flatten_dims = flatten_dims, shape_attempt = shape_attempt)

    def _get_flatteners(self, number: int) -> tuple[Flattener]:
        """
        Retrieve a set number of Flattener, creating new ones if needed
        """
        return_values = []
        for i in range(number):
            if i < len(self._flatteners):
                return_values.append(self._flatteners[i])
            else:
                self._flatteners.append(Flattener(shape_attempt= self.shape_attempt, flatten_dims = self.flatten_dims))
                return_values.append(self._flatteners[-1])

        return return_values

    def _apply_flattening(self, data : tuple[np.ndarray] | np.ndarray):
        if isinstance(data, tuple):
            flatteners = self._get_flatteners(len(data))
            return tuple(flatteners[i].apply(data_item) for i,data_item in enumerate(data))
        return self._get_flatteners(1)[0].apply(data)


    def _undo_flattening(self, data):
        if isinstance(data, tuple):
            flatteners = self._get_flatteners(len(data))
            return tuple(np.stack(flatteners[i].undo(item)) for i, item in enumerate(data))
        else:
            return self._get_flatteners(1)[0].undo(data)


@SequentialIterator
class Dimension(DataOperation):
    def __init__(self, index: DataStep, dimensions: str | list[str], append: bool = True):
        super().__init__(index, apply_func=self._order_dims, recognised_types=[xr.Dataset, xr.DataArray], split_tuples=True)
        
        self.dimensions = dimensions if isinstance(dimensions, (list, tuple)) else [dimensions]
        self.append = append
        
        self.__doc__ = "Reordering Dimensions"
        self._info_  = dict(dimensions = dimensions, append = append)

    def _order_dims(self, ds : xr.Dataset, xr.DataArray):
        dims = ds.dims
        dims = set(dims).difference(set(self.dimensions))
        if self.append:
            dims = [*self.dimensions, *dims]
        else:
            dims = [*dims, self.dimensions]
        return ds.transpose(*dims)