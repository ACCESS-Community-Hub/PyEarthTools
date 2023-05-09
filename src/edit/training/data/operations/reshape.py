from __future__ import annotations

import einops
import numpy as np
from scipy import interpolate

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
        skip: bool = False,
        *rearrange_args,
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

        Args:
            index (DataStep): 
                Underlying DataStep to use to retrieve data
            rearrange (str): 
                String entry to einops.rearrange
            skip (bool, optional): 
                Whether to skip data that cannot be rearranged. Defaults to False.
            *rearrange_args (Any, optional):
                Extra arguments to be passed to the einops.rearrange call
            rearrange_kwargs (dict, optional):
                Extra keyword arguments to be passed to the einops.rearrange call. Defaults to {}.
        """        
        super().__init__(index, self._apply_rearrange, self._undo_rearrange, **kwargs)
        self.pattern = rearrange
        self.rearrange_args = rearrange_args
        self.rearrange_kwargs = rearrange_kwargs

        self.skip = skip
        self.__doc__ = f"Rearrange Data according to {rearrange}"

    def __rearrange(
        self, data: tuple[np.ndarray] | np.ndarray, pattern: str, catch=True
    ):
        try:
            if isinstance(data, tuple):
                return tuple(
                    map(
                        lambda x: einops.rearrange(x, pattern, *self.rearrange_args, **self.rearrange_kwargs),
                        data,
                    )
                )
            return einops.rearrange(data, pattern)

        except einops.EinopsError as excep:
            if not catch:
                if self.skip:
                    return data
                raise excep
            pattern = "->".join(["p " + side for side in pattern.split("->")])
            return self.__rearrange(data, pattern, catch=False)

    def _apply_rearrange(self, data):
        return self.__rearrange(data, self.pattern)

    def _undo_rearrange(self, data):
        pattern = self.pattern.split("->")
        pattern.reverse()
        return self.__rearrange(data, "->".join(pattern))


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
    def __init__(self) -> None:
        self.shape = None
    def apply(self, data : np.ndarray) -> np.ndarray:
        self.shape = data.shape
        return data.flatten()
    def undo(self, data: np.ndarray) -> np.ndarray:
        if self.shape is None:
            raise RuntimeError(f"Shape not set, therefore cannot undo")
        return data.reshape(self.shape)

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

    def __init__(self, index: DataStep, seperate_patch: bool = False) -> None:
        """DataOperation to flatten inoming data

        Args:
            index (DataStep): 
                Underlying index to retrieve data from
            seperate_patch (bool, optional): 
                Separate patches so they aren't squashed. Use only if using [PatchingDataIndex][edit.training.data.operations.PatchingDataIndex]. Defaults to False.
        """        
        super().__init__(index, apply_func=self._apply_flattening, undo_func=self._undo_flattening)

        self.seperate_patch = seperate_patch
        self._flatteners = []

    def _get_flatteners(self, number: int) -> tuple[Flattener]:
        """
        Retrieve a set number of Flattener, creating new ones if needed
        """
        return_values = []
        for i in range(number):
            if i < len(self._flatteners):
                return_values.append(self._flatteners[i])
            else:
                self._flatteners.append(Flattener())
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
            if self.seperate_patch:
                return tuple(np.stack(tuple(map(flatteners[i].undo, item))) for i, item in enumerate(data))
            return tuple(np.stack(flatteners[i].undo(item)) for i, item in enumerate(data))
        else:
            if self.seperate_patch:
                return np.stack(tuple(map(self._get_flatteners(1)[0].undo, data)))
            return self._get_flatteners(1)[0].undo(data)


    def __getitem__(self, idx):
        data = self.index[idx]
        if self.apply_get and self.apply_func:
            if isinstance(data, tuple):
                flatteners = self._get_flatteners(len(data))
                if self.seperate_patch:
                    return tuple(np.stack(tuple(map(flatteners[i].apply, item))) for i, item in enumerate(data))
                return tuple(np.stack(flatteners[i].apply(item)) for i, item in enumerate(data))
            else:
                if self.seperate_patch:
                    return np.stack(tuple(map(self._get_flatteners(1)[0].apply, data)))
                return self._get_flatteners(1)[0].apply(data)
        return data
