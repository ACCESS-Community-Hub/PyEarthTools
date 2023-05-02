
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
        
        !!! Note
            This will occur on each iteration, and on `__getitem__`,
            so it is best to leave patches code out if using [PatchingDataIndex][edit.training.data.operations.PatchingDataIndex].

            'p t c h w' == 't c h w'

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
        self.rearrange = rearrange
        self.rearrange_args = rearrange_args
        self.rearrange_kwargs = rearrange_kwargs

        self.skip = skip
        self.__doc__ = f"Rearrange Data according to {rearrange}"

    def __rearrange(
        self, data: tuple[np.ndarray] | np.ndarray, rearrange: str, catch=True
    ):
        """
        Apply einops.rearrange on data.

        If this fails, attempt to add 'p' to either side.
        """
        try:
            if isinstance(data, tuple):
                return tuple(
                    map(
                        lambda x: einops.rearrange(x, rearrange, *self.rearrange_args, **self.rearrange_kwargs),
                        data,
                    )
                )
            return einops.rearrange(data, self.rearrange)

        except einops.EinopsError as excep:
            if not catch:
                if self.skip:
                    return data
                raise excep
            rearrange = "->".join(["p " + side for side in rearrange.split("->")])
            return self.__rearrange(data, rearrange, catch=False)

    def _apply_rearrange(self, data):
        return self.__rearrange(data, self.rearrange)

    def _undo_rearrange(self, data):
        reversed_rearrange = self.rearrange.split("->")
        reversed_rearrange.reverse()
        return self.__rearrange(data, "->".join(reversed_rearrange))


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
        If use this with [PatchingDataIndex][edit.training.data.operations.PatchingDataIndex], as patch dim only exists on `__getitem__` calls, axis indexing may be off.
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
        Squish One Dimensional on axis {self.axis!r}'
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
        If use this with [PatchingDataIndex][edit.training.data.operations.PatchingDataIndex], as patch dim only exists on `__getitem__` calls, axis indexing may be off.
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
        Expand One Dimensional on axis {self.axis!r}
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
