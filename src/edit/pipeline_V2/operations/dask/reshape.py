# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

#type: ignore[reportPrivateImportUsage]

from typing import Union, Optional, Any

import math
import einops

import dask.array as da

from edit.pipeline_V2.operation import Operation


class Rearrange(Operation):
    """
    Operation to rearrange data using einops

    """
    _override_interface = ['Serial']

    def __init__(
        self,
        rearrange: str,
        skip: bool = False,
        reverse_rearrange: Optional[str] = None,
        rearrange_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Using Einops rearrange, rearrange data.

        !!! Warning
            This will occur on each iteration, and on `__getitem__`,
            so it is best to leave patches code out if using [PatchingDataIndex][edit.pipeline.operations.PatchingDataIndex].

            ```
            'p t c h w' == 't c h w'
            ```

            As this will attempt to add the patch dim if the first attempt fails

        Args:

            rearrange (str):
                String entry to einops.rearrange
            skip (bool, optional):
                Whether to skip data that cannot be rearranged. Defaults to False.
            reverse_rearrange (Optional[str], optional):
                Override for reverse operation, if not given flip rearrange. Defaults to None.
            rearrange_kwargs ( Optional[dict[str, Any]], optional):
                Extra keyword arguments to be passed to the einops.rearrange call. Defaults to {}.
        """
        super().__init__(
            split_tuples=True,
            recursively_split_tuples=True,
            recognised_types=(da.Array),
        )
        self.record_initialisation()

        self.pattern = rearrange
        self.reverse_pattern = reverse_rearrange
        self.rearrange_kwargs = rearrange_kwargs or {}

        self.skip = skip

    def __rearrange(self, data: da.Array, pattern: str, catch=True):
        try:
            return einops.rearrange(data, pattern, **self.rearrange_kwargs)
        except einops.EinopsError as excep:
            if not catch:
                if self.skip:
                    return data
                raise excep
            pattern = "->".join(["p " + side for side in pattern.split("->")])
            return self.__rearrange(data, pattern, catch=False)

    def apply_func(self, data: da.Array):
        return self.__rearrange(data, self.pattern)

    def undo_func(self, data: da.Array):
        if self.reverse_pattern:
            pattern = self.reverse_pattern
        else:
            pattern = self.pattern.split("->")
            pattern.reverse()
            pattern = "->".join(pattern)
        return self.__rearrange(data, pattern)


class Squish(Operation):
    """
    Operation to Squish one Dimensional axis at 'axis' location

    """
    _override_interface = ['Serial']

    def __init__(self, axis: Union[tuple[int, ...], int]) -> None:
        """Squish Dimension of Data

        Args:
            axis (Union[tuple[int, ...], int]):
                Axis to squish at
        """
        super().__init__(
            split_tuples=True,
            recursively_split_tuples=True,
            recognised_types=(da.Array),
        )
        self.record_initialisation()

        self.axis = axis

    def apply_func(self, sample: da.Array) -> da.Array:
        try:
            sample = da.squeeze(sample, self.axis)
        except ValueError as e:
            e.args = (*e.args, f"Shape {sample.shape}")
            raise e
        return sample

    def undo_func(self, sample: da.Array) -> da.Array:
        return da.expand_dims(sample, self.axis)


class Expand(Operation):
    """
    Operation to Expand One Dimensional axis at 'axis' location

    """
    _override_interface = ['Serial']

    def __init__(self, axis: Union[tuple[int, ...], int]) -> None:
        """Expand Dimension of Data

        Args:
            axis (Union[tuple[int, ...], int]):
                Axis to expand at
        """
        super().__init__(
            split_tuples=True,
            recursively_split_tuples=True,
            recognised_types=(da.Array),
        )
        self.record_initialisation()

        self.axis = axis

    def apply_func(self, sample: da.Array) -> da.Array:
        try:
            sample = da.squeeze(sample, self.axis)
        except ValueError as e:
            e.args = (*e.args, f"Shape {sample.shape}")
            raise e
        return sample

    def _apply_expand(self, sample: da.Array) -> da.Array:
        return da.expand_dims(sample, self.axis)


class Flattener:
    _unflattenshape = None
    _fillshape = None

    def __init__(
        self,
        flatten_dims: Optional[int] = None,
        shape_attempt: Optional[tuple[Union[str, int], ...]] = None,
    ) -> None:
        self.shape_attempt = shape_attempt

        if isinstance(flatten_dims, int) and flatten_dims < 1:
            raise ValueError(f"'flatten_dims' cannot be smaller than 1.")
        self.flatten_dims = flatten_dims

    def _prod_shape(self, shape):
        if isinstance(shape, da.Array):
            shape = shape.shape
        return math.prod(shape)

    def _configure_shape_attempt(self) -> tuple[Union[str, int], ...]:
        if not self._fillshape or not self.shape_attempt:
            raise RuntimeError(f"Cannot find shape to unflatten with, try flattening first.")
        if not "..." in self.shape_attempt:
            return self.shape_attempt

        shape_attempt = list(self.shape_attempt)
        if not len(shape_attempt) == len(self._fillshape):
            raise IndexError(f"Shapes must be the same length, not {shape_attempt} and {self._unflattenshape}")

        while "..." in shape_attempt:
            shape_attempt[shape_attempt.index("...")] = self._fillshape[shape_attempt.index("...")]

        return tuple(shape_attempt)

    def apply(self, data: da.Array) -> da.Array:
        # if self._unflattenshape is None:
        self._unflattenshape = data.shape
        self._fillshape = self._fillshape or data.shape

        self.flatten_dims = self.flatten_dims or len(data.shape)

        self._unflattenshape = self._unflattenshape[-1 * self.flatten_dims :]
        return data.reshape(
            (
                *data.shape[: -1 * self.flatten_dims],
                self._prod_shape(self._unflattenshape),
            )
        )

    def undo(self, data: da.Array) -> da.Array:
        if self._unflattenshape is None:
            raise RuntimeError(f"Shape not set, therefore cannot undo")

        def _unflatten(data, shape):
            while len(data.shape) > len(shape):
                shape = (data[-len(shape)], *shape)
            return data.reshape(shape)

        if self.flatten_dims is None:
            raise RuntimeError(f"`flatten_dims` was not set, and this set hasn't been used. Cannot Unflatten.")

        data_shape = data.shape
        parsed_shape = data_shape[: -1 * min(1, (self.flatten_dims - 1))] if len(data_shape) > 1 else data_shape
        attempts = [
            (*parsed_shape, *self._unflattenshape),
        ]

        if self.shape_attempt:
            shape_attempt = self._configure_shape_attempt()
            if shape_attempt:
                attempts.append((*parsed_shape, *shape_attempt[-1 * self.flatten_dims :]))  # type: ignore

        for attemp in attempts:
            try:
                return _unflatten(data, attemp)
            except ValueError:
                continue
        raise ValueError(f"Unable to unflatten array of shape: {data.shape} with any of {attempts}")


class Flatten(Operation):
    """
    Operation to Flatten parts of data samples into a one dimensional array
    """
    _override_interface = ['Serial']

    def __init__(
        self,
        flatten_dims: Optional[int] = None,
        *,
        shape_attempt: Optional[tuple[int, ...]] = None,
    ) -> None:
        """Operation to flatten incoming data

        Args:
            flatten_dims (Optional[int], optional):
                Number of dimensions to flatten, counting from the end. If None, flatten all, with size being stored from first use.
                Is used for negative indexing, so for last three dims `flatten_dims` == 3, Defaults to None.
            shape_attempt (Optional[tuple[int, ...]], optional):
                Reshape value to try if discovered shape fails. Used if data coming to be undone is different.
                Can have `'...'` as wildcards to get from discovered, Defaults to None.

        Examples:
            >>> incoming_data = da.zeros((5,4,3,2))
            >>> flattener = Flatten(flatten_dims = 2)
            >>> flattener.apply_func(incoming_data).shape
            (5, 4, 6)
            >>> flattener = Flatten(flatten_dims = 3)
            >>> flattener.apply_func(incoming_data).shape
            (5, 24)
            >>> flattener = Flatten(flatten_dims = None)
            >>> flattener.apply_func(incoming_data).shape
            (120)

        ??? tip "shape_attempt Advanced Use"
            If using a model which does not return a full sample, say an XGBoost model only returning the centre value, set `shape_attempt`.

            If incoming data is of shape `(1, 1, 3, 3)`, and data for undoing is `(1, 1, 1, 1)` aka `(1)`, set `shape_attempt` to `('...','...', 1, 1)`


            ```python title="Spatial Size Change"
            incoming_data = da.zeros((1,1,3,3))
            flattener = Flatten(shape_attempt = (1,1,1,1))
            flattener.apply_func(incoming_data).shape   #(9,)

            undo_data = da.zeros((1))
            flattener.undo_func(undo_data).shape        #(1,1,1,1)
            ```


            If incoming data is of shape `(8, 1, 3, 3)`, and data for undoing is `(2, 1, 1, 1)` aka `(2)`, set `shape_attempt` to `(2,'...',1,1)`

            ```python title=" Channel or Time Size Change also"
            incoming_data = da.zeros((8,1,3,3))
            flattener = Flatten(shape_attempt = (2,1,1,1))
            flattener.apply_func(incoming_data).shape   #(72,)

            undo_data = da.zeros((2))
            flattener.undo_func(undo_data).shape        #(2,1,1,1)
            ```
        """
        super().__init__(
            split_tuples=False,
            recognised_types=(da.Array),
        )
        self.record_initialisation()

        self.shape_attempt = shape_attempt
        self.flatten_dims = flatten_dims
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
                self._flatteners.append(Flattener(shape_attempt=self.shape_attempt, flatten_dims=self.flatten_dims))
                return_values.append(self._flatteners[-1])

        return tuple(return_values)

    def apply_func(self, sample: Union[tuple[da.Array, ...], da.Array]):
        if isinstance(sample, tuple):
            flatteners = self._get_flatteners(len(sample))
            return tuple(flatteners[i].apply(data_item) for i, data_item in enumerate(sample))
        return self._get_flatteners(1)[0].apply(sample)

    def undo_func(self, sample: Union[tuple[da.Array, ...], da.Array]) -> Union[tuple[da.Array, ...], da.Array]:
        if isinstance(sample, tuple):
            flatteners = self._get_flatteners(len(sample))
            return tuple(flatteners[i].undo(item) for i, item in enumerate(sample))
        else:
            return self._get_flatteners(1)[0].undo(sample)
