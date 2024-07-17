# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

# type: ignore[reportPrivateImportUsage]

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Type, Union
import warnings

import xarray as xr
import numpy as np

import edit.data


from edit.pipeline.controller import PipelineIndex
from edit.pipeline.parallel import ParallelEnabledMixin
from edit.pipeline.warnings import PipelineWarning

DASK_IMPORTED = True
try:
    from dask.delayed import Delayed, delayed
    import dask.array as da
except (ImportError, ModuleNotFoundError) as _:
    DASK_IMPORTED = False

MERGE_FUNCTIONS = {
    xr.Dataset: xr.combine_by_coords,
    xr.DataArray: xr.merge,
    np.ndarray: lambda x, **k: np.merge(x, **k) if len(x) > 1 else x[0],
    list: lambda x: [*x],
    tuple: lambda x: x,
}

if DASK_IMPORTED:
    MERGE_FUNCTIONS[da.Array] = lambda x, **k: da.stack(x, **k) if len(x) > 1 else x[0]


class IdxOverride(PipelineIndex):
    """Override `idx` on any `__getitem__` call"""

    def __init__(self, index: Any):
        super().__init__()
        self._index = index
        self.record_initialisation()

    def __getitem__(self, *_, **__):
        return self.parent_pipeline()[self._index]


class IdxModifier(PipelineIndex, ParallelEnabledMixin):
    """Modify index used in `__getitem__`, allows for multiple samples.


    Examples
        >>> pipeline = Pipeline(IdxModifier((0, 1)))
        >>> pipeline[1] # Will get sample with (1, 2)
    """

    _override_interface = ["Serial"]

    def __init__(
        self,
        modification: Union[Any, tuple[Union[Any, tuple[Any, ...]], ...]],
        *extra_mods: Any,
        merge: Union[bool, int] = False,
        concat: bool = False,
        merge_function: Optional[Callable[[Any, ...], Any]] = None,
        merge_kwargs: Optional[dict[str, Any]] = None,
    ):
        """
        Index modification

        Args:
            modification (Union[Any, tuple[Union[Any, tuple[Any, ...]], ...]]):
                Can be Any type, if tuple will map across elements.
            merge (Union[bool, int], optional):
                Merge retrieved tuple, must all be the same type.
                If `int` corresponds to how many layers to merge from the bottom up.
                If `True`, merge one layer.
                Defaults to False.
            concat (bool, optional):
                Whether to concat arrays instead of stack. Defaults to False.
            merge_function (Optional[Callable], optional):
                Override for function to use when merging.
                Defaults to None.
            merge_kwargs (Optional[dict[str, Any]], optional):
                Optional extra kwargs for the merge function if `merge`. Defaults to None.

        Examples:
        >>> IdxModifier((0, 1))
            # Will get samples with (idx+0, idx+1)
        >>> IdxModifier((0, (1, 2)))
            # Will get samples with (idx+0, (idx+1, idx+2))
        >>> IdxModifier((0, (1, 2)), merge = 1)
            # Will get samples with (idx+0, merged(idx+1, idx+2))
        >>> IdxModifier((0, (1, 2)), merge = True)
            # Will get samples with (idx+0, merged(idx+1, idx+2))
        >>> IdxModifier((0, (1, 2)), merge = 2)
            # Will get samples with merged(idx+0, merged(idx+1, idx+2))

        """
        super().__init__()
        self.record_initialisation()

        if extra_mods:
            modification = (
                *(modification if isinstance(modification, tuple) else (modification,)),
                *extra_mods,
            )
        self._modification = modification

        if concat:
            MERGE_FUNCTIONS[np.ndarray] = lambda x, **k: np.concatenate(x, **k) if len(x) > 1 else x[0]
            if DASK_IMPORTED:
                MERGE_FUNCTIONS[da.Array] = lambda x, **k: da.concatenate(x, **k) if len(x) > 1 else x[0]

        merge = int(merge) if isinstance(merge, bool) else int(merge)

        def find_depth(mod, depth=0):
            if isinstance(mod, tuple):
                return max(map(lambda x: find_depth(x, depth=depth + 1), mod))
            return depth

        self._merge = find_depth(modification) - merge

        self._merge_kwargs = merge_kwargs or {}
        self._merge_function = merge_function

    def _run_merge(self, sample: tuple[Any | tuple[Any, ...], ...]) -> Any:
        """
        Run merge on samples

        Args:
            sample (tuple[Any  |  tuple[Any, ...], ...]):
                Samples to merge

        Raises:
            TypeError:
                If types differ between elements

        Returns:
            (Any):
                Merged Samples
        """

        def find_types(sam: tuple[Any, ...]) -> tuple[Type, ...]:
            return tuple(set(map(type, sam)))

        def trim(s):
            if isinstance(s, tuple) and len(s) == 1:
                return s[0]
            return s

        types = find_types(sample)

        if not all([types[0] == t for t in types[1:]]):
            raise TypeError(f"Cannot merge objects of differing types, {types}.")

        if self._merge_function is not None:
            return self._merge_function(sample, **self._merge_kwargs)

        if DASK_IMPORTED and types[0] == Delayed:
            return delayed(self._run_merge)(sample)

        if types[0] not in MERGE_FUNCTIONS:
            warnings.warn(f"Cannot merge samples of type {types[0]}.", PipelineWarning)
            return trim(sample)

        return MERGE_FUNCTIONS[types[0]](sample, **self._merge_kwargs)

    def _get_tuple(self, idx, mod: tuple[Any, ...], layer: int) -> Union[tuple[Any], Any]:
        """
        Collect all elements from tuple of modification

        Will descend through nested tuples.
        """
        super_get = self.parent_pipeline().__getitem__

        samples = []
        for m in mod:
            if isinstance(m, tuple):
                samples.append(self.parallel_interface.submit(self._get_tuple, idx, m, layer + 1))
            else:
                samples.append(self.parallel_interface.submit(super_get, idx + m))

        samples = tuple(self.parallel_interface.collect(samples))

        # def trim(s):
        #     if isinstance(s, tuple) and len(s) == 1:
        #         return s[0]
        #     return s

        if layer >= self._merge:
            return self._run_merge(samples)
        return samples

    def __getitem__(self, idx: Any):

        if not isinstance(self._modification, tuple):
            return self.parent_pipeline()[idx + self._modification]

        return self._get_tuple(idx, self._modification, 0)


class TimeIdxModifier(IdxModifier):
    """`IdxModifier` which converts all `modification`'s to `edit.data.TimeDelta`"""

    def __init__(
        self,
        modification: Union[Any, tuple[Union[Any, tuple[Any, ...]], ...]],
        *extra_mods: Union[Any, tuple[Any, ...]],
        **kwargs,
    ):
        """
        Modify `idx` but convert all `modification`'s to `edit.data.TimeDelta`

        Args:
            modification (Union[Any, tuple[Union[Any, tuple[Any, ...]], ...]]):
                Expected to be `TimeDelta` compatible, or tuples of `TimeDelta`'s.
            merge (Union[bool, int], optional):
                Merge retrieved tuple, must all be the same type.
                If `int` corresponds to how many layers to merge from the bottom up.
                If `True`, merge one layer.
                Defaults to False.
            merge_function (Optional[Callable], optional):
                Override for function to use when merging.
                Defaults to None.
            merge_kwargs (Optional[dict[str, Any]], optional):
                Optional extra kwargs for the merge function if `merge`. Defaults to None.
        """

        def can_map(mod):
            return isinstance(mod, tuple)  # and len(mod) > 0 a#nd isinstance(mod[0], tuple)

        def map_to_time(mod):
            """Map elements to `TimeDelta`"""
            if can_map(mod):
                return tuple(map(map_to_time, mod))
            return edit.data.TimeDelta(mod)

        if extra_mods:
            modification = (
                *(modification if isinstance(modification, tuple) else (modification,)),
                *extra_mods,
            )

        modification = map_to_time(modification)
        super().__init__(modification, **kwargs)
        self.record_initialisation()


@dataclass
class SequenceSpecification:
    offset: int
    num_samples: int = 1
    interval: int = 1

    def convert(self, pos):
        pos += self.offset
        total = self.num_samples * self.interval
        return tuple(range(pos, pos + total, self.interval)), pos + total - (1 * self.interval)


class SequenceRetrieval(IdxModifier):
    """
    Subclassing from `IdxModifier`, retrieve a sequence of samples based on rules.

    ## Notes
    Will attempt to stack samples, and may create a new 0 axis.

    ## Int

    If `samples` is an `int`:
        Retrieve the idx originally asked for, and the sample offset by `samples`.
        This will return 'sorted'.
        >>> SequenceRetrieval(1)[0]
        # Will get (0, 1)
        >>> SequenceRetrieval(-1)[0]
        # Will get (-1, 0)
        >>> SequenceRetrieval(-6)[0]
        # Will get (-6, 0)

    ## Single element
    If `samples` is a single element it must be of length 2 or 3, with the third being optional:
        Corresponds to a (offset, num_of_samples, Optional[interval, defaults to 1])

        The `idx` being requested is first offset, then num_of_samples retrieved, merged where applicable.

        If a single sample is retrieved, it will not be in a tuple if cannot be merged.

        >>> SequenceRetrieval((0, 3))[0]
        # Will get (0, 1, 2)
        >>> SequenceRetrieval((-1, 2))[0]
        # Will get (-1, 0)
        >>> SequenceRetrieval((2, 3))[0]
        # Will get (2,3,4)
        >>> SequenceRetrieval((2, 3, 2))[0]
        # Will get (2,4,6)
        >>> SequenceRetrieval((2, 1))[0]
        # Will get 2

    ## Multiple elements
    If `samples` is of multiples element it can consist of either tuples or ints.
        A tuple in this sequence corresponds to the same as `single element`,
        and an int the next offset to retrieve a sample at.

        These index adjustments are accumulated, so if a retrieval moves the marker
        2 forwards, the next sampling config will operate from there.

        Each config in the `samples` will be returned within its own tuple, merged where applicable.

        >>> SequenceRetrieval((0, 3),(1, 2))[0]
        # Will get ((0, 1, 2), (3, 4))
        >>> SequenceRetrieval((0, 3),1)[0]
        # Will get ((0, 1, 2), 3)
        >>> SequenceRetrieval((0, 3),2)[0]
        # Will get ((0, 1, 2), 4)
        >>> SequenceRetrieval((0, 3),(-1, 2))[0]
        # Will get ((0, 1, 2), (1, 2))
        >>> SequenceRetrieval((0, 3),(-1, 1))[0]
        # Will get ((0, 1, 2), 1)
    """

    _merge_level = 1

    def __init__(
        self,
        samples: Union[int, tuple[Union[tuple[int, ...], int], ...]],
        *,
        merge_function: Optional[Callable] = None,
        concat: bool = False,
        merge_kwargs: Optional[dict[str, Any]] = None,
    ):
        """
        Sequence retrieval

        Args:
            samples (Union[int, tuple[Union[tuple[int, int], tuple[int, int, int], int], ...]]):
                Configuration of samples to retrieve.
            merge_function (Optional[Callable], optional):
                Override for function to use when merging.
                Defaults to None.
            merge_kwargs (Optional[dict[str, Any]], optional):
                Optional extra kwargs for the merge function. Defaults to None.
        """

        super().__init__(
            self._convert(self._parse_samples(samples)),
            merge=self._merge_level,
            concat=concat,
            merge_function=merge_function,
            merge_kwargs=merge_kwargs,
        )
        self.record_initialisation()

    def _convert(self, samples: tuple[SequenceSpecification, ...]) -> tuple[tuple[int, ...], ...]:
        """
        Convert `samples` from `_parse_samples` ready for `IdxModifier`.
        """
        pos = 0
        new_indexes = []

        for spec in samples:
            indexes, pos = spec.convert(pos)
            new_indexes.append(indexes)

        if len(new_indexes) == 1:
            return new_indexes[0]
        return tuple(new_indexes)

    def _parse_samples(
        self, samples: Union[int, tuple[Union[tuple[int, ...], int], ...]]
    ) -> tuple[SequenceSpecification, ...]:
        """
        Parse input samples into known format.
        """

        def parse_int(specification):
            if specification == 0:
                return (SequenceSpecification(specification, 1),)
            elif specification < 0:
                return (
                    SequenceSpecification(specification, 1),
                    SequenceSpecification(abs(specification), 1),
                )
            else:
                return (
                    SequenceSpecification(0, 1),
                    SequenceSpecification(specification, 1),
                )

        if isinstance(samples, int):
            self._merge_level += 1
            return parse_int(samples)

        elif isinstance(samples, Iterable):
            if len(samples) in [2, 3] and all(map(lambda x: not isinstance(x, Iterable), samples)):
                return (SequenceSpecification(*samples),)  # type: ignore

            specs = []
            for sam in samples:
                if isinstance(sam, int):
                    specs.append(SequenceSpecification(sam, 1))
                elif isinstance(sam, tuple):
                    specs.append(SequenceSpecification(*sam))
            return tuple(specs)
        raise ValueError(f"Unable to parse sample specification of {samples!r}")

    def __getitem__(self, idx: Any):
        return super().__getitem__(idx)


class TemporalRetrieval(SequenceRetrieval):
    """
    Retrieve a sequence of samples from `SequenceRetrieval`,
    but force all indexes to be an `edit.data.EDITDatetime`.

    Examples:
        >>> TemporalRetrieval(-6)['2000-01-01T12']
        ## Will get samples for ('2000-01-01T06' & '2000-01-01T12')
    """

    def __init__(
        self,
        samples: Union[int, tuple[Union[tuple[int, ...], int], ...]],
        *,
        merge_function: Optional[Callable] = None,
        concat: bool = False,
        merge_kwargs: Optional[dict[str, Any]] = None,
        delta_unit: Optional[str] = None,
    ):
        super().__init__(samples, merge_function=merge_function, concat=concat, merge_kwargs=merge_kwargs)

        def map_to_tuple(mod):
            if isinstance(mod, tuple):
                return tuple(map(map_to_tuple, mod))
            return edit.data.TimeDelta((mod, delta_unit))

        if delta_unit is not None:
            self._modification = map_to_tuple(self._modification)

    def __getitem__(self, idx: Any):
        if not isinstance(idx, edit.data.EDITDatetime):
            if not edit.data.EDITDatetime.is_time(idx):
                raise TypeError(f"Cannot convert {idx!r} to `edit.data.EDITDatetime`.")
            idx = edit.data.EDITDatetime(idx)

        return super().__getitem__(idx)
