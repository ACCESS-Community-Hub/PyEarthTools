from __future__ import annotations

from typing import Any, Union
import xarray as xr
import functools

from pathlib import Path

from edit.data import DataIndex
from edit.data.patterns import PatternIndex
from edit.data import CachingIndex as dataCachingIndex

from edit.training.data.utils import get_transforms
from edit.training.data.templates import TrainingOperatorIndex, DataStep
from edit.training.data.sequential import Sequential, SequentialIterator


@SequentialIterator
class CachingIndex(TrainingOperatorIndex, dataCachingIndex):
    """
    [edit.training][edit.training] Implementation of [CachingIndex][edit.data.cacheIndex.CachingIndex]

    !!! Example
        ```python
        CachingIndex(PipelineStep, cache = '~/CacheDirectory/', pattern = 'ExpandedDate')

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialCaching = CachingIndex(cache = '~/CacheDirectory/', pattern = 'ExpandedDate')
        partialCaching(PipelineStep)
        ```
    """

    def __init__(
        self,
        index: dict | DataIndex | DataStep,
        *,
        cache: str | Path = None,
        pattern: str | PatternIndex = None,
        pattern_kwargs: dict = {},
        **kwargs,
    ):
        """
        Initalise the CachingIndex

        !!! Warning:
            Either cache, or pattern must be defined

        Args:
            index (dict | DataIndex | DataStep):
                Prior Data Retrieval Step, can be dict which will be automatically initialised
            cache (str | Path, optional):
                Path to cache data to. Defaults to None.
            pattern (str | PatternIndex, optional):
                Pattern to use to cache data, if str use `pattern_kwargs` to initalise. Defaults to None.
            pattern_kwargs (dict, optional):
                Kwargs to initalise the pattern with. Defaults to {}.


        """
        super().__init__(
            index=index,
            cache=cache,
            pattern=pattern,
            pattern_kwargs=pattern_kwargs,
            **kwargs,
        )

    @functools.wraps(dataCachingIndex.generate)
    def generate(self, querytime, **kwargs):
        return self.index(querytime, **kwargs)

    @property
    def __doc__(self):
        return f"Caching Index for {self.index.__class__.__name__!r}. Saving at {self.cache}"

    @property
    def ignore_sanity(self):
        return True
