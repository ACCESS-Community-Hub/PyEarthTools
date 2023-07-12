from __future__ import annotations

import functools
import warnings

from pathlib import Path

from edit.data import DataIndex
from edit.data.patterns import PatternIndex
from edit.data import CachingIndex as dataCachingIndex

from edit.training.data.warnings import PipelineWarning
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
        regenerate: bool = False,
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
            regenerate (bool, optional):
                Force this to regenerate the cache, overriding exisiting data. Defaults to False.


        """
        super().__init__(
            index=index,
            cache=cache,
            pattern=pattern,
            pattern_kwargs=pattern_kwargs,
            **kwargs,
        )
        if not regenerate:
            warnings.warn(f"Data will be read from cache at {cache}. Even if the config has been changed.", PipelineWarning)
        
        self.__doc__ = "Caching Index"
        self._info_ = dict(cache_location = cache, regenerate = regenerate)

        self.regenerate = regenerate

    @functools.wraps(dataCachingIndex.generate)
    def generate(self, querytime, **kwargs):
        return self.index(querytime, **kwargs)

    @functools.wraps(dataCachingIndex.filesystem)
    def filesystem(self, *args, **kwargs):
        if self.regenerate:
            kwargs['force_generate'] = True
        return super().filesystem(*args, **kwargs)
    
    @property
    def ignore_sanity(self):
        return True
