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
    """

    def __init__(
        self,
        index: DataIndex | DataStep,
        cache: str | Path = None,
        pattern: str | PatternIndex = None,
        pattern_kwargs: dict = {},
        **kwargs,
    ):
        """
        Initalise the CachingIndex

        Args:
            index (DataIndex | DataStep): DataIndex or DataStep to use to get data
            cache (str | Path, optional): Path to cache data to. Defaults to None.
            pattern (str | PatternIndex, optional): Pattern to use to cache data, if str use `pattern_kwargs` to initalise. Defaults to None.
            pattern_kwargs (dict, optional): Kwargs to initalise the pattern with. Defaults to {}.

        Note:
            Either cache, or pattern must be defined
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

    def _formatted_name(self):
        desc = f"Caching Index for {self.index.__class__.__name__!r}. Saving at {self.cache}"
        return super()._formatted_name(desc)

    @property
    def ignore_sanity(self):
        return True
