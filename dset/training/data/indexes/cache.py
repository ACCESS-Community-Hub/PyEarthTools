
from typing import Any, Union
import xarray as xr
import datetime

from pathlib import Path

from dset.data.patterns import PatternIndex
from dset.data import CachingIndex as dataCachingIndex
from dset.data.default import OperatorIndex

from dset.training.data.templates import DataStep

from dset.training.data.utils import get_transforms
from dset.training.data.templates import SequentialIterator, TrainingOperatorIndex

@SequentialIterator
class CachingIndex(TrainingOperatorIndex, dataCachingIndex):
    """
    dset.training Implementation of dset.data.CachingIndex
    """
    def __init__(
        self,
        index: Union[dict, Any],
        cache: Union[str, Path] = None,
        pattern: Union[str, PatternIndex] = None,
        pattern_kwargs: dict = {},
        **kwargs,
    ):
        super().__init__(index = index, cache = cache, pattern = pattern, pattern_kwargs = pattern_kwargs, **kwargs)

        
    def generate(self, querytime, **kwargs):
        return self.index(querytime, **kwargs)

    def _formatted_name(self):
        desc = f"Caching Index for {self.index.__class__.__name__!r}. Saving at {self.cache}"
        return super()._formatted_name(desc)

    @property
    def ignore_sanity(self):
        return True
