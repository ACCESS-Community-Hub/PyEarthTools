
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

        if len(self.index) > 1:
            raise RuntimeError(f"Only one index can be provided. Not {len(self.index)}")
        
    def generate(self, querytime, **kwargs):
        return self.index[0](querytime, **kwargs)

    def __repr__(self):
        string = "Data Pipeline with the following:"
        operations = self._formatted_name()
        operations = operations.split("\n")
        operations.reverse()
        operations = "\n".join(["\t* " + oper for oper in operations])
        return f"{string}\n{operations}"
    
    def _formatted_name(self):
        padding = lambda name, length_: name + "".join([" "] * (length_ - len(name)))
        desc = f"Caching Index for {self.index[0].__class__.__name__!r}. Saving at {self.cache}"
        desc = desc.replace("\n", "").replace("\t", "").strip()
        formatted = f"{padding(self.__class__.__name__, 30)}{desc}"

        if hasattr(self.index[0], '_formatted_name'):
            formatted += f"\n{self.index[0]._formatted_name()}"
        return formatted