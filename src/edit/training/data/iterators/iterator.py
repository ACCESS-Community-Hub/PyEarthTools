import functools
import time
from itertools import zip_longest
from typing import Union

import numpy as np
import xarray as xr

from edit.training.data.templates import DataStep, DataIterator
from edit.training.data.sequential import Sequential, SequentialIterator


@SequentialIterator
class Iterator(DataIterator):
    """
    Basic DataIterator to provide Date Based iteration


    !!! Example
        ```python
        Iterator(PipelineStep)

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialIterator = Iterator()
        partialIterator(PipelineStep)
        ```
    """    
    def __init__(
        self, index: DataStep, catch: tuple[Exception] | Exception | str = None
    ) -> None:
        """DataIterator to provide Date Based iteration        
        
        Args:
            index (DataStep): 
                Prior Pipeline step
            catch (tuple[Exception] | Exception | str, optional): 
                Name/s or Exceptions to catch and ignore. Defaults to None.
        """        
        super().__init__(index, catch)

    @property
    def ignore_sanity(self):
        return True
