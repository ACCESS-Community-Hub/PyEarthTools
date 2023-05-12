from __future__ import annotations
from datetime import datetime
import logging

import numpy as np


from edit.data import EDITDatetime

from edit.training.data.templates import DataStep, DataIterator
from edit.training.data.sequential import Sequential, SequentialIterator


@SequentialIterator
class RandomIterator(DataIterator):
    """
    DataIterator to provide Date Based iteration and randomly sample from said dates


    !!! Example
        ```python
        RandomIterator(PipelineStep)

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialRandomIterator = RandomIterator()
        partialRandomIterator(PipelineStep)
        ```
    """    
    def __init__(
        self, index: DataStep, catch: tuple[Exception] | Exception | str = None, seed: int = 42
    ) -> None:
        """DataIterator to provide Date Based iteration and Random Sampling      
        
        Args:
            index (DataStep): 
                Prior Pipeline step
            catch (tuple[Exception] | Exception | str, optional): 
                Name/s or Exceptions to catch and ignore. Defaults to None.
            seed (int, optional):
                Random data selection seed. Defaults to 42.
        """        
        super().__init__(index, catch)
        self.seed = seed
        self._all_timesteps = []
        self._info_ = dict(seed = seed)

    def set_iterable(self, start: str | datetime | EDITDatetime, end: str | datetime | EDITDatetime, interval: int | tuple):
        super().set_iterable(start, end, interval)
        self._all_timesteps = [self._start + self._interval * i for i in range(int((self._end-self._start)/self._interval))]


    def __iter__(self):
        if not self._iterator_ready:
            raise RuntimeError(
                f"Iterator not set for {self.__class__.__name__}. Run .set_iterable()"
            )

        timesteps = list(self._all_timesteps)
        rng = np.random.default_rng(self.seed)

        while len(timesteps) > 0:
            time = timesteps.pop(rng.integers(0,len(timesteps)))
            try:
                yield self[time]
            except self._error_to_catch as e:
                logging.info(e)

    @property
    def ignore_sanity(self):
        return True
