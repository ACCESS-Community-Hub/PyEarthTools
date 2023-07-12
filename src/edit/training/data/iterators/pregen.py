from __future__ import annotations
from datetime import datetime
import logging

import numpy as np
import concurrent.futures


from edit.data import EDITDatetime

from edit.training.data.templates import DataStep, DataIterator
from edit.training.data.sequential import Sequential, SequentialIterator

_executor = concurrent.futures.ThreadPoolExecutor(10) 

def pregenerate(index : DataStep, samples: list[str | EDITDatetime]):
    futures = []
    for sample in samples:
        print(sample)
        futures.append(_executor.submit(index, sample))
    return futures

@SequentialIterator
class PregenerateIterator(DataIterator):
    """
    DataIterator to provide Date Based iteration


    !!! Example
        ```python
        PregenerateIterator(PipelineStep)

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialRandomIterator = RandomIterator()
        partialRandomIterator(PipelineStep)
        ```
    """    
    def __init__(
        self, index: DataStep, request_size: int = 10, min_size = 5, **kwargs
    ) -> None:
        """DataIterator to provide Date Based iteration and Random Sampling      
        
        Args:
            index (DataStep): 
                Prior Pipeline step
        """        
        super().__init__(index, **kwargs)
        self.request_size = request_size
        self.min_size = min_size
        self._info_ = dict(request_size = request_size, min_size = min_size)

    def __iter__(self):
        if not self._iterator_ready:
            raise RuntimeError(
                f"Iterator not set for {self.__class__.__name__}. Run .set_iterable()"
            )
        buffer = []
        next_futures = []
        current_time = EDITDatetime(self._start)
        while current_time < self._end:
            try:
                if len(next_futures) == 0 and len(buffer) < self.min_size:
                    next_futures = list(pregenerate(self.index, [current_time + (i* self._interval) for i in range(self.request_size)]))
                    current_time += self._interval * self.request_size

                if len(buffer) == 0:
                    for next in next_futures:
                        buffer.append(next.result())
                        # buffer.append(next)
                    next_futures = []

                yield buffer.pop(0)


            except self._error_to_catch as e:
                logging.info(e)
        

    @property
    def ignore_sanity(self):
        return True
