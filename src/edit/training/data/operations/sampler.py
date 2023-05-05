from __future__ import annotations

from abc import abstractmethod

import random
import warnings
import numpy as np

from edit.training.data.templates import (
    DataOperation,
    DataStep,
)
from edit.training.data.sequential import Sequential, SequentialIterator


class DataSampler(DataOperation):
    """
    DataOperation Child to override `__iter__` method to provide a way of sampling data

    !!! Example
        ```python
        DataSamplerChild(PipelineStep)

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialDataSamplerChild = DataSamplerChild()
        partialDataSamplerChild(PipelineStep)
        ```
    """
    def __init__(self, index : DataStep) -> None:
        """DataOperation to sample incoming data         
        
        Args:
            index (DataStep): 
                Underlying DataStep to get data from
        """        
        super().__init__(
            index, apply_func=None, undo_func=None, apply_iterator=True, apply_get=False
        )

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError(f"Child Filter must define Iterator")


@SequentialIterator
class RandomSampler(DataSampler):
    """
    DataSampler to collect a list of samples and randomly choose one to give back
    """
    def __init__(self, index: DataStep, buffer_size: int = 10) -> None:
        """Semi-Randomlly Sample a buffer of data samples

        Args:
            index (DataStep): 
                Underlying DataStep to retrieve data from
            buffer_size (int, optional): 
                Size of buffer to create and to sample from. 
                Larger buffer will improve the randomness. Defaults to 10.
        """        
        super().__init__(index)
        buffer_size = max(1, buffer_size)
        self.buffer_size = buffer_size

    def __iter__(self):
        buffer = []
        iterator = iter(self.index)
        while True:
            try:
                while len(buffer) < self.buffer_size:
                    buffer.append(iterator.__next__())
                yield buffer.pop(random.randint(0,len(buffer)))
            except StopIteration:
                return


@SequentialIterator
class DropOut(DataSampler):
    """
    DataSampler to randomally drop out data
    """
    def __init__(self, index: DataStep, chance: int = 0) -> None:
        """Randomly drop out samples from iteration

        Args:
            index (DataStep): 
                Underlying DataStep to retrieve data from
            chance (int, optional): 
                Chance each data step is dropped. Percentage between 0 & 100. Defaults to 0.
        """        
        super().__init__(index)
        if chance < 0 or chance > 100:
            raise ValueError(f"Invalid `chance` given. {chance!r}. Must be between 0 and 100.")
        self.chance = chance
        if chance > 50:
            warnings.warn(f"Dropout chance is high {chance!r}, unlikely to be an effective training pipeline", RuntimeWarning)
        self.__doc__ = f"DataSampler with a {chance}% chance to drop data"

    def __iter__(self):
        for data in self.index:
            if random.randint(0,100) < self.chance:
                continue
            yield data