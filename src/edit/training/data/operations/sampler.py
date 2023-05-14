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
from edit.training.data.warnings import PipelineResourceWarning

class DataSampler(DataOperation):
    """
    DataOperation Child to override `__iter__` method to provide a way of sampling data
    Parent Class of Data Samplers

    !!! Warning
        Cannot be used directly, this is simply the parent class to provide a structure

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

    def _sanity(self):
        return 'Sampler'

@SequentialIterator
class RandomSampler(DataSampler):
    """
    DataSampler to collect a list of samples and randomly choose one to give back

    !!! Example
        ```python
        RandomSampler(PipelineStep, buffer_size = 10)

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialRandomSampler = RandomSampler(buffer_size = 10)
        partialRandomSampler(PipelineStep)
        ```
    """
    def __init__(self, index: DataStep, buffer_size: int = 10, seed: int = 42) -> None:
        """Semi-Randomly Sample a buffer of data samples

        Args:
            index (DataStep): 
                Underlying DataStep to retrieve data from
            buffer_size (int, optional): 
                Size of buffer to create and to sample from. 
                Larger buffer will improve the randomness. Defaults to 10.
            seed (int, optional):
                Seed of random number generator. Defaults to 42.
        """        
        super().__init__(index)
        buffer_size = max(1, buffer_size)
        self.buffer_size = buffer_size
        self.seed = seed
        self._info_ = dict(buffer_size = buffer_size, seed = seed)

    def __iter__(self):
        buffer = []
        iterator = iter(self.index)
        rng = np.random.default_rng(self.seed)
        while True:
            try:
                while len(buffer) < self.buffer_size:
                    buffer.append(iterator.__next__())
                yield buffer.pop(rng.integers(0,len(buffer)))
            except StopIteration:
                while len(buffer) > 0:
                    yield buffer.pop(rng.integers(0,len(buffer)))
                return


@SequentialIterator
class RandomDropOut(DataSampler):
    """
    DataSampler to randomally drop out data

    !!! Example
        ```python
        RandomDropOut(PipelineStep, chance = 10)

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialRandomDropOut = RandomDropOut(chance = 10)
        partialRandomDropOut(PipelineStep)
        ```
    """
    def __init__(self, index: DataStep, chance: int = 0, seed: int = 42) -> None:
        """Randomly drop out samples from iteration

        Args:
            index (DataStep): 
                Underlying DataStep to retrieve data from
            chance (int, optional): 
                Chance each data step is dropped. Percentage between 0 & 100. Defaults to 0.
            seed (int, optional):
                Seed of random number generator. Defaults to 42.
        """        
        super().__init__(index)
        if chance < 0 or chance > 100:
            raise ValueError(f"Invalid `chance` given. {chance!r}. Must be between 0 and 100.")
            
        self.chance = chance
        self.seed = seed

        if chance > 50:
            warnings.warn(f"Dropout chance is high {chance!r}, unlikely to be an effective training pipeline", PipelineResourceWarning)

        self.__doc__ = f"DataSampler with a {chance}% chance to drop data"
        self._info_ = dict(chance = chance, seed = seed)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        for data in self.index:
            if rng.integers(0,100) < self.chance:
                continue
            yield data

@SequentialIterator
class DropOut(DataSampler):
    """
    DataSampler to drop out data at given step interval

    !!! Example
        ```python
        DropOut(PipelineStep, step = 10)

        ## As this is decorated with @SequentialIterator, it can be partially initialised

        partialDropOut = DropOut(step = 10)
        partialDropOut(PipelineStep)
        ```
    """
    def __init__(self, index: DataStep, step: int, yield_on_step: bool = False) -> None:
        """Drop out samples from iteration at given step interval

        Args:
            index (DataStep): 
                Underlying DataStep to retrieve data from
            step (int): 
                Step value in which to drop out data, or if `yield_on_step` when to yield data
            yield_on_step (bool, optional):
                Reverse behaviour of this Sampler, such that on `step` yield data.

        ??? Warning:
            If `step` is set to `1` and `yield_on_step` == False, this will drop data every iteration, returning nothing.
        """        
        super().__init__(index)
        self.step_val = step
        self.yield_on_step = yield_on_step

        self.__doc__ = f"DataSampler {'dropping' if yield_on_step else 'returning'} every {step} data sample."
        self._info_ = dict(step = step, yield_on_step = yield_on_step)

    def __iter__(self):
        for i, data in enumerate(self.index):
            if (i % self.step_val) == 0:
                if self.yield_on_step:
                    yield data
                else:
                    continue
            else:
                if self.yield_on_step:
                    continue
                else:
                    yield data
