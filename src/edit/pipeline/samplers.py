# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
Sampling Setups for `edit.pipeline`

Allows data to be on the fly iterated through, and sampled from the stream.
"""

from __future__ import annotations
from abc import ABCMeta, abstractmethod

from typing import Any, Generator, Union

from edit.pipeline.recording import PipelineRecordingMixin


class EmptyObject:
    """
    Empty object to be skipped by Iteration

    Used to mark where the sampler cannot return data.
    """

    pass


class Sampler(PipelineRecordingMixin, metaclass=ABCMeta):
    """
    Base level Sampler

    All sampler classes must implement this class, and provide
    `generator`, which should act as a generator.

    See `DefaultSampler` for an example, and the process to make a sampler.

    """

    @abstractmethod
    def generator(self) -> Generator[Any, Any, Any]:
        """
        Generator to control the sampling of data.

        When passed None, and no samples remain within, should exit.

        How to:
            Yield an `EmptyObject` to begin with, and capture what is sent.
            Run sampling routine, yield `EmptyObject` if sampler cannot return obj,
            else return obj.
            Exit when None is encountered.
            If any stored within the sampler, yield them all afterwards.

        Yields:
            Generator[Any, Any, Any]:
                Sampling of data
        """
        pass

    def __add__(self, other: Sampler):
        """
        Combine multiple `Sampler`'s together into a `SuperSampler`
        """
        if isinstance(other, SuperSampler):
            return SuperSampler(self, *other)

        elif isinstance(other, Sampler):
            return SuperSampler(self, other)

        return NotImplemented


class Default(Sampler):
    """
    Default Sampler

    Simply passes back any object given to it.
    """

    def generator(self) -> Generator[Any, Any, Any]:
        obj = EmptyObject()  # Yields an Empty Object to start with

        # Run forever until None is encountered
        while True:
            # Yield the prior object and capture what is sent
            obj = yield obj
            # If None is encountered, exit the generator
            if obj is None:
                break


class SuperSampler(Sampler):
    """
    A collection of `Sampler`'s run one after another.
    """

    def __init__(self, *samplers: Sampler):
        """
        Construct a new `SuperSampler` from multiple `Sampler`'s.

        Args:
            *samplers (Sampler):
                Sampling is run sequentially, so order may be important.

        """
        super().__init__()
        self.record_initialisation()

        self._samplers = samplers

    def __iter__(self):
        """Iterate over `Sampler` which makes up this `SuperSampler`"""
        for sampler in self._samplers:
            yield sampler

    def __getitem__(self, idx):
        """
        Get `Sampler` from samplers
        """
        return self._samplers[idx]

    def __len__(self):
        return len(self._samplers)

    def generator(self) -> Generator[Any, Any, Any]:
        """
        SuperSampler version of `generator`,

        Runs each `Sampler` one after another.
        """
        obj = EmptyObject()
        # Initial yield

        generators = [samp.generator() for samp in self._samplers]
        [next(gen) for gen in generators]
        # Setup generators from `Samplers`.

        while True:
            # Get obj, and yield sampled obj
            obj = yield obj
            # If None exit
            if obj is None:
                break

            # Run obj through the samplers
            # When one yields data, it will be yielded.
            for sampler in generators:
                obj = sampler.send(obj)

        # Empty samplers after exhausted.
        for i, sampler in enumerate(generators):
            for sam in sampler:
                if i < len(generators) - 1:
                    new_sam = generators[i + 1].send(sam)
                    yield new_sam
                if i == len(generators) - 1:
                    yield sam


class Random(Sampler):
    """
    Randomly sample objects from stream.

    Builds a buffer of controllable length from which to sample from,
    once size is reached.
    """

    def __init__(self, buffer_len: int, seed: Union[int, None] = 42):
        """
        Construct a Random Sampler

        Args:
            buffer_len (int):
                Length of buffer to build.
                No objects will be yielded until this length is reached or exhausted.
            seed (Union[int, None], optional):
                Seed to initialise rng module with. Defaults to 42.
        """
        super().__init__()
        self.record_initialisation()

        self._buffer_len = buffer_len
        self._seed = seed

    def generator(self) -> Generator[Any, Any, Any]:
        import numpy as np

        buffer = []
        begin = False
        rng = np.random.default_rng(self._seed)

        while len(buffer) > 0 or not begin:
            if begin or len(buffer) > self._buffer_len:
                begin = True
                obj = yield buffer.pop(rng.integers(0, len(buffer)))
            else:
                obj = yield EmptyObject()

            if len(buffer) > self._buffer_len:
                begin = True
            if obj is None:
                break
            if not isinstance(obj, EmptyObject):
                buffer.append(obj)

        while len(buffer) > 0:
            yield buffer.pop(rng.integers(0, len(buffer)))


class DropOut(Sampler):
    """
    DropOut samples from the stream at a given interval.
    """

    def __init__(self, step: int, yield_on_step: bool = False):
        """
        Construct a DropOut Sampler

        Args:
            step (int):
                Step value in which to drop out objects.
                or if `yield_on_step` when to yield data
            yield_on_step (bool, optional):
                Reverse behaviour of this Sampler, such that on `step` yield objects. Defaults to False.
        """
        super().__init__()
        self.record_initialisation()

        self._step = step
        self._yield_on_step = yield_on_step

    def generator(self) -> Generator[Any, Any, Any]:
        i = 0
        obj = EmptyObject()

        while True:
            if (i % self._step) == 0:
                obj = yield obj if self._yield_on_step else EmptyObject()
            else:
                obj = yield EmptyObject() if self._yield_on_step else obj
            if obj is None:
                break

            i += 1


class RandomDropOut(Sampler):
    """
    Randomly DropOut samples from the stream.
    """

    def __init__(self, chance: float, seed: int = 42):
        """
        Construct a RandomDropOut Sampler

        Args:
            chance (float):
                Chance for samples to dropped, between 0 & 100.
            seed (int, optional):
                Seed to initialise rng module with. Defaults to 42.
        """
        super().__init__()
        self.record_initialisation()

        self._chance = chance
        self._seed = seed

    def generator(self) -> Generator[Any, Any, Any]:
        import numpy as np

        rng = np.random.default_rng(self._seed)

        obj = EmptyObject()

        while True:
            obj = yield obj
            if obj is None:
                break
            if (rng.random() * 100) < self._chance:
                obj = EmptyObject()


__all__ = [
    "Sampler",
    "SuperSampler",
    "EmptyObject",
    "Random",
    "DropOut",
    "RandomDropOut",
]
