# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Iteration control for `pyearthtools.pipeline`

Allows a pipeline to be iterated over, and data retrieved from.
"""

from __future__ import annotations
from functools import cached_property
from abc import ABCMeta, abstractmethod

from typing import Any, Callable, Generator, Hashable, Iterable, Optional, Union
from pathlib import Path

import numpy as np
import random

import pyearthtools.data

from pyearthtools.pipeline.recording import PipelineRecordingMixin


class Iterator(PipelineRecordingMixin, metaclass=ABCMeta):
    """
    Base level Iterator.

    Provides the indexes from which to query the pipeline.

    All Iterator classes must implement this class, and provide
    `__iter__`, which should act as a generator.
    """

    @abstractmethod
    def __iter__(self) -> Generator[Hashable, None, None]:
        pass

    @cached_property
    def samples(self) -> tuple[Any, ...]:
        """Get tuple of samples returned by this `Iterator`."""
        return tuple(self)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __add__(self, other: Iterator):
        """
        Combine multiple `Sampler`'s together into a `SuperSampler`
        """
        if isinstance(other, SuperIterator):
            return SuperIterator(self, *other._iterators)

        elif isinstance(other, Iterator):
            return SuperIterator(self, other)

        return NotImplemented

    def __radd__(self, other: Iterator):
        """
        Combine multiple `Sampler`'s together into a `SuperSampler`
        """
        if isinstance(other, SuperIterator):
            return SuperIterator(*other._iterators, self)

        elif isinstance(other, Iterator):
            return SuperIterator(other, self)

        return NotImplemented

    def randomise(self, seed: Optional[int] = 42):
        """Randomise this interator"""
        return Randomise(self, seed=seed)


class Range(Iterator):
    """
    Range based Iterator

    Constructs a `range` object and yields all elements within.
    """

    def __init__(self, min: int, max: int, step: int = 1):
        """
        Construct Range Iterator

        Args:
            min (int):
                Minimum value of range
            max (int):
                Maximum value of range
            step (int, optional):
                Step of range. Defaults to 1.
        """
        super().__init__()
        self.record_initialisation()

        self._range = tuple(range(min, max, step))

    def __iter__(self) -> Generator[Hashable, None, None]:
        for i in self._range:
            yield i


class Predefined(Iterator):
    """
    Predefined Iterator

    Takes any iterable as provided, and yields all elements within.
    """

    _indexes: Iterable[Any]

    def __init__(self, indexes: Iterable[Any]):
        """
        Construct PreDefined iterator

        Args:
            indexes (Iterable[Any]):
                Iterable to get elements from
        """
        super().__init__()
        self.record_initialisation()
        self._indexes = indexes

    def __iter__(self) -> Generator[Any, None, None]:
        for i in self._indexes:
            yield i


class File(Predefined):
    """
    Iterate over elements in file

    Each line will be treated as a seperate index.
    """

    def __init__(self, file: Union[str, Path], type_conversion: Optional[Callable] = None):
        """
        Iterate over file

        Args:
            file (Union[str, Path]):
                File to load.
            type_conversion (Optional[Callable], optional):
                Function to convert lines in file with. Defaults to None.
        """
        super().__init__("")
        self.record_initialisation()

        self._indexes = open(file).readlines()
        if type_conversion:
            self._indexes = list(map(type_conversion, self._indexes))


class DateRange(Iterator):
    """
    DateRange Iterator

    Uses `pyearthtools.data.TimeRange` to create a range of times.
    """

    def __init__(
        self,
        start: str,
        end: str,
        interval,
        *,
        allowlist: Optional[Iterable[str]] = None,
        blocklist: Optional[Iterable[str]] = None,
    ):
        """
        Construct DateRange Iterator

        Args:
            start (str):
                Start time. Must be understandable by
                `pyearthtools.data.Petdt`.
            end (str):
                End time. Must be understandable by
                `pyearthtools.data.Petdt`.
            interval (Any):
                Interval between times. Must be understandable by
                `pyearthtools.data.TimeDelta`.
            allowlist: A list of pyearthtools.data.time.Petdt which should be filtered to
            blocklist: A list of pyearthtools.data.time.Petdt which should be skipped

        Note:
            You cannot specify both a blocklist and an allowlist.

            The entries in the blocklist and allowlist must be a complete list of exact
            dates. It is not possible to do fuzzy matching or range-based constraints
            at this stage. For example, if the underlying data is hourly, and a particular
            day needs to be skipped entirely, each hour of that day will need to be in the
            blocklist.

        Examples:
            >>> known_bad = ['2015-01-01T06']
            >>> iterator=DateRange(2015, 2016, interval='6 hours', blocklist=known_bad)

        """
        super().__init__()
        self.record_initialisation()

        import pyearthtools.data

        if allowlist and blocklist:
            raise ValueError("An allowlist and a blocklist cannot be specified at the same time")

        self.allowlist = allowlist
        if allowlist:
            self.allowlist = set(allowlist)

        self.blocklist = blocklist
        if blocklist:
            self.blocklist = set(blocklist)

        self._timerange = pyearthtools.data.TimeRange(start, end, interval)

    def __iter__(self) -> Generator[pyearthtools.data.Petdt, None, None]:

        # If in allowlist mode, yield only samples from the allow list
        if self.allowlist:
            print("Processing allow list")
            for i in self._timerange:
                if i in self.allowlist:
                    yield i

        # If in blocklist mode, yield only samples not in the blocklist
        elif self.blocklist:
            print("processing block list")
            for i in self._timerange:
                if i not in self.blocklist:
                    yield i

        # If not filtering, yield everything
        else:
            for i in self._timerange:
                yield i

    def randomise(self, seed: Optional[int] = 42):
        """Randomise this interator"""
        return DateRandomise(self, seed=seed)


class DateRangeLimit(DateRange):
    """
    DataRange configured with the number of samples from start

    Uses `pyearthtools.data.TimeRange` to create a range of times.
    """

    def __init__(self, start: str, interval: Any, num: int):
        """
        Construct DateRange with limit

        Args:
            start (str):
                Start time
            interval (Any):
                Interval between times. Must be understandable by
                `pyearthtools.data.TimeDelta`.
            num (int):
                Number of total samples to iterate over.
        """

        end = pyearthtools.data.Petdt(start) + (pyearthtools.data.TimeDelta(interval) * num)
        super().__init__(start, str(end), interval)


class DateRandomise(Iterator):
    """
    Wrap around another `Iterator` and randomly sample
    """

    def __init__(self, iterator: DateRange, seed: Union[int, None] = 42):
        """
        Randomise `iterator`

        Args:
            iterator (Iterator):
                Underlying `Iterator` to randomise.
            seed (Union[int, None], optional):
                Random selection seed. If None, will be `random`.
                Defaults to 42.
        """
        super().__init__()
        self.record_initialisation()

        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self._iterator = iterator
        self.valid_times = list(iterator._timerange)

        print("Calculated indexes")

        if getattr(iterator, "allowlist", None):
            self.valid_times = [t for t in self.valid_times if t in iterator.allowlist]
            print(len(self.valid_times))

        if getattr(iterator, "blocklist", None):
            self.valid_times = [t for t in self.valid_times if t not in iterator.blocklist]
            print(len(self.valid_times))

        random.shuffle(self.valid_times)

    def __iter__(self):

        for key in self.valid_times:
            yield key


class Randomise(Iterator):
    """
    Wrap around another `Iterator` and randomly sample
    """

    def __init__(self, iterator: Iterator, seed: Union[int, None] = 42):
        """
        Randomise `iterator`

        Args:
            iterator (Iterator):
                Underlying `Iterator` to randomise.
            seed (Union[int, None], optional):
                Random selection seed. If None, will be `random`.
                Defaults to 42.
        """
        super().__init__()
        self.record_initialisation()

        self.seed = seed
        self._samples = iterator.samples

    def __iter__(self):
        samples = list(self._samples)

        rng = np.random.default_rng(self.seed)

        while len(samples) > 0:
            yield samples.pop(rng.integers(0, len(samples)))


class SuperIterator(Iterator):
    """
    Iterate over a sequence of iterators
    """

    def __init__(self, *iterators: Iterator):
        """
        Create SuperIterator

        Args:
            *iterators (Iterator):
                Iterating is run sequentially, so order may be important.
        """
        super().__init__()
        self.record_initialisation()

        self._iterators = iterators

    def __getitem__(self, idx):
        """
        Get `Iterator` from iterators
        """
        return self._iterators[idx]

    def __len__(self):
        return len(self._iterators)

    def __iter__(self):
        for iterator in self._iterators:
            for i in iterator:
                yield i


class IterateResults:
    """
    A wrapper which informs pipeline, that the object in question should be iterated over,
    instead of yielded directly.

    Allows all attrs and items to be retrieved from underlying object.
    """

    def __init__(self, obj: Any):
        """
        Construct a IterateResults

        Args:
            obj (Any):
                Object that should be iterated over.
        """
        self._obj = obj

    def __getattr__(self, key):
        return getattr(self._obj, key)

    def __getitem__(self, *args):
        return self._obj.__getitem__(*args)

    def iterate_over_object(self) -> Generator[Any, None, None]:
        """Iterate over underlying object"""
        for i in self._obj:
            yield i
