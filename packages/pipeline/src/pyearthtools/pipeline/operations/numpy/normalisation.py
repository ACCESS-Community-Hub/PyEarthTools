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


from abc import abstractmethod
from pathlib import Path
from typing import Union

import numpy as np

from pyearthtools.data.utils import parse_path

from pyearthtools.utils.decorators import BackwardsCompatibility
from pyearthtools.pipeline.operation import Operation


FILE = Union[str, Path]

__all__ = ["numpyNormalisation", "Anomaly", "Deviation"]


class numpyNormalisation(Operation):
    """
    Parent numpy normalisation class

    """

    _override_interface = ["Delayed", "Serial"]

    @classmethod
    def open_file(cls, file: FILE) -> np.ndarray:
        """Open numpy file"""
        return np.load(str(parse_path(file)))  # type: ignore

    def __init__(self, expand: bool = True):
        """Base Numpy Normalisation

        Args:
            expand (bool, optional): 
                Expand normalisation arrays to shape of `sample` by appending axis of size 1. Defaults to True.
        """        
        super().__init__(split_tuples=True, recursively_split_tuples=True, recognised_types=(np.ndarray))
        self._expand = expand

    def apply_func(self, sample: np.ndarray) -> np.ndarray:
        return self.normalise(sample)

    def undo_func(self, sample: np.ndarray) -> np.ndarray:
        return self.unnormalise(sample)

    @abstractmethod
    def normalise(self, sample: np.ndarray) -> np.ndarray:
        return sample

    @abstractmethod
    def unnormalise(self, sample: np.ndarray) -> np.ndarray:
        return sample

    def expand(self, factor, sample):
        if not self._expand:
            return factor
        if len(factor.shape) == len(sample.shape):
            return factor

        dims = tuple(range(len(factor.shape), len(sample.shape)))
        return np.expand_dims(factor, dims)


class Anomaly(numpyNormalisation):
    """Anomaly Normalisation"""

    _interface_kwargs = {"Delayed": {"name": "AnomalyNormalisation"}}

    def __init__(self, mean: FILE, expand: bool = True):
        super().__init__(expand)
        self.record_initialisation()

        self.mean = self.open_file(mean)

    def normalise(self, sample):
        return sample - self.expand(self.mean, sample)

    def unnormalise(self, sample):
        return sample + self.expand(self.mean, sample)


class Deviation(numpyNormalisation):
    """Deviation Normalisation"""

    _interface_kwargs = {"Delayed": {"name": "DeviationNormalisation"}}

    def __init__(self, mean: FILE, deviation: FILE, expand: bool = True):
        super().__init__(expand)
        self.record_initialisation()

        self.mean = self.open_file(mean)
        self.deviation = self.open_file(deviation)

    def normalise(self, sample):
        return (sample - self.expand(self.mean, sample)) / self.expand(self.deviation, sample)

    def unnormalise(self, sample):
        return (sample * self.expand(self.deviation, sample)) + self.expand(self.mean, sample)


class Division(numpyNormalisation):
    """Division based Normalisation"""

    def __init__(self, division_factor: FILE, expand: bool = True):
        super().__init__(expand)
        self.record_initialisation()

        self.division_factor = self.open_file(division_factor)

    def normalise(self, sample):
        return sample / self.expand(self.division_factor, sample)

    def unnormalise(self, sample):
        return sample * self.expand(self.division_factor, sample)


@BackwardsCompatibility(Division)
def TemporalDifference(*a, **k): ...


class Evaluated(numpyNormalisation):
    """
    `eval` based normalisation
    """

    _interface_kwargs = {"Delayed": {"name": "EvalNormalisation"}}

    def __init__(self, normalisation_eval: str, unnormalisation_eval: str, **kwargs):
        """
        Run a normalisation calculation using `eval`.

        Will get all `kwargs` passed to this class, and `sample` as the data to be normalised.

        All kwargs will be loaded from file if a str.

        Args:
            normalisation_eval (str):
                Normalisation eval str
            unnormalisation_eval (str):
                Unnoralisation eval str
        """
        super().__init__()
        self.record_initialisation()

        for key, val in kwargs.items():
            if isinstance(val, (str, Path)):
                kwargs[key] = self.open_file(val)

        self._normalisation_eval = normalisation_eval
        self._unnormalisation_eval = unnormalisation_eval
        self._kwargs = kwargs

    def normalise(self, sample):
        return eval(self._normalisation_eval, {"sample": sample, **self._kwargs})

    def unnormalise(self, sample):
        return eval(self._unnormalisation_eval, {"sample": sample, **self._kwargs})
