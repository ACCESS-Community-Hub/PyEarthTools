# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from abc import abstractmethod
from pathlib import Path
from typing import Union

import numpy as np

from edit.utils.decorators import BackwardsCompatibility
from edit.pipeline.operation import Operation


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
        return np.load(file)

    def __init__(self):
        super().__init__(split_tuples=True, recursively_split_tuples=True, recognised_types=(np.ndarray))

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


class Anomaly(numpyNormalisation):
    """Anomaly Normalisation"""

    _interface_kwargs = {"Delayed": {"name": "AnomalyNormalisation"}}

    def __init__(self, mean: FILE):
        super().__init__()
        self.record_initialisation()

        self.mean = self.open_file(mean)

    def normalise(self, sample):
        return sample - self.mean

    def unnormalise(self, sample):
        return sample + self.mean


class Deviation(numpyNormalisation):
    """Deviation Normalisation"""

    _interface_kwargs = {"Delayed": {"name": "DeviationNormalisation"}}

    def __init__(self, mean: FILE, deviation: FILE):
        super().__init__()
        self.record_initialisation()

        self.mean = self.open_file(mean)
        self.deviation = self.open_file(deviation)

    def normalise(self, sample):
        return (sample - self.mean) / self.deviation

    def unnormalise(self, sample):
        return (sample * self.deviation) + self.mean


class Division(numpyNormalisation):
    """Division based Normalisation"""

    def __init__(self, division_factor: FILE):
        super().__init__()
        self.record_initialisation()

        self.division_factor = self.open_file(division_factor)

    def normalise(self, sample):
        return sample / self.division_factor

    def unnormalise(self, sample):
        return sample * self.division_factor


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
