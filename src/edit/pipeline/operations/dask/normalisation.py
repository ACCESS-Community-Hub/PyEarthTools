# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

# type: ignore[reportPrivateImportUsage]

from abc import abstractmethod
from pathlib import Path
from typing import Union


import numpy as np
import dask.array as da


from edit.data.utils import parse_path

from edit.utils.decorators import BackwardsCompatibility
from edit.pipeline.operations.dask.dask import DaskOperation


FILE = Union[str, Path]

__all__ = ["daskNormalisation", "Anomaly", "Deviation", "Evaluated"]


class daskNormalisation(DaskOperation):
    """
    Parent dask normalisation class

    """

    _override_interface = ["Serial"]

    @classmethod
    def open_file(cls, file: FILE) -> da.Array:
        """Open dask file"""
        return da.from_array(np.load(str(parse_path(file))))

    def __init__(self, expand: bool = True):
        super().__init__(split_tuples=True, recursively_split_tuples=True, recognised_types=(da.Array))
        self._expand = expand

    def apply_func(self, sample: da.Array) -> da.Array:
        return self.normalise(sample)

    def undo_func(self, sample: da.Array) -> da.Array:
        return self.unnormalise(sample)

    @abstractmethod
    def normalise(self, sample: da.Array) -> da.Array:
        return sample

    @abstractmethod
    def unnormalise(self, sample: da.Array) -> da.Array:
        return sample

    def expand(self, factor, sample):
        if not self._expand:
            return factor
        if len(factor.shape) == len(sample.shape):
            return factor

        dims = tuple(range(len(factor.shape), len(sample.shape)))
        return np.expand_dims(factor, dims)

class Anomaly(daskNormalisation):
    """Anomaly Normalisation"""

    _numpy_counterpart = "normalisation.Anomaly"

    def __init__(self, mean: FILE, expand: bool = True):
        super().__init__(expand)
        self.record_initialisation()

        self.mean = self.open_file(mean)

    def normalise(self, sample):
        return sample - self.expand(self.mean, sample)

    def unnormalise(self, sample):
        return sample + self.expand(self.mean, sample)


class Deviation(daskNormalisation):
    """Deviation Normalisation"""

    _numpy_counterpart = "normalisation.Deviation"

    def __init__(self, mean: FILE, deviation: FILE, expand: bool = True):
        super().__init__(expand)
        self.record_initialisation()

        self.mean = self.open_file(mean)
        self.deviation = self.open_file(deviation)

    def normalise(self, sample):
        return (sample - self.expand(self.mean, sample)) / self.expand(self.deviation, sample)

    def unnormalise(self, sample):
        return (sample * self.expand(self.deviation, sample)) + self.expand(self.mean, sample)



class Division(daskNormalisation):
    """Division based Normalisation"""

    _numpy_counterpart = "normalisation.Division"

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


class Evaluated(daskNormalisation):
    """
    `eval` based normalisation
    """

    _numpy_counterpart = "normalisation.Evaluated"

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
