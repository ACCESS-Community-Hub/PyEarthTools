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
from typing import TypeVar, Union

import xarray as xr

from pyearthtools.data.utils import parse_path

from pyearthtools.utils.decorators import BackwardsCompatibility
from pyearthtools.pipeline.operation import Operation


FILE = Union[str, Path]

T = TypeVar("T", xr.Dataset, xr.DataArray)


class xarrayNormalisation(Operation):
    """
    Parent xarray Normalisation
    """

    _override_interface = "Serial"

    @classmethod
    def open_file(cls, file: FILE) -> xr.Dataset:
        """Open xarray file"""
        return xr.open_dataset(parse_path(file))

    def __init__(self):
        super().__init__(split_tuples=True, recursively_split_tuples=True, recognised_types=(xr.Dataset, xr.DataArray))

    def apply_func(self, sample: T) -> T:
        return self.normalise(sample)

    def undo_func(self, sample: T) -> T:
        return self.unnormalise(sample)

    @abstractmethod
    def normalise(self, sample: T) -> T:
        return sample

    @abstractmethod
    def unnormalise(self, sample: T) -> T:
        return sample


class Anomaly(xarrayNormalisation):
    """Anomaly Normalisation"""

    def __init__(self, mean: FILE):
        super().__init__()
        self.record_initialisation()
        self.mean = self.open_file(mean)

    def normalise(self, sample):
        return sample - self.mean

    def unnormalise(self, sample):
        return sample + self.mean


class Deviation(xarrayNormalisation):
    """Deviation Normalisation"""

    def __init__(self, mean: FILE, deviation: FILE):
        super().__init__()
        self.record_initialisation()
        self.mean = self.open_file(mean)
        self.deviation = self.open_file(deviation)

    def normalise(self, sample):
        return (sample - self.mean) / self.deviation

    def unnormalise(self, sample):
        return (sample * self.deviation) + self.mean


class Division(xarrayNormalisation):
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


class Evaluated(xarrayNormalisation):
    """
    `eval` based normalisation
    """

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
