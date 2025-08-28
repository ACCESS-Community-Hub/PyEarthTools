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
import os

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
        return self.denormalise(sample)

    @abstractmethod
    def normalise(self, sample: T) -> T:
        return sample

    @abstractmethod
    def denormalise(self, sample: T) -> T:
        return sample


class Anomaly(xarrayNormalisation):
    """Anomaly Normalisation"""

    def __init__(self, mean: FILE):
        super().__init__()
        self.record_initialisation()
        self.mean = self.open_file(mean)

    def normalise(self, sample):
        return sample - self.mean

    def denormalise(self, sample):
        return sample + self.mean


class MagicNorm(xarrayNormalisation):
    """
    Automatically normalise any variables

    For every data variable in any xarray passing through
    Based on some sample period (20 samples by default)
    Calculate the mean and standard deviation for that data variable
    Once sufficient samples are observed, cache the mean and standard devation together with the reference period
    By default will use the first 20 samples it sees
    Apply the normalisation to all data
    Denormalise accordingly
    """

    def __init__(self, cache_dir=".", samples_needed=20):
        super().__init__()
        self.record_initialisation()
        self.vars = {}
        # import random
        # myid = random.randint(0, 20)
        # print(f"Initialising {myid}")

        self.means_filename = os.path.join(cache_dir, "magic_means.nc")
        self.deviation_filename = os.path.join(cache_dir, "magic_std.nc")
        self.samples_needed = samples_needed
        self.sample_count = 0
        self.samples = []
        self.mean = None
        self.deviation = None

        if os.path.exists(self.means_filename):
            # print(f"Found file for {myid})")
            self.mean = xr.load_dataset(self.means_filename)
            self.deviation = xr.load_dataset(self.deviation_filename)
            self.samples_needed = 0

    def update_norms(self, sample):

        # Return early if norms already well calculated
        if self.sample_count >= self.samples_needed:
            return

        # This can happen in a multithreading situation
        # Throw out own weights and all use the same pls
        if os.path.exists(self.means_filename):
            self.mean = xr.load_dataset(self.means_filename)
            self.deviation = xr.load_dataset(self.deviation_filename)
            self.samples_needed = 0

        # Update the calculations
        self.samples.append(sample)
        self.sample_count = len(self.samples)
        ds = xr.concat(self.samples, dim="samples")
        self.mean = ds.mean()
        self.deviation = ds.std()

        # Cache to disk once we have enough data
        if self.sample_count >= self.samples_needed:
            if os.path.exists(self.means_filename):
                self.mean = xr.load_dataset(self.means_filename)
                self.deviation = xr.load_dataset(self.deviation_filename)
                self.samples_needed = 0
            else:
                self.mean.to_netcdf(self.means_filename)
                self.deviation.to_netcdf(self.deviation_filename)

    def normalise(self, sample):

        if self.sample_count < self.samples_needed:
            self.update_norms(sample)

        return (sample - self.mean) / self.deviation

    def denormalise(self, sample):

        return (sample * self.deviation) + self.mean


class Deviation(xarrayNormalisation):
    """Deviation Normalisation"""

    def __init__(
        self,
        mean: FILE | xr.Dataset | xr.DataArray | float,
        deviation: FILE | xr.Dataset | xr.DataArray | float,
        debug=False,
    ):
        """
        Each argument take take a Dataset, DataArray, float or file object.

        Args:
            mean: mean values to subtract
            deviation: deviation value to divide by
        """
        super().__init__()
        self.record_initialisation()

        if debug:
            import pdb

            pdb.set_trace()

        if isinstance(mean, xr.Dataset):
            self.mean = mean
        elif isinstance(mean, xr.DataArray):
            self.mean = mean
        elif isinstance(mean, float):
            self.mean = mean
        else:
            self.mean = self.open_file(mean)

        if isinstance(deviation, xr.Dataset):
            self.deviation = deviation
        elif isinstance(deviation, xr.DataArray):
            self.deviation = deviation
        elif isinstance(deviation, float):
            self.deviation = deviation
        else:
            self.deviation = self.open_file(deviation)

    def normalise(self, sample):
        return (sample - self.mean) / self.deviation

    def denormalise(self, sample):
        return (sample * self.deviation) + self.mean


class Division(xarrayNormalisation):
    """Division based Normalisation"""

    def __init__(self, division_factor: FILE):
        super().__init__()
        self.record_initialisation()

        self.division_factor = self.open_file(division_factor)

    def normalise(self, sample):
        return sample / self.division_factor

    def denormalise(self, sample):
        return sample * self.division_factor


class SingleValueDivision(xarrayNormalisation):
    """Division based Normalisation"""

    def __init__(self, division_factor: float):
        super().__init__()
        self.record_initialisation()

        self.division_factor = division_factor

    def normalise(self, sample):
        return sample / self.division_factor

    def denormalise(self, sample):
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

    def denormalise(self, sample):
        return eval(self._unnormalisation_eval, {"sample": sample, **self._kwargs})
