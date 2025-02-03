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


# type: ignore[reportPrivateImportUsage]


from __future__ import annotations


import numpy as np
import xarray as xr
import pandas as pd

from pyearthtools.data.derived.derived import AdvancedTimeDerivedValue

DASK_IMPORTED = True
try:
    import dask.array as da
except (ModuleNotFoundError, ImportError):
    DASK_IMPORTED = False

ar = da if DASK_IMPORTED else np


def array(x):
    return (
        da.from_array(x) if not isinstance(x, da.Array) else (da.rechunk(x) if DASK_IMPORTED else np.array(x))
    )  # noqa: E731


to_list = lambda x: x.compute().tolist() if DASK_IMPORTED else lambda x: x.tolist()  # noqa: E731


class Insolation(AdvancedTimeDerivedValue):
    """
    Calculate the approximate solar insolation for given dates.

    Use `like` to mimic a dataset, it must have `latitude` and `longitude` in the coords.
    """

    def __init__(
        self,
        latitude,
        longitude,
        interval: tuple[int, str] | int | str | None = None,
        *,
        S: float = 1.0,
        daily: bool = False,
        clip_zero: bool = True,
    ):
        """
        Calculate the approximate solar insolation for given dates.

        For an example reference, see:
        https://brian-rose.github.io/ClimateLaboratoryBook/courseware/insolation.html

        Args:
            latitude (np.ndarray | list):
                1d or 2d array of latitudes
            longitude (np.ndarray | list):
                1d or 2d array of longitudes (0-360deg). If 2d, must match the shape of latitude.
            interval (tuple[int, str] | int | str | None, optional):
                TimeDelta of data. E.g. `6 hour`. Used for series retrieval. Can be None to not default have interval awareness.
                Defaults to None.
            S (float, optional):
                scaling factor (solar constant). Defaults to 1.0.
            daily (bool, optional):
                if True, return the daily max solar radiation (lat and day of year dependent only). Defaults to False.
            clip_zero (bool, optional):
                if True, set values below 0 to 0. Defaults to True.

        Raises:
            ValueError:
                If `latitude` or `longitude` are invalid.
        """
        super().__init__(interval, split_time=False)
        self.record_initialisation()

        latitude = array(np.array(latitude))
        longitude = array(np.array(longitude))

        self.update_initialisation(latitude=to_list(latitude), longitude=to_list(longitude))

        if len(latitude.shape) != len(longitude.shape):
            raise ValueError("'latitude' and 'longitude' must have the same number of dimensions")
        if len(latitude.shape) >= 2 and latitude.shape != longitude.shape:
            raise ValueError(f"shape mismatch between latitude ({latitude.shape} and longitude ({longitude.shape})")
        if len(latitude.shape) == 1:
            longitude, latitude = np.meshgrid(longitude, latitude)

        self._latitude = latitude
        self._longitude = longitude
        self._S = S
        self._daily = daily
        # self._enforce_2d = enforce_2d
        self._clip_zero = clip_zero

    def derive(self, time) -> xr.Dataset:
        n_dim = len(self._latitude.shape)

        time = np.atleast_1d(time)

        # Constants for year 1995 (standard in climate modeling community)
        # Obliquity of Earth
        eps = 23.4441 * np.pi / 180.0
        # Eccentricity of Earth's orbit
        ecc = 0.016715
        # Longitude of the orbit's perihelion (when Earth is closest to the sun)
        om = 282.7 * np.pi / 180.0
        beta = ar.sqrt(1 - ecc**2.0)

        # Get the day of year as a float.
        start_years = np.array([pd.Timestamp(pd.Timestamp(d).year, 1, 1) for d in time], dtype="datetime64")
        days_arr = (np.array(time, dtype="datetime64") - start_years) / np.timedelta64(1, "D")

        for d in range(n_dim):
            days_arr = np.expand_dims(days_arr, -1)
        # For daily max values, set the day to 0.5 and the longitude everywhere to 0 (this is approx noon)
        if self._daily:
            days_arr = 0.5 + np.round(days_arr)
            new_lon = self._longitude.copy().astype(np.float32)
            new_lon[:] = 0.0
        else:
            new_lon = self._longitude.astype(np.float32)
        # Longitude of the earth relative to the orbit, 1st order approximation
        lambda_m0 = ecc * (1.0 + beta) * ar.sin(om)
        lambda_m = lambda_m0 + 2.0 * np.pi * (days_arr - 80.5) / 365.0
        lambda_ = lambda_m + 2.0 * ecc * ar.sin(lambda_m - om)
        # Solar declination
        dec = ar.arcsin(ar.sin(eps) * ar.sin(lambda_))
        # Hour angle
        h = 2 * np.pi * (days_arr + new_lon / 360.0)
        # Distance
        rho = (1.0 - ecc**2.0) / (1.0 + ecc * ar.cos(lambda_ - om))

        # Insolation

        sol = (
            self._S
            * (
                ar.sin(np.pi / 180.0 * self._latitude[None, ...]) * ar.sin(dec)
                - ar.cos(np.pi / 180.0 * self._latitude[None, ...]) * ar.cos(dec) * ar.cos(h)
            )
            * rho**-2.0
        )
        if self._clip_zero:
            sol[sol < 0.0] = 0.0

        insolation = xr.Dataset(
            data_vars={"insolation": (["time", "latitude", "longitude"], array(sol))},
            coords={"time": time, "latitude": self._latitude[:, 0], "longitude": self._longitude[0, :]},
        )
        insolation.time.encoding.update(
            {"dtype": "int32", "units": "hours since 1900-01-01 00:00:00.0", "calendar": "gregorian"}
        )
        return insolation
