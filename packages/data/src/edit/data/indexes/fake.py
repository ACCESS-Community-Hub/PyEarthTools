# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations

import xarray as xr
import numpy as np

from edit.data.time import EDITDatetime
from edit.data.indexes.indexes import AdvancedTimeDataIndex

from edit.data.indexes.decorators import variable_modifications


class FakeIndex(AdvancedTimeDataIndex):
    """
    Get fake random seed data at a given interval.

    Appears to be a latitude longitude dataset.

    As this implements the `AdvancedTimeDataIndex`, selecting lower resolutions behaves correctly.
    """

    @property
    def _desc_(self):
        return {
            "singleline": "Fake Data Indexer",
        }

    @variable_modifications()
    def __init__(
        self,
        variable: list[str] | str = "data",
        *,
        interval=(1, "hour"),
        max_value: float = 1.0,
        data_size: tuple[int, int] = (128, 128),
        random: bool = True,
        **kwargs,
    ):
        """
        Setup fake data indexer

        Args:
            variable (list[str] | str, optional):
                Name/Names of variables. Defaults to "data".
            interval (tuple, optional):
                Interval of data. Defaults to (1, "hour").
            max_value (float, optional):
                Maximum value in random data. Defaults to 1.0.
            data_size (tuple[int, int], optional):
                Lat, Lon size. Defaults to (128, 128).
            random (bool):
                Whether to make random data, if not, will make data with `max_value` as all values.
        """
        super().__init__(data_interval=interval, **kwargs)
        self.record_initialisation()
        self.variable = variable

        self.rng = np.random.default_rng(42)
        self.max_value = float(max_value)
        self.data_size = tuple(map(int, data_size))

        self._data_function = self.rng.random if random else np.ones

    def get(self, time: EDITDatetime | str) -> xr.Dataset:
        time = EDITDatetime(time)

        def make_data(name) -> xr.Dataset:
            data = xr.DataArray(
                data=self._data_function((1, self.data_size[0], self.data_size[1])) * self.max_value,
                dims=["time", "latitude", "longitude"],
                coords=dict(
                    time=[time.datetime64()],
                    latitude=(["latitude"], np.linspace(90, -90, self.data_size[0])),
                    longitude=(["longitude"], np.linspace(0, 360, self.data_size[1])),
                ),
                attrs=dict(
                    description="Fake Data.",
                    long_name="Fake Data.",
                    standard_name="FakeData",
                    units="Random",
                ),
            ).to_dataset(name=name)

            data.latitude.attrs.update({"units": "degrees_north", "long_name": "latitude"})
            data.longitude.attrs.update({"units": "degrees_east", "long_name": "longitude"})

            data.time.encoding.update({"units": "hours since 1900-01-01 00:00:00.0", "calendar": "gregorian"})

            data.attrs.update(
                {
                    # 'Conventions': 'CF-1.6',
                    "summary": "Random Fake Data generated for testing purposes only",
                    # 'title': '`edit.data` Fake Data'
                }
            )
            return data

        if isinstance(self.variable, list):
            return xr.merge(tuple(map(make_data, self.variable)))
        return make_data(self.variable)
