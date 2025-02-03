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


from __future__ import annotations

from typing import Literal, Any
import warnings
import xarray as xr

import pyearthtools.data
from pyearthtools.data.indexes import (
    alias_arguments,
    check_arguments,
    VariableDefault,
    VARIABLE_DEFAULT,
)

from pyearthtools.data.download.ecmwf_opendata import opendata_variables
from pyearthtools.data.download import DownloadIndex


VALID_MODELS = Literal["aifs", "ifs"]
VALID_STREAM = Literal["oper", "enfo", "waef", "wave"]
VALID_TYPES = Literal[
    "fc",
    "ef",
    "ep",
]

VARIABLES = [*opendata_variables.single, *opendata_variables.pressure, "all"]

RENAME_DICT = {"t2m": "2t", "u10": "10u", "v10": "10v"}


def rounder(time: pyearthtools.data.pyearthtoolsDatetime, round_value: int = 6) -> int:
    return (time.hour // round_value) * round_value


def split_variables(variables: list[str]) -> tuple[list[str], list[str]]:
    single = []
    pressure = []

    for var in variables:
        if var in opendata_variables.single:
            single.append(var)
        elif var in opendata_variables.pressure:
            pressure.append(var)

    return single, pressure


class OpenData(DownloadIndex):
    """
    ECMWF Data Store Index

    Implements the `DownloadIndex` to automatically cache the data as downloaded.

    ## Usage
    AIFS latest download
    >>> import pyearthtools.data
    >>> opendata_index = pyearthtools.data.download.opendata.AIFS('msl', step = (0, 120, 6), cache = '~/AIFS_Data/')
    >>> opendata_index.latest()

    IFS latest download, subsetting on level

    >>> import pyearthtools.data
    >>> opendata_index = pyearthtools.data.download.opendata.IFS(['q', 't'], stream = 'oper', step = 120, levels = [1000], cache = '~/IFS_Data/')
    >>> opendata_index.latest()

    ## Overriding
    If data has been previously downloaded for a set date, but the levels or steps differ on disk to what is requested, data may not be downloaded
    To fix this, simply use the `.override` context manager provided.
    ```python
    with opendata_index.override:
        opendata_index.latest()
    ```
    """

    _client = None

    @property
    def _desc_(self):
        return {
            "singleline": "ECMWF Data Store ",
            "Source": "https://data.ecmwf.int/forecasts/",
            "Documentation": "https://www.ecmwf.int/en/forecasts/datasets/open-data",
        }

    @check_arguments("pyearthtools.data.download.ecmwf_opendata.opendata.struc", variables=VARIABLES)
    def __init__(
        self,
        variables: list[str] | str,
        model: VALID_MODELS,
        *,
        step: int | list[int] | tuple[int, ...] | None = None,
        levels: int | list[int] | None = None,
        source: str = "ecmwf",
        resolution: str = "0p25",
        stream: VALID_STREAM | VARIABLE_DEFAULT = VariableDefault,
        type: VALID_TYPES | VARIABLE_DEFAULT = VariableDefault,
        number: int | None = None,
        **kwargs,
    ):
        """
        DataIndex for access to ECMWF's Opendata store

        Args:
            variables (list[str] | str):
                Variables to retrieve, will auto split on single and pressure
            model (VALID_MODELS):
                Model to retrieve data from, either IFS or AIFS
            step (int | list[int] | tuple[int, ...] | None, optional):
                Forecast step to retrieve, can be None for all, an int for one,
                a list for selection, or a tuple to be passed to range to make a selection.
                Defaults to None.
            levels (int | list[int] | None, optional):
                Levels to select pressure variables at, If None, will be all. Defaults to None.
            resolution (str, optional):
                Resolution to retrieve data at. Defaults to "0p25".
            stream (VALID_STREAM | VARIABLE_DEFAULT, optional):
                Stream of data to retrieve. Defaults to VariableDefault.
            source (str, optional):
                Source of data, can be `ecmwf` or `azure`. Defaults to "ecmwf".
            type (VALID_TYPES | VARIABLE_DEFAULT, optional):
                Type of data to retrieve. Defaults to VariableDefault.
            number (int | None, optional):
                Ensemble member number. Defaults to None.
        """

        if isinstance(step, tuple):
            step = list(range(*step))

        if step is None:
            step = list(range(0, 360, 6))

        try:
            from ecmwf.opendata import Client

            self._client = Client(source=source, model=model, resol=resolution)

        except Exception as e:
            warnings.warn(
                f"Setting up ecmwf.opendata raised the following error: {e}. \nWill not be able to download data.",
                RuntimeWarning,
            )

        if variables == "all":
            variables = [*opendata_variables.single, *opendata_variables.pressure]

        self._variables = [variables] if not isinstance(variables, (tuple, list)) else variables
        self._levels = levels

        self._request_base: dict[str, Any] = dict(type=type, stream=stream, step=step, number=number)

        transforms = kwargs.pop("transforms", pyearthtools.data.TransformCollection())
        # Rename variables to match other indexes, and trim out any not requested
        self.download_transforms = pyearthtools.data.transforms.variables.rename_variables(
            **RENAME_DICT  # type: ignore
        ) + pyearthtools.data.transforms.variables.variable_trim(
            *variables
        )  # + pyearthtools.data.transforms.coordinates.drop("meanSea", "valid_time", "heightAboveGround", "entireAtmosphere", ignore_missing=True)

        if levels:
            transforms += pyearthtools.data.transforms.coordinates.select(level=levels, ignore_missing=True)

        pattern_kwargs = kwargs.pop("pattern_kwargs", {})
        pattern_kwargs.update(variables=self._variables)
        super().__init__(
            pattern_kwargs=pattern_kwargs,
            transforms=transforms,
            pattern=kwargs.pop("pattern", "ForecastExpandedDateVariable"),
            **kwargs,
        )
        self.record_initialisation()

    def _get_latest(self) -> str:
        """Get latest datetime from `ecmwf-opendata`"""
        if self._client is None:
            raise ImportError(f"`ecwmwf.opendata` was not imported, cannot download new data.")
        return str(self._client.latest(self._request_base))

    def latest(self) -> xr.Dataset:
        """Get the latest data from `ecwmf-opendata`"""
        return self(self._get_latest())

    def download(self, querytime: str | pyearthtools.data.pyearthtoolsDatetime) -> xr.Dataset:
        """Download data from `ecwmf-opendata`"""
        if self._client is None:
            raise ImportError(f"`ecwmwf.opendata` was not imported, cannot download new data.")

        if querytime == "l1atest":
            querytime = self._get_latest()

        date = pyearthtools.data.pyearthtoolsDatetime(querytime).at_resolution("day")
        time = rounder(pyearthtools.data.pyearthtoolsDatetime(querytime))

        request = dict(self._request_base)
        path = self.get_tempdir()

        request.update(
            date=str(date),
            time=time,
        )

        def filter_if_none(dictionary):
            for key in list(dictionary.keys()):
                if dictionary[key] is None:
                    dictionary.pop(key)
            return dictionary

        try:
            single_variables, pressure_variables = split_variables(self._variables)
            saved_paths = []

            if single_variables:
                single_path = path / "single_download.grib"
                single_dict = dict(request)
                single_dict.update(target=str(single_path), param=single_variables)

                single_dict = filter_if_none(single_dict)

                self._client.retrieve(single_dict)

                xr.open_dataset(single_path).to_netcdf(path / "single_download.nc")
                saved_paths.append(single_path.parent / "single_download.nc")

            if pressure_variables:
                pressure_path = path / "pressure_download.grib"
                pressure_dict = dict(request)
                pressure_dict.update(target=str(pressure_path), levelist=self._levels)

                pressure_dict = filter_if_none(pressure_dict)

                self._client.retrieve(pressure_dict)

                xr.open_dataset(pressure_path, filter_by_keys={"typeOfLevel": "isobaricInhPa"}).to_netcdf(
                    path / "pressure_download.nc"
                )
                saved_paths.append(pressure_path.parent / "pressure_download.nc")

        except Exception as e:
            raise pyearthtools.data.DataNotFoundError("Could not download data from ECMWF Data Store.") from e

        return self.download_transforms(xr.open_mfdataset(saved_paths))
