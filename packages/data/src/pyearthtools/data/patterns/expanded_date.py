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
Generate FilePath Structure based upon expanded date pattern

Pattern:     {ROOT_DIR}/{FILE_DATE}/{prefix}{date_info}T{time_info}{extension}
e.g.:        data/is/here/2020/01/01/prefix_20200101T0000.nc
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pyearthtools.data import pyearthtoolsDatetime, TimeResolution
from pyearthtools.data.patterns.default import (
    PatternIndex,
    PatternTimeIndex,
    PatternForecastIndex,
    PatternVariableAware,
)
from pyearthtools.data.indexes import decorators

import pyearthtools.utils
from pyearthtools.utils.decorators import classproperty


DIRECTORY_PATTERN = "{ROOT_DIR}/{FILE_DATE}/{FILE}"
FILE_PATTERN = "{prefix}{time}{extension}"
DEFAULT_EXTENSION = pyearthtools.utils.config.get("data.patterns.default_extension")


def parse_time_str(time, directory: bool = False, delimiter: str | tuple | list = "") -> str:
    """Parse time str based on if directory splitting or delimiter replacing"""
    if isinstance(delimiter, str):
        delimiter = (delimiter, "")
    elif hasattr(delimiter, "__len__") and len(delimiter) < 2:
        delimiter = [*delimiter, *([""] * (2 - len(delimiter)))]

    def nonNone(*args):
        for arg in args:
            if arg is not None:
                return arg
        return None

    return (
        str(time)
        .replace("-", "/" if directory else str(nonNone(delimiter[0], "-")))
        .replace(":", str(nonNone(delimiter[1], ":")))
        .replace("T", "/T" if directory else "T")
    )


class _ExpandedDate(PatternIndex):
    @decorators.alias_arguments(delimiter=["deliminator"])
    def __init__(
        self,
        root_dir: str | Path,
        *,
        extension: str = DEFAULT_EXTENSION,
        prefix: None | str = None,
        delimiter: str | list[str | None] | tuple[str | None] = "",
        file_resolution: str | TimeResolution = "minute",
        directory_resolution: str | TimeResolution = "day",
        **kwargs,
    ):
        """
        Expanded Date based DataIndex

        Args:
            root_dir (str | Path):
                Root Path to use
            extension (str, optional):
                File extension to load. Defaults to 'data.patterns.default_extension'
            prefix (None | str, optional):
                File prefix to add. Defaults to None.
            delimiter (str | list[str | None] | tuple[str | None], optional):
                str/s to seperate time values with. If iterable,
                First element used to replace '-' in date, and second ':' in time'.
                Can set either element to None to not replace.  Defaults to ""
            file_resolution (str | TimeResolution, optional):
                Resolution of the files. Defaults to 'minute'.
            directory_resolution (str | TimeResolution, optional):
                Resolution of directories. Defaults to 'day'.
            kwargs (Any, optional):
                Kwargs passed to PatternIndex
        """

        super().__init__(root_dir=root_dir, **kwargs, add_default_transforms="nc" in extension)
        self.record_initialisation()

        if file_resolution is None:
            file_resolution = "minute"

        self.delimiter = delimiter

        self.file_resolution = TimeResolution(file_resolution)
        self.directory_resolution = TimeResolution(directory_resolution)

        self.extension = str(f".{extension.removeprefix('.')}")
        self.prefix = prefix

    def filesystem(
        self,
        basetime: str | pyearthtoolsDatetime,
    ) -> Path:
        basetime = pyearthtoolsDatetime(basetime).at_resolution(self.file_resolution)
        folder_datetime = pyearthtoolsDatetime(basetime).at_resolution(self.directory_resolution)

        basepath = Path(self.root_dir).resolve() / parse_time_str(
            folder_datetime, directory=True, delimiter=self.delimiter
        )

        basepath /= FILE_PATTERN.format(
            prefix=self.prefix if self.prefix else "",
            time=parse_time_str(basetime, delimiter=self.delimiter),
            extension=f".{self.extension.removeprefix('.')}",
        )

        return basepath


class ExpandedDate(_ExpandedDate):
    """Generate FilePath Structure based upon expanded date pattern

    Examples:
        >>> pattern = pyearthtools.data.patterns.ExpandedDate('/dir/', extension = '.nc')
        >>> str(pattern.search('2020-01-02T0030'))
        '/dir/2020/01/02/20200102T0030.nc'
        >>> pattern = pyearthtools.data.patterns.ExpandedDate('/dir/', extension = '.nc', deliminator = ('#', None))
        >>> str(pattern.search('2020-01-02T0030'))
        '/dir/2020/01/02/2020#01#02T00:30.nc'
    """

    def to_temporal(self, data_interval: tuple[int, str] | int | str) -> TemporalExpandedDate:
        """Get pattern as `TemporalExpandedDate`"""
        return TemporalExpandedDate(data_interval=data_interval, **self.initialisation)

    @classproperty
    def factory(self):
        return ExpandedDateFactory


class TemporalExpandedDate(_ExpandedDate, PatternTimeIndex):
    """ExpandedDate PatternIndex which is also a AdvancedTimeIndex

    Will create its path using the `data_interval` if set.

    If using this with data saved using `ExpandedDate`, set `data_interval` to (1, 'min'), the paths will match.

    Examples:
        >>> pattern = pyearthtools.data.patterns.TemporalExpandedDate('/dir/', extension = '.nc', data_interval = (1, 'month'))
        >>> str(pattern.search('2020-01-02'))
        '/dir/2020/01/202001.nc'
        >>> str(pattern.search('2020-01'))
        '/dir/2020/01/202001.nc'

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.data_resolution and "file_resolution" not in kwargs:
            self.file_resolution = self.data_resolution

    def filesystem(
        self,
        basetime: str | pyearthtoolsDatetime,
    ) -> Path:
        basetime = pyearthtoolsDatetime(basetime)

        basetime = basetime.at_resolution(self.file_resolution)
        folder_datetime = pyearthtoolsDatetime(basetime).at_resolution(self.directory_resolution)

        basepath = Path(self.root_dir).resolve() / parse_time_str(folder_datetime, directory=True)
        basepath /= FILE_PATTERN.format(
            prefix=self.prefix + "_" if self.prefix else "",
            time=parse_time_str(basetime),
            extension=f".{self.extension.removeprefix('.')}",
        )

        return basepath


class ForecastExpandedDate(_ExpandedDate, PatternForecastIndex):
    """ExpandedDate PatternIndex which is also a ForecastIndex"""

    pass


class _ExpandedDataVariable(PatternVariableAware):
    @property
    def default_variable_parse(self) -> str:
        return "root_dir"


class ExpandedDateVariable(_ExpandedDataVariable, _ExpandedDate):
    """
    ExpandedDate pattern which is variable aware

    Will split each variable into a seperate file,
    using the variable as another layer in the root_dir

    Examples:
        >>> expanded_var = ExpandedDateVariable(root_dir = '/test/', variables = 'variable', extension = 'nc')
        >>> str(expanded_var.search('2020-01-02'))
        {'variable' : '/test/variable/2020/01/02/20200102T0000.nc'}

        >>> expanded_var = ExpandedDateVariable(root_dir = '/test/', variables = 'variable', extension = 'nc', data_interval = (1,'year'), variable_parse = 'prefix')
        >>> str(expanded_var.search('2020-01'))
        {'variable' : '/test/2020/01/02/variable_20200102T0000.nc'}
    """

    @property
    def root_pattern(self) -> type[ExpandedDate]:
        return ExpandedDate


class ForecastExpandedDateVariable(_ExpandedDataVariable, ForecastExpandedDate):
    """
    ForecastExpandedDate pattern which is variable aware and retrieves Forecasts

    Will split each variable into a separate file,
    using the variable as another layer in the root_dir

    Examples:
        >>> expanded_var = ForecastExpandedDateVariable(root_dir = '/test/', variables = 'variable', extension = 'nc')
        >>> str(expanded_var.search('2020-01-02'))
        {'variable' : '/test/variable/2020/01/02/20200102T0000.nc'}

        >>> expanded_var = ForecastExpandedDateVariable(root_dir = '/test/', variables = 'variable', extension = 'nc', data_interval = (1,'year'), variable_parse = 'prefix')
        >>> str(expanded_var.search('2020-01'))
        {'variable' : '/test/2020/01/02/variable_20200102T0000.nc'}
    """

    @property
    def root_pattern(self) -> type[ForecastExpandedDate]:
        return ForecastExpandedDate


class TemporalExpandedDateVariable(_ExpandedDataVariable, TemporalExpandedDate):
    """
    TemporalExpandedDate pattern which is variable aware

    Will split each variable into a seperate file,
    using the variable as another layer in the root_dir

    Examples:
        >>> expanded_var = TemporalExpandedDateVariable(root_dir = '/test/', variables = 'variable', extension = 'nc', data_interval = (1,'year'))
        >>> str(expanded_var.search('2020-01'))
        {'variable' : '/test/variable/2020/2020.nc'}

        >>> expanded_var = TemporalExpandedDateVariable(root_dir = '/test/', variables = 'variable', extension = 'nc', data_interval = (1,'year'), variable_parse = 'prefix')
        >>> str(expanded_var.search('2020-01'))
        {'variable' : '/test/2020/variable_2020.nc'}
    """

    @property
    def root_pattern(self) -> type[TemporalExpandedDate]:
        return TemporalExpandedDate


def ExpandedDateFactory(
    *args: Any,
    temporal: bool = False,
    variable: bool = False,
    forecast: bool = False,
    **kwargs,
) -> _ExpandedDate:
    """Create an ExpandedDate pattern based on the requirements

    Args:
        temporal (bool, optional):
            Temporally aware, exclusive with `forecast`, allows for `.series` operations. Defaults to False.
        variable (bool, optional):
            Variable aware, splits variables when loading and saving. Defaults to False.
        forecast (bool, optional):
            Forecast product, exclusive with `temporal`, provides `.series` but with forecasts. Defaults to False.

    Raises:
        ValueError:
            If both `temporal` and `forecast` set. Cannot be both.

    Returns:
        (_ExpandedDate):
            Created `_ExpandedDate` pattern.
    """
    if forecast and temporal:
        raise ValueError("A pattern cannot be both temporally aware and a forecast product, must be either, not both.")

    cls = ExpandedDate
    if not any((temporal, forecast, variable)):
        return ExpandedDate(*args, **kwargs)

    elif temporal and not any((forecast, variable)):
        cls = TemporalExpandedDate
    elif temporal and variable:
        cls = TemporalExpandedDateVariable

    elif forecast and not any((temporal, variable)):
        cls = ForecastExpandedDate
    elif forecast and variable:
        cls = ForecastExpandedDateVariable

    elif variable and not any((temporal, forecast)):
        cls = ExpandedDateVariable
    elif variable and forecast:
        cls = ForecastExpandedDateVariable
    elif variable and temporal:
        cls = TemporalExpandedDateVariable

    return cls(*args, **kwargs)


__all__ = [
    "ExpandedDate",
    "TemporalExpandedDate",
    "ForecastExpandedDate",
    "ExpandedDateVariable",
    "ForecastExpandedDateVariable",
    "TemporalExpandedDateVariable",
    "ExpandedDateFactory",
]
