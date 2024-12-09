# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
Generate FilePath Structure based upon direct date pattern

Pattern:      {ROOT_DIR}/{prefix}{date_info}T{time_info}{extension}
e.g.:         data/is/here/prefix_20200101T0000.nc

"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pyearthtools.data.time import pyearthtoolsDatetime, TimeResolution
from pyearthtools.data.patterns import (
    PatternIndex,
    PatternTimeIndex,
    PatternVariableAware,
    PatternForecastIndex,
)
from pyearthtools.data.indexes import TimeIndex, decorators

import pyearthtools.utils
from pyearthtools.utils.decorators import classproperty

DIRECTORY_PATTERN = "{ROOT_DIR}/{FILE}"
FILE_PATTERN = "{prefix}{time}{extension}"
DEFAULT_EXTENSION = pyearthtools.utils.config.get("data.patterns.default_extension")


def nonNone(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None


class _Direct(TimeIndex, PatternIndex):
    @decorators.alias_arguments(delimiter=["deliminator"])
    def __init__(
        self,
        root_dir: str | Path,
        *,
        extension: str = DEFAULT_EXTENSION,
        prefix: None | str = None,
        file_resolution: str | TimeResolution = "minute",
        delimiter: str | tuple[str | None] | list[str | None] = "",
        **kwargs,
    ):
        """
        Direct time based DataIndexer.

        Args:
            root_dir (str | Path):
                Root Path to use
            extension (str, optional):
                File extension to load. Defaults to 'data.patterns.default_extension'
            prefix (None | str, optional):
                File prefix to add. Defaults to None.
            file_resolution (str | TimeResolution, optional):
                Temporal resolution of the file name. Defaults to 'minute'.
            delimiter (str | tuple[str | None] | list[str | None], optional):
                str/s to seperate time values with. If iterable,
                First element used to replace '-' in date, and second ':' in time'.
                Can set either element to None to not replace.  Defaults to ""
            kwargs (Any, optional):
                Kwargs passed to PatternIndex

        """
        super().__init__(root_dir=root_dir, **kwargs, add_default_transforms="nc" in extension)

        self.record_initialisation()
        self.extension = str(f".{extension.removeprefix('.')}")
        self.prefix = prefix
        self.file_resolution = self.data_resolution or TimeResolution(file_resolution)

        if isinstance(delimiter, str):
            delimiter = (delimiter, "")  # type: ignore
        elif hasattr(delimiter, "__len__") and len(delimiter) < 2:
            delimiter = [*delimiter, *([""] * (2 - len(delimiter)))]

        self.delimiter = delimiter

    # self.pattern_func = self.search

    def filesystem(
        self,
        basetime: str | pyearthtoolsDatetime,
    ) -> Path:
        basetime = pyearthtoolsDatetime(basetime).at_resolution(self.file_resolution)

        basepath = Path(self.root_dir).resolve() / FILE_PATTERN.format(
            prefix=self.prefix + "_" if self.prefix else "",
            time=f"{str(basetime).replace('-', str(nonNone(self.delimiter[0], ''))).replace(':',str(nonNone(self.delimiter[1], '')))}",  # type: ignore
            extension=f".{self.extension.removeprefix('.')}",
        )

        return basepath


class Direct(_Direct):
    """Generate Filepath structure based on time at given root directory

    Examples:
        >>> pattern = pyearthtools.data.patterns.Direct('/dir/', extension = '.nc')
        >>> str(pattern.search('2020-01-02T0030'))
        '/dir/20200102T0030.nc'
        >>> pattern = pyearthtools.data.patterns.Direct('/dir/', extension = '.nc', deliminator = ('@', '%'))
        >>> str(pattern.search('2020-01-02T0030'))
        '/dir/2020@01@02T00%30.nc'
    """

    def to_temporal(self, data_interval: tuple[int, str] | int) -> TemporalDirect:
        """Get pattern as `TemporalDirect`"""
        return TemporalDirect(data_interval=data_interval, **self.initialisation)

    @classproperty
    def factory(self):
        return DirectFactory


class TemporalDirect(_Direct, PatternTimeIndex):
    """Direct PatternIndex which is also a AdvancedTimeIndex

    Examples:
        >>> pattern = pyearthtools.data.patterns.TemporalDirect('/dir/', extension = '.nc', data_interval = (1, 'month'))
        >>> str(pattern.search('2020-01-02'))
        '/dir/202001.nc'

    """

    def filesystem(
        self,
        basetime: str | pyearthtoolsDatetime,
    ) -> Path:
        basetime = pyearthtoolsDatetime(basetime)

        if self.data_resolution:
            basetime = basetime.at_resolution(self.data_resolution)

        return super().filesystem(basetime)


class ForecastDirect(_Direct, PatternForecastIndex):
    """Direct PatternIndex which is also a ForecastIndex"""

    pass


class DirectVariable(PatternVariableAware, _Direct):
    """
    Direct pattern which is variable aware

    Will split each variable into a seperate file,
    using the variable as the prefix

    Examples:
        >>> direct_var = DirectVariable(root_dir = '/test/', variables = 'variable', extension = 'nc')
        >>> str(direct_var.search('2021-01'))
        {'variable' : '/test/variable_202101.nc'}

        >>> direct_var = DirectVariable(root_dir = '/test/', variables = 'variable', extension = 'nc', variable_parse = 'root_dir')
        >>> str(direct_var.search('2021-01-01'))
        {'variable' : '/test/variable/202101.nc'}
    """

    @property
    def default_variable_parse(self) -> str:
        return "root_dir"

    @property
    def root_pattern(self) -> type[Direct]:
        return Direct


class ForecastDirectVariable(PatternVariableAware, ForecastDirect):
    """
    Direct pattern which is variable aware and retrieves Forecasts

    Will split each variable into a seperate file,
    using the variable as the prefix

    Examples:
        >>> direct_var = ForecastDirectVariable(root_dir = '/test/', variables = 'variable', extension = 'nc')
        >>> str(direct_var.search('2021-01'))
        {'variable' : '/test/variable_202101.nc'}

        >>> direct_var = ForecastDirectVariable(root_dir = '/test/', variables = 'variable', extension = 'nc', variable_parse = 'root_dir')
        >>> str(direct_var.search('2021-01-01'))
        {'variable' : '/test/variable/202101.nc'}
    """

    @property
    def root_pattern(self) -> type[ForecastDirect]:
        return ForecastDirect


class TemporalDirectVariable(PatternVariableAware, TemporalDirect):
    """
    TemporalDirect pattern which is variable aware

    Will split each variable into a seperate file,
    using the variable as the prefix.

    Examples:
        >>> direct_var = TemporalDirectVariable(root_dir = '/test/', variables = 'variable', extension = 'nc', data_interval = (1,'year'))
        >>> direct_var.search('2021-01-01')
        {'variable' : '/test/variable/2021.nc'}

        >>> direct_var = TemporalDirectVariable(root_dir = '/test/', variables = 'variable', extension = 'nc', data_interval = (1,'year'), variable_parse = 'root_dir')
        >>> direct_var.search('2021-01-01')
        {'variable' : '/test/variable/2021.nc'}
    """

    @property
    def root_pattern(self) -> type[TemporalDirect]:
        return TemporalDirect


def DirectFactory(
    *args: Any,
    temporal: bool = False,
    variable: bool = False,
    forecast: bool = False,
    **kwargs,
) -> _Direct:
    """Create an Direct pattern based on the requirements

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
        (_Direct):
            Created `_Direct` pattern.
    """
    if forecast and temporal:
        raise ValueError("A pattern cannot be both temporally aware and a forecast product, must be either, not both.")

    cls = Direct
    if not any((temporal, forecast, variable)):
        return Direct(*args, **kwargs)

    elif temporal and not any((forecast, variable)):
        cls = TemporalDirect
    elif temporal and variable:
        cls = TemporalDirectVariable

    elif forecast and not any((temporal, variable)):
        cls = ForecastDirect
    elif forecast and variable:
        cls = ForecastDirectVariable

    elif variable and not any((temporal, forecast)):
        cls = DirectVariable
    elif variable and forecast:
        cls = ForecastDirectVariable
    elif variable and temporal:
        cls = TemporalDirectVariable

    return cls(*args, **kwargs)


__all__ = [
    "Direct",
    "TemporalDirect",
    "ForecastDirect",
    "DirectVariable",
    "ForecastDirectVariable",
    "TemporalDirectVariable",
    "DirectFactory",
]
