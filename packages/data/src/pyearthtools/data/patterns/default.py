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
from abc import abstractmethod

import os
from pathlib import Path
from typing import Any, Callable
import warnings

import xarray as xr

from pyearthtools.data import patterns
from pyearthtools.data.exceptions import DataNotFoundError
from pyearthtools.data.warnings import pyearthtoolsDataWarning
from pyearthtools.data.indexes import (
    DataIndex,
    FileSystemIndex,
    AdvancedTimeIndex,
    ForecastIndex,
    decorators,
)
from pyearthtools.data.save import save


class PatternIndex(DataIndex, FileSystemIndex):
    def __init__(self, *args, root_dir: str | Path, **kwargs):
        super().__init__(*args, **kwargs)

        root_dir, temp_dir = patterns.utils.parse_root_dir(root_dir)
        self.root_dir, self.temp_dir = root_dir, temp_dir

        self.update_initialisation(root_dir=root_dir)

    @staticmethod
    def from_pattern(pattern_function: Callable | str, *args, **kwargs) -> "PatternIndex":
        """
        Create Pattern Index from given pattern name

        Args:
            *args (Any): Passed to discovered pattern
            pattern_function (Callable | str): Either the function to use, or the pattern name within pyearthtools.data.patterns
            *kwargs (Any): Passed to discovered pattern

        Raises:
            KeyError: If pattern not found
            TypeError: If not callable

        Returns:
            PatternIndex: Loaded Pattern Index
        """
        if isinstance(pattern_function, str):
            if not hasattr(patterns, pattern_function):
                raise KeyError(f"Pattern: '{pattern_function} not found. Must be in pyearthtools.data.patterns")
            pattern_class = getattr(patterns, pattern_function)
        elif isinstance(pattern_function, Callable):
            pattern_class = pattern_function
        else:
            raise TypeError(
                f"'pattern_function' must be either callable or name of function found within pyearthtools.data.patterns, not {type(patterns)}"
            )
        return pattern_class(*args, **kwargs)

    def save(self, data: Any, *args, **kwargs):
        """
        Save data using this pattern to find where to save

        Args:
            data (Any):
                Data to save
            *args (Any, optional):
                Arguments to pass to `search` to find filepath
            *kwargs (Any, optional):
                Keyword arguments to pass to `search` to find filepath
        """
        save(data, self, *args, **kwargs)

    def cleanup(self, safe: bool = False):
        """Clean up temp_dir if it exists.

        If not safe and not `temp_dir` raise AttributeError
        """
        if hasattr(self, "temp_dir") and self.temp_dir is not None:
            self.temp_dir.cleanup()
        elif not safe:
            raise AttributeError(f"{self.__class__.__name__} has no temporary directory assigned.")

    def get_root_dir(self) -> str | Path:
        """Get root dir if set.

        Raises:
            RuntimeError:
                If `root_dir` not set

        Returns:
            (str | Path):
                Set `root_dir`
        """
        if hasattr(self, "root_dir"):
            return self.root_dir
        else:
            raise RuntimeError(f"{self.__class__.__name__} has no 'root_dir' set. ")

    def __del__(self):
        self.cleanup(safe=True)


class PatternTimeIndex(AdvancedTimeIndex, PatternIndex):
    """Temporal Pattern Index

    Used for when a pattern can advanced time indexing, like [series][pyearthtools.data.AdvancedTimeIndex.series]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PatternForecastIndex(ForecastIndex, PatternIndex):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    pass


class PatternVariableAware(PatternIndex):
    """
    Base Pattern class for patterns which are variable aware.

    That means, any dataset passed to be saved will be saved in individual variables,
    and files can be loaded from different variables.

    A child class must implement `root_pattern`, this informs this class which pattern to use when
    constructing a new PatternIndex for each variable. Using `variable_parse` allows a user
    to specify which arguments the variable is added to.

    A child class pattern can set a default `variable_parse` by setting the `default_variable_parse` property.

    Examples:
        Say a pattern is initalised as,
        `ExpandedDateVariable(root_dir = 'test', prefix = 'prefix_1')`

        If `variable_parse` was set to `root_dir`, any variable being requested will be added to the end of `root_dir`.
        This new `root_dir` = `test/VARIABLE` will be used to create a new pattern soley used for that variable,
        `ExpandedDateVariable(root_dir = 'test/VARIABLE', prefix = 'prefix_1')`


    """

    @decorators.alias_arguments(variables=["var", "variable"])
    @decorators.variable_modifications("variables", skip_if_invalid_class=True)
    def __init__(
        self,
        variables: str | list[str] | None = None,
        *args,
        variable_parse: str | list[str] | None = None,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Construct a variable aware pattern.

        Args:
            variables (str | list[str], optional):
                Variables to find by default.
                When saving, variables will be appended to this list.
                If not given, can't really be used to load data, but is useful for saving data.
                Defaults to []
            variable_parse (str | list[str], optional):
                Initalisation argument/s to add variable to when constructing the new pattern.
                Defaults to 'prefix' if `default_variable_parse` not set.
            verbose (bool, optional) :
                Whether to warn the user if a variable is saved and not already given. Defaults to True.

        !!! Note
            `variable_parse` can be used to reference many different types, and the following
            table details the behaviour.
            | Type | Behaviour |
            | ---- | --------- |
            | Path | Added to the end as directory layer |
            | str  | Attempt to parse to Path, or just append |
            | None | Replaced |
            | list | Appended |

        """
        if variables is None:
            variables = []

        super().__init__(*args, **kwargs)
        self.record_initialisation()

        variable_parse = variable_parse or getattr(self, "default_variable_parse", "root_dir")

        variables = [variables] if isinstance(variables, str) else variables
        self.verbose = verbose

        if len(variables) == 0 and verbose:
            warnings.warn(
                f"Given variables are empty, will be unable to retrieve data from {self.root_dir!r}.",
                pyearthtoolsDataWarning,
            )

        variable_parse = [variable_parse] if isinstance(variable_parse, str) else variable_parse

        self.variables = variables
        self.variable_parse = variable_parse

    @property
    @abstractmethod
    def root_pattern(self) -> PatternIndex:
        """Get pattern for finding/saving a specific variable

        !!! Note
            Must be implemented by the child class

        Returns:
            (PatternIndex):
                Uninitalised pattern to use to find location of variable
        """

        raise NotImplementedError(
            "Pattern has not correctly implemented `root_pattern`. This is needed for Variable awareness."
        )

    def variable_pattern(self, variable: str) -> PatternIndex:
        """
        Using the given `variable` and the `root_pattern`, parse `variable_parse`
        so that the variable is added correctly to init arguments to construct a new pattern
        specific to that variable.

        Args:
            variable (str):
                Variable to make pattern for

        Raises:
            TypeError:
                If cannot add variable to init argument
            KeyError:
                If variable parse not in init_kwargs

        Returns:
            (PatternIndex):
                Initialised pattern to use for the parsed `variable`
        """
        root_pattern = self.root_pattern

        initialisation = dict(self.initialisation)

        init_args = list(initialisation.pop("__args", []))
        init_kwargs = dict(initialisation)
        init_kwargs["root_dir"] = self.get_root_dir()

        init_kwargs.pop("variables", None)
        init_kwargs.pop("variable_parse", None)
        init_kwargs.pop("verbose", None)

        def parse_init(init_value: Path | str | list | None, variable: str) -> Path | str | list:
            if isinstance(init_value, Path):
                init_value = init_value / variable
            elif isinstance(init_value, str):
                try:
                    if os.sep not in init_value:
                        raise TypeError()
                    init_value = Path(init_value) / variable
                except (TypeError, AttributeError):
                    init_value = init_value + str(variable)
            elif isinstance(init_value, list):
                init_value.append(variable)
            elif init_value is None:
                return parse_init("", variable)
            else:
                raise TypeError(f"Cannot parse initial value of type {type(init_value)}")
            return init_value

        for parse in self.variable_parse or []:
            if parse not in init_kwargs:
                raise KeyError(f"Cannot add variable to {parse} as it isn't in the init_kwargs. {init_kwargs}")
            init_kwargs[parse] = parse_init(init_kwargs[parse], variable)

        return root_pattern(*init_args, **init_kwargs)

    def _update_variables(self, new_var: str | list[str], *args):
        """
        Add new variable

        Args:
            new_var (str | list[str]):
                New variable/s to add
        """
        if self.variables is None:
            self.variables = []

        variables = list(self.variables)

        if isinstance(new_var, list):
            variables.extend(new_var)
        else:
            variables.append(new_var)

        variables.extend(args)

        self.update_initialisation(variables=variables)
        self.variables = variables

    def save(self, data: xr.Dataset, *save_args, **save_kwargs):
        """
        Save a [dataset][xarray.Dataset] splitting it by variable.

        Extra arguments are used in a `search` call to find path to save data at.

        Args:
            data (xr.Dataset):
                Data to save
            *save_args (Any, optional):
                Arguments to pass to underlying pattern save
            **save_kwargs (Any, optional):
                Keyword arguments to pass to underlying pattern save
        Raises:
            TypeError:
                If data is not a [dataset][xarray.Dataset]
        """
        # if hasattr(self, "temp_dir") and self.temp_dir is not None:
        # warnings.warn("Cannot save variables inside temp directory.", UserWarning)

        if isinstance(data, xr.Dataset):
            for variable in data.data_vars:
                if variable not in self.variables:
                    if self.verbose:
                        warnings.warn(
                            f"{variable!r} was not included in `init` variables, but was in the dataset. Appending it to variables.",
                            pyearthtoolsDataWarning,
                        )
                    self._update_variables(str(variable))

            super().save(data, *save_args, **save_kwargs)
        else:
            raise TypeError(f"Must be `xarray.Dataset` to save with variable awareness, not {type(data)}.")

    def filesystem(self, *args, variables: str | list[str] | None = None, **kwargs) -> dict:
        """
        Find paths on disk for all variables given the arguments

        Args:
            *args (Any, optional):
                Arguments to pass to underlying pattern `filesystem`
            variables (list[str] | str, optional):
                Extra variables to add to find. Defaults to None
            **kwargs (Any, optional):
                Keyword arguments to pass to underlying pattern `filesystem`
        Returns:
            (dict):
                Dictionary of paths to each variable
                {variable: PathToVariable}
        """
        paths = {}

        if variables is None:
            variables = self.variables
        else:
            variables = self.variables
            _ = tuple(
                variables.append(x) for x in (variables if isinstance(variables, (list, tuple)) else (variables,))
            )

        if variables == [] or variables is None:
            raise DataNotFoundError(
                "No variables given, therefore cannot find data. " "\nOverride '.variables' with a list of variables."
            )

        for variable in variables:
            pattern = self.variable_pattern(variable)
            paths[variable] = pattern.filesystem(*args, **kwargs)
        return paths
