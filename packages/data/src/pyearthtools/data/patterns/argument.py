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

from pathlib import Path
from typing import Any, Iterable

from pyearthtools.data.patterns import PatternIndex, PatternVariableAware
from pyearthtools.data.indexes import decorators
from pyearthtools.data.indexes.utilities import spellcheck

import pyearthtools.utils
from pyearthtools.utils.decorators import classproperty

"""
Generate FilePath Structure based upon expansion of arguments
"""

DEFAULT_EXTENSION = pyearthtools.utils.config.get("data.patterns.default_extension")


def flattened_combinations(iterable: Iterable[Any | Iterable[Any]], r: int = 1) -> list[Any | list[Any]]:
    """
    Given an input iterable, flatten any items which themselves are a list or tuple, providing all combinations.

    If no elements are iterable, will early return.

    Args:
        iterable (Iterable[Any | list[Any] | tuple[Any]]):
            Iterable containing items of list, tuple or other
        r (int, optional):
            Degrees to descend in the Iterable, if any left.

    Returns:
        (list[Any | list[Any]]):
            List containing all combinations of flattened input.


    Examples:
    >>> flattened_combinations([1,2,[3,4,5]])
    [[1,2,3],[1,2,4],[1,2,5]]
    """

    if r <= 0 or not any([isinstance(x, (list, tuple)) for x in iterable]):
        return [iterable]

    combinations = []
    for i, value in enumerate(iterable):
        if isinstance(value, (list, tuple)):
            for sub_value in value:
                local_copy = list(iterable)
                local_copy[i] = sub_value
                combinations.extend(flattened_combinations(local_copy, r=r - 1))
    return combinations


class _Argument(PatternIndex):
    """
    Generate FilePath Structure based upon expansion of arguments

    If `filename_as_arguments` is False:
        First argument specifies the FileID, and
        subsequent arguments are used to create folder path.
    Otherwise:
        Filename is made from all args, and directory is all args too.

    Examples:
        >>> pattern = pyearthtools.data.patterns.ArgumentExpansion('/dir/')
        >>> str(pattern.search('test','arg'))
        '/dir/arg/test.nc'
        >>> str(pattern.search('test','arg', 'another_arg'))
        '/dir/arg/another_arg/test.nc'
        >>> pattern = pyearthtools.data.patterns.ArgumentExpansion('/dir/', filename_as_arguments = True)
        >>> str(pattern.search('test','arg'))
        '/dir/test/arg/test_arg.nc'
        >>> pattern = pyearthtools.data.patterns.ArgumentExpansion('/dir/', expand_tuples = True)
        >>> [str(x) for x in pattern.search('test',('arg1', 'arg2'))]
        ['/dir/arg1/test.nc', '/dir/arg2/test.nc']
    """

    @decorators.alias_arguments(filename_delimiter=["filename_deliminator"])
    def __init__(
        self,
        root_dir: str | Path,
        *,
        prefix: str = "",
        extension: str = DEFAULT_EXTENSION,
        valid_arguments: list[Any] | None = None,
        filename_as_arguments: bool = False,
        filename_delimiter: str = "_",
        expand_tuples: bool | int = False,
        **kwargs,
    ):
        """
        Argument Expansion based DataIndexer.


        Args:
            root_dir (str | Path):
                Root Path to use
            prefix (str, optional):
                prefix to add.
                Defaults to "".
            extension (str, optional):
                File Extension to use. Used to determine saving and loading function.
                Defaults to DEFAULT_EXTENSION.
            valid_arguments (list[Any], optional):
                Valid arguments to limit usability to. Defaults to None.
            filename_as_arguments (bool, optional):
                Whether the filename should be constructed from all argumenets.
                E.g.
                    ArgumentExpansion('name','dir1')
                    # root_dir/name/dir1/name_dir1.extension
                If False, filename is first argument given.
                Defaults to False.
            filename_delimiter (str, optional):
                delimiter for filename if `filename_as_arguments` is True. Defaults to '_'.
            expand_tuples (bool | int, optional):
                Whether to expand tuples when given in search. If True, levels = 1.
                If `int` represents how many levels to descend in the Iterable.
                Defaults to False.


        """
        super().__init__(
            root_dir=root_dir, add_default_transforms=kwargs.pop("add_default_transforms", extension == ".nc"), **kwargs
        )
        self.record_initialisation()

        self.prefix = prefix
        self.extension = str(f".{extension.removeprefix('.')}")

        self.valid_arguments = valid_arguments
        self.filename_as_arguments = filename_as_arguments
        self._filename_delimiter = filename_delimiter
        self._expand_tuples = expand_tuples

    @staticmethod
    def parse_name(name: str) -> str:
        if name == "":
            return name
        return str(Path(name).with_suffix(""))

    def filesystem(self, *args: str) -> Path | list[Path]:
        """
        Get filepath from arguments.

        If `filename_as_arguments` is True, filename will be made from all args.
        Otherwise, filename will be first arg, with remaining making up the directory.
        """
        if len(args) == 0:
            raise ValueError(
                "Arguments must be supplied to generate the path."
                "\nFirst argument is used for ID, and the rest for folders."
            )

        if self.valid_arguments:
            for arg in args:
                spellcheck.check_prompt(arg, self.valid_arguments, name="Argument")

        basepath = Path(self.root_dir).resolve()

        def get_name(name_args: Iterable[Any]) -> Path:
            name_args = list(name_args)

            if self.filename_as_arguments:
                FileID = self.parse_name(self._filename_delimiter.join(tuple(map(str, name_args))))
            else:
                FileID = self.parse_name(str(name_args.pop(0)))

            FileID = Path(FileID + self.prefix)
            return basepath / Path(*(tuple(map(str, name_args)))) / FileID.with_suffix(self.extension)

        paths = [get_name(name) for name in flattened_combinations(args, r=int(self._expand_tuples))]

        if len(paths) == 1:
            return paths[0]
        return paths


class ArgumentExpansion(_Argument):
    """
    Generate FilePath Structure based upon expansion of arguments

    If `filename_as_arguments` is False:
        First argument specifies the FileID, and
        subsequent arguments are used to create folder path.
    Otherwise:
        Filename is made from all args, and directory is all args too.

    Examples:
        >>> pattern = pyearthtools.data.patterns.ArgumentExpansion('/dir/')
        >>> str(pattern.search('test','arg'))
        '/dir/arg/test.nc'
        >>> str(pattern.search('test','arg', 'another_arg'))
        '/dir/arg/another_arg/test.nc'
        >>> pattern = pyearthtools.data.patterns.ArgumentExpansion('/dir/', filename_as_arguments = True)
        >>> str(pattern.search('test','arg'))
        '/dir/test/arg/test_arg.nc'
        >>> pattern = pyearthtools.data.patterns.ArgumentExpansion('/dir/', expand_tuples = True)
        >>> [str(x) for x in pattern.search('test',('arg1', 'arg2'))]
        ['/dir/arg1/test.nc', '/dir/arg2/test.nc']
    """

    @classproperty
    def factory(self):
        return ArgumentExpansionFactory


class Argument(_Argument):
    """
    Generate FilePath Structure based upon a single argument

    The argument specifies the filename, and the path is built out from `__init__` params

    Examples:
        >>> pattern = pyearthtools.data.patterns.Argument('/dir/', extension = '.nc')
        >>> str(pattern.search('test'))
        '/dir/test.nc'
    """

    def filesystem(self, filename: str) -> Path:
        path = super().filesystem(filename)

        if isinstance(path, list):
            if len(path) == 1:
                path = path[0]
            else:
                raise TypeError(f"{filename!r} search on {self.__class__.__name__} returned more than one path.")
        return path


class ArgumentExpansionVariable(PatternVariableAware, ArgumentExpansion):
    """
    ArgumentExpansion pattern which is variable aware

    Will split each variable into a seperate file,
    using the variable as another layer in the `root_dir`

    Examples:
        >>> pattern = ArgumentExpansionVariable(root_dir = '/test/', variables = 'variable', extension = 'nc')
        >>> str(pattern.search('filename', 'arg2'))
        {'variable' : '/test/arg2/variable/filename.nc'}
    """

    @property
    def root_pattern(self) -> type[ArgumentExpansion]:
        return ArgumentExpansion

    @property
    def default_variable_parse(self) -> str:
        return "root_dir"


class ArgumentVariable(PatternVariableAware, Argument):
    """
    Argument pattern which is variable aware

    Will split each variable into a seperate file,
    using the variable as another layer in the `root_dir`

    Examples:
        >>> pattern = ArgumentVariable(root_dir = '/test/', variables = 'variable', extension = 'nc')
        >>> str(pattern.search('filename'))
        {'variable' : '/test/variable/filename.nc'}
    """

    @property
    def root_pattern(self) -> type[_Argument]:
        return Argument

    @property
    def default_variable_parse(self) -> str:
        return "root_dir"


def ArgumentExpansionFactory(*args: Any, single_argument: bool = False, variable: bool = False, **kwargs) -> _Argument:
    """Create an ArgumentExpansion pattern based on the requirements

    Args:
        single_argument (bool, optional):
            Single Argument pattern. Defaults to False.
        variable (bool, optional):
            Variable aware, splits variables when loading and saving. Defaults to False.


    Returns:
        (ArgumentExpansion):
            Created `ArgumentExpansion` pattern.
    """

    cls = ArgumentExpansion
    if single_argument and variable:
        cls = ArgumentVariable
    elif single_argument:
        cls = Argument
    elif variable:
        cls = ArgumentExpansionVariable

    return cls(*args, **kwargs)


__all__ = [
    "ArgumentExpansion",
    "Argument",
    "ArgumentExpansionVariable",
    "ArgumentVariable",
    "ArgumentExpansionFactory",
]
