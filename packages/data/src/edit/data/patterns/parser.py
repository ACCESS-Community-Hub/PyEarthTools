# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
PatternIndex that parses and formats from a given str, and retrieves info from the dataset when saving

"""

from __future__ import annotations

from collections import OrderedDict
import itertools
from typing import Any, Callable
from collections.abc import Iterable

from pathlib import Path

import pandas as pd
import xarray as xr

from string import Formatter

from edit.data.patterns import PatternIndex
from edit.data.transforms import Transform, TransformCollection
from edit.data.exceptions import DataNotFoundError


def update_value(old_val: Any | list[Any] | tuple[Any, ...], new_vals: Any) -> list[Any]:
    if not isinstance(old_val, (list, tuple)):
        old_val = [old_val]

    if not isinstance(new_vals, Iterable) or isinstance(new_vals, str):
        new_vals = [new_vals]
    new_vals = list(new_vals)

    old_val = list(old_val)
    tuple(old_val.append(val) for val in new_vals if val not in old_val)
    return old_val


class ParsingPattern(PatternIndex):
    """
    PatternIndex to parse and format paths from str formats.

    Values for the formatting are expected in **kwargs / if data is saved will be added.

    Will split datasets based on what is specified in the `parse_str`
    """

    def __init__(
        self,
        root_dir: str,
        parse_str: str,
        *,
        transforms: Transform | TransformCollection = TransformCollection(),
        add_default_transforms: bool = True,
        preprocess_transforms: Transform | TransformCollection | Callable | None = None,
        **kwargs,
    ):
        """
        Create pattern from a formatting string

        If being used to retrieve data without saving it first,
        set values in `parse_str` through `kwargs` or when using `search`.

        Args:
            root_dir (str):
                Root directory to begin the path, can be 'temp' for temp directory.
            parse_str (str):
                str to parse to find paths. Use 'variable' for data vars
                E.g. '{level}/{variable}/{time:%Y%M}'.
            transforms (Transform | TransformCollection, optional):
                Transforms to add on retrieval. Defaults to TransformCollection().
            add_default_transforms (bool, optional):
                Whether to add default transforms. Defaults to True.
            preprocess_transforms (Transform | TransformCollection | Callable | None, optional):
                Transforms to always add. Defaults to None.
            kwargs (Any, optional):
                Any values to fill `parse_str` with.

        Examples:
            >>> pattern = ParsingPattern('temp', '{level:04d}.nc', level = 10)
            >>> pattern.search()
            [PosixPath('/tmp/tmpz99k9xzy/0010.nc')]
            >>> pattern = ParsingPattern('temp', '{level:04d}.nc', level = [10,20])
            >>> pattern.search()
            [PosixPath('/tmp/tmpz99k9xzy/0010.nc'), PosixPath('/tmp/tmpz99k9xzy/0020.nc')]
            >>> pattern = ParsingPattern('temp', '{time:%Y}.nc')
            >>> pattern.save(data)

        """
        super().__init__(
            root_dir=root_dir,
            transforms=transforms,
            add_default_transforms=add_default_transforms,
            preprocess_transforms=preprocess_transforms,
        )
        self.record_initialisation()
        self.update_initialisation(kwargs)

        self.parse_str = parse_str
        self._parse_kwargs = {key: update_value(val, []) for key, val in kwargs.items()}

    def get(self, *args, load_kwargs: dict[str, Any] | None = None, **kwargs) -> Any:
        """
        Get data by loading it from the search.

        All args & kwargs are passed through to search to allow extra supply of format values

        Args:
            load_kwargs (dict[str, Any] | None, optional):
                kwargs to pass to the `.load` function. Defaults to None.

        Raises:
            DataNotFoundError:
                Data could not be found

        Returns:
            (Any):
                Loaded Data
        """

        load_kwargs = load_kwargs or {}
        try:
            return self.load(self.search(*args, **kwargs), **load_kwargs)
        except FileNotFoundError as e:
            raise DataNotFoundError(f"Data with args: {args} & {kwargs} could not be found.") from e

    @property
    def _parse_components(self) -> tuple[str, ...]:
        """Get components that need to be parsed"""
        comp = map(lambda x: x[1], Formatter().parse(self.parse_str))
        return tuple(set(str(c) for c in comp if c is not None))

    def _get_parse_options(self, **additions: Any) -> list[dict[str, Any]]:
        """
        From the dictionary of parsing options and adding `additions`, get all pertubations.

        Effectively generates a list of all pertubations from a dictionary of lists.

        Args:
            **additions (Any):
                Extra options to add to parsing

        Raises:
            DataNotFoundError:
                If some options needed for parsing are missing

        Returns:
            (list[dict[str, Any]]):
                All pertubations of parsing options
        """
        _parse_kwargs = dict(self._parse_kwargs)
        for key, val in additions.items():
            _parse_kwargs[key] = update_value(_parse_kwargs.get(key, []), val)

        if len(_parse_kwargs.keys()) != len(self._parse_components):
            raise DataNotFoundError(
                f"Some options for the parser are missing, either call with them, or provide in init. {self._parse_components} != {tuple(_parse_kwargs.keys())}"
            )

        keys, values = zip(*_parse_kwargs.items())

        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def _update_parser(self, data: xr.Dataset | xr.DataArray):
        """
        Update parsing options from dataset.

        Will attempt to get all components as defined in the `parse_str` from the dataset.

        Args:
            data (xr.Dataset | xr.DataArray):
                Dataset to update options from
        """
        for component in self._parse_components:
            attr = component
            if attr == "variable":
                attr = "data_vars"
            values = getattr(data, attr)

            if isinstance(values, xr.DataArray):
                values = list(values.values)

            if "datetime64" in str([values]):
                values = map(pd.to_datetime, values)

            self._parse_kwargs[component] = list(set(update_value(self._parse_kwargs.get(component, []), values)))
        self.update_initialisation(self._parse_kwargs)

    def _get_path(self, **format_dict: Any) -> Path:
        """Get path as made by `format_dict`"""
        return Path(self.root_dir) / self.parse_str.format(**format_dict)

    def filesystem(self, options: dict[str, Any] | None = None, **kwargs: Any) -> list[Path]:
        """
        Get all paths from this Index

        Args:
            **kwargs (Any):
                Extra options to provide to the parser
        """
        paths = []
        options = options if isinstance(options, dict) else {}
        for format_dict in self._get_parse_options(**options, **kwargs):
            path = self._get_path(**format_dict)
            if path in paths:
                continue
            paths.append(path)
        return paths

    def save(self, data: xr.Dataset | xr.DataArray, *_):
        """
        Save `data` with this pattern.

        Will split the dataset according to what is given in `parse_str`.

        E.g.
            If data contains a `level` coord, and `level` is in `parse_str`, the data will be split accordingly.

        Args:
            data (xr.Dataset | xr.DataArray):
                Dataset to save

        Raises:
            KeyError:
                If `variable` is being split on, and not a `xr.Dataset`.
        """
        self._update_parser(data)

        data_subsets = OrderedDict()

        for options in self._get_parse_options():
            path = self._get_path(**options)

            sub_data = data

            variable = options.pop("variable", None)
            if variable is not None and not isinstance(data, xr.Dataset):
                raise KeyError("Cannot select on variable if data is not a `xr.Dataset`.")

            if variable is not None:
                sub_data = sub_data[variable]

            sub_data = sub_data.sel({key: [val] for key, val in options.items()})

            if isinstance(sub_data, xr.DataArray):
                sub_data = sub_data.to_dataset()

            if path in data_subsets:
                data_subsets[path] = xr.merge((data_subsets[path], sub_data))
            else:
                data_subsets[path] = sub_data

        super().save(list(data_subsets.values()))
