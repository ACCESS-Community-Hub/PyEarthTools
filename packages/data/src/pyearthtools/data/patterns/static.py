# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyearthtools.data

from pyearthtools.data.patterns import PatternIndex
from pyearthtools.data.transforms import Transform, TransformCollection

"""
Retrieve Static File
"""


class Static(PatternIndex):
    """Retrieve Static File for any date retrieval"""

    def __init__(
        self,
        file: str | Path,
        variables: str | list[str] | None = None,
        *,
        enforce_existence: bool = True,
        capture_arguments: bool = False,
        transforms: Transform | TransformCollection = TransformCollection(),
        **load_kwargs: dict,
    ):
        """
        Static File based data index

        Args:
            file (str | Path):
                File to load
            variables (str | list[str], optional):
                Variables to trim loaded data to. Defaults to None.
            enforce_existence (bool, optional):
                Enforce that `file` exists. Defaults to True.
            capture_arguments (bool, optional):
                Capture arguments given to retrieval without throwing an error. Defaults to False.
            transforms (Transform | TransformCollection, optional):
                Base Transforms to apply. Defaults to TransformCollection().

        Raises:
            FileNotFoundError: If File not found

        Examples:
            >>> pattern = pyearthtools.data.patterns.Static('/dir/file.nc', enforce_existence = False)
            >>> str(pattern.search())
            '/dir/file.nc'
        """
        variables = [variables] if isinstance(variables, str) else variables
        super().__init__(
            root_dir="",
            transforms=(
                pyearthtools.data.transforms.variables.variable_trim(variables)
                if variables
                else TransformCollection() + transforms
            ),
            add_default_transforms=Path(file).suffix == ".nc",
        )
        self.record_initialisation()

        self.load_kwargs = load_kwargs
        self.capture_arguments = capture_arguments

        file = Path(file)
        if (not file.exists()) and enforce_existence:
            raise FileNotFoundError(f"{file} does not exist")
        self.file = file

    def load(self, *args, **kwargs) -> Any:
        return super().load(*args, **kwargs, **self.load_kwargs)

    def filesystem(self, *args, **kwargs):
        if not self.capture_arguments and (args or kwargs):
            raise KeyError("`filesystem` recieved arguments, but `capture_arguments` was not set.")
        return self.file

    # @functools.wraps(OperatorIndex.single)
    # def single(
    #     self,
    #     querytime: str | datetime.datetime = "",
    #     transforms: TransformCollection | Transform = None,
    #     **kwargs,
    # ) -> xr.Dataset:
    #     return super().single(querytime=querytime, transforms=transforms, **kwargs)

    # def series(
    #     self,
    #     *args,
    #     transforms: TransformCollection = TransformCollection(),
    #     **kwargs,
    # ) -> xr.Dataset:
    #     return self.single("", transforms=transforms, **kwargs)
