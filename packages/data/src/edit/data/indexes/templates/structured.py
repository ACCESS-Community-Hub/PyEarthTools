# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
Template for structured data
"""

from __future__ import annotations
from pathlib import Path

from edit.data import EDITDatetime, DataNotFoundError
from edit.data.indexes import ArchiveIndex, decorators
from edit.data.indexes.utilities.spellcheck import VARIABLE_DEFAULT, VariableDefault
from edit.data.transforms import Transform, TransformCollection
from edit.data.archive import register_archive


class Structured(ArchiveIndex):
    """
    Template for structured data

    How to Use:
        Provide:
            All of these will be formatted with variables passed to init
            DIR_STRUCTURE:
                Defines the directory structure
            GLOB_TEMPLATE:
                Glob template for files beneath directories
                Use `variable` for specific variable, and strf codes for time.

        Root Directory:
            Either provide `root_directory` or register `ROOT_DIRECTORIES` to `edit.data.archive`

    """

    DIR_STRUCTURE = "{DIRECTORY}/{WITH}/{FORMATS}"
    GLOB_TEMPLATE = "{GLOB}{TEMPLATE}-GOES-{HERE}*"

    def __init__(
        self,
        variables: str | list[str],
        *,
        config_vars: dict[str, Any] = {},
        transforms: Transform | TransformCollection = TransformCollection(),
        preprocess_transforms: Transform | TransformCollection | None = None,
        data_interval: int | tuple[int, str] | None = None,
        **kwargs,
    ):
        """
        Template for Structured data.

        Uses a given directory structure, and a glob config.
        Any kwarg given to this init can be used in either, and `variable` used in the glob.

        Additionally, any strftime codes in glob will be replaced with the requested time.

        Args:
            variables (list[str] | str):
                Variables to retrieve.
            transforms (Transform | TransformCollection, optional):
                Transforms to apply to the data. Defaults to TransformCollection().
            preprocess_transforms (Transform | TransformCollection, optional):
                Transforms to apply in preprocessing for datasets. Does not work on other file formats.
                Defaults to None.
            data_interval (int | tuple[int | str], optional):
                Temporal Resolution of data,
                    Must be in (int, unit) form. Defaults to None.
            kwargs (str):
                Any additional keywords needed for dir or glob.

        Usage:
            Subclass this, and provide an init with all the needed keywords, using VariableDefault
            can help with default keywords based on the structure.
        """

        self.record_initialisation()

        variables = [variables] if isinstance(variables, str) else variables
        self.variables = variables
        self.__init_args = dict(config_vars)

        self.dir = Path(self.DIR_STRUCTURE.format(**self.__init_args))

        super().__init__(
            transforms=transforms,
            data_interval=data_interval,
            preprocess_transforms=preprocess_transforms,
            **kwargs,
        )

    def _parse_glob(self, time: EDITDatetime, variable: str) -> Path:
        """
        Parse glob path with `time` and `variable`.
        """
        glob_path = time.strftime(self.GLOB_TEMPLATE).format(**self.__init_args, variable=variable)
        dir_path = (
            Path(
                getattr(
                    self,
                    "root_directory",
                    self.ROOT_DIRECTORIES[self.__class__.__name__],
                )
            )
            / self.dir
        )

        paths = list(dir_path.glob(glob_path))
        if not len(paths) == 1:
            raise ValueError(
                f"Searching with glob found {'more' if len(paths) > 1 else 'less'} than 1 path. Cannot parse. {paths}. {dir_path}/{glob_path}"
            )
        return paths[0]

    def filesystem(
        self,
        querytime: str | EDITDatetime,
    ) -> dict[str, Path]:
        """Get filesystem paths"""
        discovered_paths = {}

        for variable in self.variables:
            discovered_paths[variable] = self._parse_glob(EDITDatetime(querytime), variable)
        return discovered_paths


# @register_archive('DataNameHere')
class ExampleStructured(Structured):
    @property
    def _desc_(self):
        return {
            "singleline": "Template for structured data",
        }

    @decorators.check_arguments(struc="PATH.TO.STRUCTURE.FILE.HERE.struc")
    def __init__(
        self,
        variables: list[str] | str,
        frequency: str,
        *,
        nature: str | VARIABLE_DEFAULT = VariableDefault,
        domain: str | VARIABLE_DEFAULT = VariableDefault,
        transforms: Transform | TransformCollection = TransformCollection(),
    ):
        super().__init__(
            variables,
            transforms=transforms,
            frequency=frequency,
            nature=nature,
            domain=domain,
        )
