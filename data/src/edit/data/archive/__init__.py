# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

"""
Provide [Index][edit.data.ArchiveIndex] for known and widely used archived data sources.

These [Indexes][edit.data.ArchiveIndex] allow a user to retrieve data with only a date after being initialised.

More archives can be added by wrapping a class with [register_archive][edit.data.archive.register_archive]

!!! Warning
    `edit.data` contains no archives itself, and require additional modules to define them.

    Currently the following exist,
    ```
     - NCI
    ```

!!! Note
    If setup correctly, any registered archive will be automatically imported if detected to be on the appropriate system.
    So, there may be no need to explicity import it.

"""

from __future__ import annotations

import warnings

import edit.data
from edit.data import archive

from edit.data.archive.extensions import register_archive

from edit.data.warnings import EDITDataWarning


def config_root():
    """Setup Root Directories"""
    if hasattr(archive, "ROOT_DIRECTORIES"):
        ROOT_DIRECTORIES: dict = archive.ROOT_DIRECTORIES  # type: ignore
        setattr(archive, "_BACKUP_ROOT_DIRECTORIES", dict(ROOT_DIRECTORIES))
    else:
        warnings.warn(
            f"`ROOT_DIRECTORIES` not found underneath `edit.data.archive`, either archives are not installed or misconfigured. Root Directories cannot be changed. ",
            UserWarning,
        )


def set_root(root_dir: dict[str, str | None] | None = None, **kwargs: str | None):
    """
    Change root directory for data sources.

    Can set value of dictionary to None to reset.

    Args:
        root_dir (dict[str, str | None] | None, optional):
            Dictionary with root directory replacements. Defaults to None.
        **kwargs (dict[str,str | None]):
            Kwargs version of root_dir
    """
    if root_dir is None:
        root_dir = {}

    root_dir.update(**kwargs)
    if not hasattr(archive, "ROOT_DIRECTORIES"):
        raise UserWarning(f"ROOT_DIRECTORIES is not set, so cannot be updated by the user.")

    ROOT_DIRECTORIES = edit.data.archive.ROOT_DIRECTORIES  # type: ignore
    _BACKUP_ROOT_DIRECTORIES = edit.data.archive._BACKUP_ROOT_DIRECTORIES  # type: ignore

    for key, value in root_dir.items():
        if key not in ROOT_DIRECTORIES:
            raise KeyError(f"Could not find {key} in ROOT_DIRECTORIES, which contains {list(ROOT_DIRECTORIES.keys())}")

        if value is None:
            value = _BACKUP_ROOT_DIRECTORIES[key]
        else:
            warnings.warn(
                f"Changing Root Directory for {key} from {ROOT_DIRECTORIES[key]} to {value} for this session",
                EDITDataWarning,
            )
        ROOT_DIRECTORIES[key] = value


def reset_root():
    """Reset all root directories"""
    if not hasattr(archive, "ROOT_DIRECTORIES"):
        raise UserWarning(f"ROOT_DIRECTORIES is not set, so cannot be updated by the user.")

    ROOT_DIRECTORIES = edit.data.archive.ROOT_DIRECTORIES  # type: ignore
    set_root(**{key: None for key in ROOT_DIRECTORIES})  # type: ignore


__all__ = [
    "set_root",
    "reset_root",
]
