# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
Provide [Index][pyearthtools.data.ArchiveIndex] for known and widely used archived data sources.

These [Indexes][pyearthtools.data.ArchiveIndex] allow a user to retrieve data with only a date after being initialised.

More archives can be added by wrapping a class with [register_archive][pyearthtools.data.archive.register_archive]

!!! Warning
    `pyearthtools.data` contains no archives itself, and require additional modules to define them.

    Currently the following exist,
    ```
     - NCI
     - UKMO
    ```

!!! Note
    If setup correctly, any registered archive will be automatically imported if detected to be on the appropriate system.
    So, there may be no need to explicity import it.

"""

from pyearthtools.data.archive.extensions import register_archive

from pyearthtools.data.archive.root import set_root, reset_root, config_root


ZARR_IMPORTED = True
try:
    from pyearthtools.data.archive.zarr import ZarrIndex, ZarrTimeIndex  # noqa: F401
except (ImportError, ModuleNotFoundError):
    ZARR_IMPORTED = False


__all__ = [
    "set_root",
    "reset_root",
    "config_root",
    "register_archive",
]

if ZARR_IMPORTED:
    __all__.extend(["ZarrIndex", "ZarrTimeIndex"])
