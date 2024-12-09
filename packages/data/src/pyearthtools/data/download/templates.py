# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
Templates for Downloading data
"""

from __future__ import annotations
from abc import abstractmethod, ABCMeta
from typing import Type
import warnings

from tempfile import TemporaryDirectory
from pathlib import Path

import urllib3
import functools

from pyearthtools.data.indexes import CachingIndex
from pyearthtools.data import IndexWarning, DataNotFoundError

from pyearthtools.data.patterns import PatternIndex


@functools.lru_cache(None)
def has_connection(host="http://www.example.com/") -> bool:
    """Check if current kernel has an internet connection"""
    try:
        urllib3.request("GET", host, retries=False)
        return True
    except Exception:
        return False


class DownloadIndex(CachingIndex, metaclass=ABCMeta):
    """
    Core `DownloadIndex` to be implemented by any class downloading data.
    Subclasses from the `CachingIndex` to provide automated generation, and saving if a `cache` is specified.

    By default, all data is cached to a temporary directory.

    A child class must implement `.download`
    """

    def __init__(
        self,
        *args,
        pattern: PatternIndex | Type | str | None = "TemporalExpandedDateVariable",
        **kwargs,
    ):
        if not has_connection():
            warnings.warn(
                "Runtime has no internet connection, and thus cannot download data. Will only work for cached data.",
                IndexWarning,
            )
        kwargs["cache"] = kwargs.pop("cache", "temp")
        self._temp_files: list[TemporaryDirectory] = []

        super().__init__(*args, pattern=pattern, **kwargs)

    @abstractmethod
    def download(self, *args, **kwargs):
        """
        Actual method to download data, must be implemented by child class.

        Should return downloaded data object to be automatically cached.
        """
        raise NotImplementedError(f"{self.__class__} has not implemented the `download` method.")

    def _generate(self, *args, **kwargs):
        """Generate data through the download method"""
        if not has_connection():
            raise DataNotFoundError(
                f"No internet connection found, cannot download data from {self.__class__}.\n{args!r}, {kwargs!r}."
            )
        return self.download(*args, **kwargs)

    def get_tempdir(self, remove_past: bool = False) -> Path:
        """
        Get temporary directory to store files in.

        These directories will be stored and removed upon deletion of this object.

        Args:
            remove_past (bool, optional):
                Whether past temp directories should be removed when making a new one.
                Defaults to False.
        """
        file = TemporaryDirectory()
        if remove_past:
            for temp in self._temp_files:
                temp.cleanup()
        self._temp_files = []

        self._temp_files.append(file)
        return Path(file.name)

    def __del__(self):
        """Explicity cleans up temp dirs"""
        if hasattr(self, "_temp_files"):
            [x.cleanup() for x in self._temp_files]
        return super().__del__()
