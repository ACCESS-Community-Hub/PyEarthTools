# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

import os

from edit.data.archive import zarr

from edit.data import patterns
from edit.data.patterns.default import PatternIndex


class ZarrIndex(zarr.ZarrIndex, PatternIndex):
    """
    Zarr archive for use as a pattern for `CachingIndex`.

    If filling in a template archive, ensure `template` = True.

    This will cause any cache checks of existence to return False, and thus generate the data.

    For actual usage, `template` = False.
    """

    def __init__(self, root_dir: os.PathLike, **kwargs):
        root_dir, temp_dir = patterns.utils.parse_root_dir(str(root_dir))
        super().__init__(root_dir, **kwargs, root_dir=root_dir)
        self.temp_dir = temp_dir

    def search(self, *_):
        # Prevents args being passed to underlying search
        return super().search()

    def save(self, data, *_, **kwargs):
        # Prevents args being passed to underlying save
        return super().save(data, **kwargs)


class ZarrTimeIndex(zarr.ZarrTimeIndex, ZarrIndex):
    """Time aware Zarr Pattern archive"""
