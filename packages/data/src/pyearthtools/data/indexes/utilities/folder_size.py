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
#
# This file extends and adapts code taken from Stack Overflow, released under 
# Creative Commons BY-SA 4.0 (International). 
#
# This information is also included in NOTICE.md
#
# From https://stackoverflow.com/questions/1392413/calculating-a-directorys-size-using-python
#

from __future__ import annotations
import os

from typing import Literal, Sequence
from pathlib import Path
import functools
import yaml
import re
import time


@functools.total_ordering
class ByteSize:
    _KB = 1024
    _suffixes = "B", "KB", "MB", "GB", "TB", "PB"

    def __init__(
        self,
        bytes: int | str | float | ByteSize | None = None,
    ):
        if isinstance(bytes, str):
            size = bytes.split(" ")[0]
            suffix = bytes.split(" ")[-1]
            if suffix not in self._suffixes:
                raise ValueError(f"Cannot parse suffix {suffix}, must be in {self._suffixes}")
            bytes = float(size) * self._KB ** self._suffixes.index(suffix)  # type: ignore

        if isinstance(bytes, ByteSize):
            bytes = bytes.bytes  # type: ignore

        if isinstance(bytes, (str, ByteSize)):
            raise TypeError(f"Cannot parse bytes of type {type(bytes)}")

        if bytes is None:
            raise ValueError("`bytes` could not be found")

        self.bytes = self.B = bytes
        self.kilobytes = self.KB = self.bytes / self._KB**1
        self.megabytes = self.MB = self.bytes / self._KB**2
        self.gigabytes = self.GB = self.bytes / self._KB**3
        self.terabytes = self.TB = self.bytes / self._KB**4
        self.petabytes = self.PB = self.bytes / self._KB**5

        for suffix in self._suffixes:
            if getattr(self, suffix) < 1:
                suffix = self._suffixes[max(self._suffixes.index(suffix) - 1, 0)]
                break

        self.readable = suffix, getattr(self, suffix)

        super().__init__()

    def __str__(self):
        return self.__format__(".2f")

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.__format__(".2f"))

    def __format__(self, format_spec):
        suffix, val = self.readable
        return "{val:{fmt}} {suf}".format(val=val, fmt=format_spec, suf=suffix)

    def __lt__(self, other):
        return self.bytes < (other.bytes if isinstance(other, ByteSize) else other)

    def __gt__(self, other):
        return self.bytes > (other.bytes if isinstance(other, ByteSize) else other)

    def __eq__(self, other):
        return self.bytes == (other.bytes if isinstance(other, ByteSize) else other)

    def __sub__(self, other):
        return self.bytes - (other.bytes if isinstance(other, ByteSize) else other)


class FolderSize(ByteSize):
    def __init__(
        self,
        files: Sequence[str | Path] | None = None,
        folder: str | Path | None = None,
    ):
        if files is None and folder is None:
            raise ValueError("Either 'files' or 'folder' must be given.")

        if files is None and folder is not None:
            files = list(Path(folder).rglob("*.*"))

        assert files is not None

        self._file_size = {}

        self._file_size = {file: Path(file).stat().st_size for file in files if Path(file).is_file()}
        bytes = sum(self._file_size.values())

        super().__init__(bytes)

    def limit(
        self,
        max_size: ByteSize | str | int | float,
        key: Literal["modified", "created"] = "modified",
    ) -> list[Path]:
        """
        Using known directory size, limit directory to be at most `max_size`.

        This will return the oldest of `key` files to delete to limit directory size to `max_size`.

        Args:
            max_size (ByteSize | str):
                Maximum size the directory should be.
            key (Literal['modified', 'created'], optional):
                Key to find age with. Defaults to 'modified'.

        Raises:
            RuntimeError:
                If this class was inited without files specified

        Returns:
            (list[Path]):
                Oldest files, which deleting would limit this directory at `max_size`
        """
        if isinstance(max_size, str):
            max_size = ByteSize(max_size)

        if self._file_size is None:
            raise RuntimeError(
                "As this `ByteSize` was not initialised with a `folder` or `files`, a limit cannot be found."
            )

        key_to_func = {
            "modified": os.path.getmtime,
            "created": os.path.getctime,
        }
        func = key_to_func[key]
        files = sorted(
            self._file_size.items(),
            key=lambda x: time.time() - func(x[0]) if Path(x[0]).exists() else 0,
            reverse=True,
        )

        difference = self - max_size
        count = 0

        extra_files: list[Path] = []
        while count < difference and len(files) > 0:
            file, size = files.pop(0)
            count += size
            extra_files.append(Path(file))

        return extra_files


def ByteSizeRepresenter(dumper, data):
    return dumper.represent_scalar("!ByteSize", "%s" % data)


def ByteSizeConstructor(loader, node):
    value = loader.construct_scalar(node)
    return ByteSize(value)


pattern = re.compile(r"\d* [A-z]?B")

yaml.add_implicit_resolver("!ByteSize", pattern)
yaml.add_constructor("!ByteSize", ByteSizeConstructor)
yaml.add_representer(ByteSize, ByteSizeRepresenter)
