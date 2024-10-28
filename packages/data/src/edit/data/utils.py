# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations

from typing import Any

import re
import os

from pathlib import Path
import logging

LOG = logging.getLogger("edit.data")


def parse_path(path: os.PathLike[Any] | str) -> Path:
    """
    Parse given `root_dir`

    Parse environment variables. .e.g. $USER evalutes correctly.

    Args:
        path (os.PathLike[Any]):
            Path to parse.

    Returns:
        Path:
            Parsed path

    """
    path_str = str(path)

    if path_str == "temp":
        LOG.warn(
            "Path being parsed was 'temp', yet this parser does not autocreate temp directories. Use `patterns` to use auto temp dirs."
        )

    matches: list[str] = re.findall(r"(\$[A-z0-9]+)", path_str)  # type: ignore
    for match in matches:
        key = match.replace("$", "")
        if key not in os.environ:
            raise ValueError(
                f"{match} was not present in the os environment. Cannot parse {path_str!r}.",
            )
        path_str = path_str.replace(match, os.environ[key])
    return Path(path_str).expanduser().resolve().absolute()
