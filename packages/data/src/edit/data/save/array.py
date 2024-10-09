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

import numpy as np

from edit.data.indexes import FileSystemIndex
from edit.data.save.utils import ManageFiles

VALID_EXTENSIONS = [".npy", ".numpy"]
ARRAY_TIMEOUT = 10


def save(
    dataarray: np.ndarray,
    callback: FileSystemIndex,
    *args,
    save_kwargs: dict[str, Any] = {},
    try_thread_safe: bool = True,
    **kwargs,
):
    path = callback.search(*args, **kwargs)
    if not isinstance(path, (str, Path)):
        raise NotImplementedError(f"Cannot handle saving with paths as {type(path)}")

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if path.suffix not in VALID_EXTENSIONS:
        raise ValueError(
            f"Saving numpy arrays requires a suffix in {VALID_EXTENSIONS}, not {path.suffix!r} on {path!r}"
        )

    def _save(data, file, **kwargs):
        with ManageFiles(file, timeout=ARRAY_TIMEOUT, lock=try_thread_safe, uuid=not try_thread_safe) as (
            temp_file,
            exist,
        ):
            if not exist:
                assert isinstance(temp_file, (str, Path))
                np.save(temp_file, data, **kwargs)

    if isinstance(dataarray, (tuple, list)):
        for i, data in enumerate(dataarray):
            subpath = path / f"{i}{path.suffix}"
            subpath.parent.mkdir(parents=True, exist_ok=True)
            _save(data, subpath, **save_kwargs)

    else:
        _save(dataarray, path, **save_kwargs)

    return path
