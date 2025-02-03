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


from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pyearthtools.data.indexes import FileSystemIndex
from pyearthtools.data.save.utils import ManageFiles

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
