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

import json
from pathlib import Path
from typing import Any

from pyearthtools.data.indexes import FileSystemIndex


def save(
    data: dict,
    callback: FileSystemIndex,
    *args,
    save_kwargs: dict[str, Any] = {},
    **kwargs,
):
    """Save json files"""
    path = callback.search(*args, **kwargs)
    if not isinstance(path, (str, Path)):
        raise NotImplementedError(f"Cannot handle saving with paths as {type(path)}")
    path = Path(path)

    with open(path, "w") as file:
        json.dump(data, file, indent=4, **save_kwargs)
    return path
