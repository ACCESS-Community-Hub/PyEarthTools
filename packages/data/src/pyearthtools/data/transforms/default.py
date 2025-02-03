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

import pyearthtools.data

REPLACEMENT_NAMES = {
    "latitude": ["lat", "Latitude", "yt_ocean", "yt"],
    "longitude": ["lon", "Longitude", "xt_ocean", "xt"],
    # "depth": ["st_ocean"],
    "time": ["Time"],
}


def get_default_transforms(
    intelligence_level: int = 2,
) -> "pyearthtools.data.transforms.TransformCollection":
    """
    Get Default Transforms to be applied to all datasets

    Args:
        intelligence_level (int, optional): Level of Intelligence in operation. Defaults to 2.

    Returns:
        pyearthtools.data.transforms.TransformCollection: Collection of default transforms
    """

    transforms = pyearthtools.data.TransformCollection(None, apply_default=False)

    if intelligence_level > 0:
        transforms.append(pyearthtools.data.transforms.coordinates.StandardCoordinateNames(**REPLACEMENT_NAMES))  # type: ignore
        # transforms.append(pyearthtools.data.transforms.coordinates.standard_longitude())
    # if intelligence_level > 1:
    #     transforms.append(pyearthtools.data.transforms.coordinates.set_type("float"))

    return transforms
