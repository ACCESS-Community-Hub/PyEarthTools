# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations

import edit.data

REPLACEMENT_NAMES = {
    "latitude": ["lat", "Latitude", "yt_ocean", "yt"],
    "longitude": ["lon", "Longitude", "xt_ocean", "xt"],
    # "depth": ["st_ocean"],
    "time": ["Time"],
}


def get_default_transforms(
    intelligence_level: int = 2,
) -> "edit.data.transforms.TransformCollection":
    """
    Get Default Transforms to be applied to all datasets

    Args:
        intelligence_level (int, optional): Level of Intelligence in operation. Defaults to 2.

    Returns:
        edit.data.transforms.TransformCollection: Collection of default transforms
    """

    transforms = edit.data.TransformCollection(None, apply_default=False)

    if intelligence_level > 0:
        transforms.append(edit.data.transforms.coordinates.StandardCoordinateNames(**REPLACEMENT_NAMES))  # type: ignore
        # transforms.append(edit.data.transforms.coordinates.standard_longitude())
    # if intelligence_level > 1:
    #     transforms.append(edit.data.transforms.coordinates.set_type("float"))

    return transforms
