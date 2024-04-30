# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

"""
Transform to apply to values of datasets
"""

from __future__ import annotations
from typing import Literal

import xarray as xr

import edit.data
from edit.data.transform.transform import Transform


def fill(
    coordinates: str | list[str],
    *coords,
    direction: Literal["forward", "backward"] = "forward",
    limit: int | None = None,
) -> Transform:
    """
    Apply ffill or bfill on a dataset depending on given `direction`

    Args:
        coordinates (str | list[str]):
            Coordinates to run fill on
        direction (Literal["forward", "backward"], optional):
            Direction to apply fill, either ffill or bfill. Defaults to 'forward'.
        limit (int | None, optional):
            limit to pass to fill. Defaults to None.

    Returns:
        (Transform):
            Transform to apply fill
    """
    coordinates = coordinates if isinstance(coordinates, (list, tuple)) else [coordinates]
    coordinates = [*coords, *coordinates]

    class FillTransform(Transform):
        """Apply fill on the given coordinates"""

        @property
        def _info_(self):
            return dict(coordinates=coordinates, limit=limit, direction=direction)

        def apply(self, dataset: xr.Dataset) -> xr.Dataset:
            encod = edit.data.transform.attributes.set_encoding(reference=dataset)
            for coord in coordinates:
                if direction == "forward":
                    dataset = dataset.ffill(coord, limit=limit)
                elif direction == "backward":
                    dataset = dataset.bfill(coord, limit=limit)
                else:
                    raise ValueError(f"Cannot parse {direction}, must be either 'forward' or 'backward'.")
            return encod(dataset)

    return FillTransform()


ffill = lambda *a, **b: fill(*a, direction=b.pop("direction", "forward"), **b)
bfill = lambda *a, **b: fill(*a, direction=b.pop("direction", "backward"), **b)
