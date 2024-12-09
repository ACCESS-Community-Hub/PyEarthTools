# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
Transform Utility Functions
"""

from __future__ import annotations

import xarray as xr
from typing import Any
from pathlib import Path


# DEFAULT_TRANSFORM_LOCATIONS = [
#     "__main__.",
#     "",
#     "pyearthtools.data.transforms.",
#     "pyearthtools.data.",
# ]


# def get_transforms(
#     sources: dict, order: list | None = None
# ) -> pyearthtools.data.transforms.Transform | pyearthtools.data.transforms.TransformCollection:
#     """Load [Transforms][pyearthtools.data.transforms] and initialise them from a dictionary.

#     !!! tip "Path Tip"
#         A path to the class doesn't always have to be specified, the below are automatically tried.

#         - `__main__.`
#         - `pyearthtools.data.transforms.`
#         - `pyearthtools.data.`

#     !!! tip "Multiple Tip"
#         If two or more of the same [Transform][pyearthtools.data.transforms] are wanted, add '[NUMBER]', to distinguish the key, this will be removed before import

#     Args:
#         sources (dict):
#             Dictionary specifying transforms to load and keyword arguments to pass
#         order (list, optional):
#             Override for order to load them in. Defaults to None.

#     Raises:
#         ValueError:
#             If an error occurs importing the transform
#         TypeError:
#             If an invalid type was imported
#         RuntimeError:
#             If an error occurs initialising the transforms

#     Returns:
#         (pyearthtools.data.transforms.Transform | pyearthtools.data.transforms.TransformCollection):
#             Imported and Initialised Transforms from the configuration

#     Examples:
#         >>> get_transforms(sources = {'region.lookup':{'key': 'Adelaide'}})
#         Transform Collection:
#         BoundingCut                   Cut Dataset to Adelaide region
#     """
#     transforms = []

#     if isinstance(sources, pyearthtools.data.transforms.Transform):
#         return sources

#     transforms = get_items(
#         sources,
#         order,
#         pyearthtools.data.transforms,
#         import_locations=DEFAULT_TRANSFORM_LOCATIONS,
#     )
#     return pyearthtools.data.transforms.TransformCollection(transforms)


def parse_dataset(value: str | Path | Any) -> Any:
    """
    Attempt to load dataset if value is str,
    If fails in any way, continue returning initial value
    """
    if isinstance(value, (str, Path)):
        try:
            if Path(value).exists():
                return xr.open_dataset(value)
        except Exception:
            pass
    return value


# def guess_coordinate_name(dataset) -> dict:
#     pass
# TODO Create ^
