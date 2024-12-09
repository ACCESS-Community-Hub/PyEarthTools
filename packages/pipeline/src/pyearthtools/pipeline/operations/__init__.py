# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
Pipeline Operations

| SubModules | Info |
| ---------- | ---- |
| numpy | Numpy arrays |
| xarray | Xarray |
| dask   | Dask arrays |
| transform   | Transformations |
"""

import warnings

from pyearthtools.pipeline.operations import xarray, numpy
from pyearthtools.pipeline.operations.transforms import Transforms
from pyearthtools.pipeline.operations import transform

try:
    from pyearthtools.pipeline.operations import dask
except (ImportError, ModuleNotFoundError) as e:
    warnings.warn(f"Unable to import `operations.dask` due to {e}", ImportWarning)

__all__ = [
    "xarray",
    "numpy",
    "transform",
    "Transforms",
    "dask",
]
