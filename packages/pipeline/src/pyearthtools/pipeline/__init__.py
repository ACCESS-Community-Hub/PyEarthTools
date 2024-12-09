# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

# type: ignore[reportUnusedImport]
# ruff: noqa: F401

"""
# `pyearthtools.pipeline`

Create repeatable pipelines, transforming data and preparing for downstream applications.

Utilises `pyearthtools.data` to provide the data indexes, transforms to apply on data, and introduces 
operations, filters, samplers and iterators.

```python
import pyearthtools.data
import pyearthtools.pipeline

pipeline = pyearthtools.pipeline.Pipeline(
    pyearthtools.data.archive.ERA5.sample(), # Get ERA5

    pyearthtools.pipeline.operations.xarray.values.FillNan(), # FillNans
    pyearthtools.pipeline.operations.xarray.conversion.ToNumpy(), # Convert to Numpy
)

pipeline['2000-01-01T00']

```

"""

__version__ = "0.1.0"

import pyearthtools.pipeline.logger

from pyearthtools.pipeline.save import save, load
from pyearthtools.pipeline.controller import Pipeline, PipelineIndex

from pyearthtools.pipeline.operation import Operation

from pyearthtools.pipeline import (
    branching,
    exceptions,
    filters,
    iterators,
    samplers,
    operations,
    modifications,
)

from pyearthtools.pipeline.marker import Marker, Empty

from pyearthtools.pipeline.modifications import Cache, SequenceRetrieval, TemporalRetrieval

from pyearthtools.pipeline.samplers import Sampler

from pyearthtools.pipeline.iterators import Iterator

from pyearthtools.pipeline.parallel import get_parallel

from pyearthtools.pipeline.exceptions import (
    PipelineException,
    PipelineFilterException,
    PipelineRuntimeError,
    PipelineTypeError,
)
from pyearthtools.pipeline.warnings import PipelineWarning

from pyearthtools.pipeline import config

__all__ = [
    "Sampler",
    "Iterator",
    "Pipeline",
    "Operation",
    "branching",
    "exceptions",
    "filters",
    "iterators",
    "samplers",
    "operations",
    "modifications",
]

__version__ = "0.1.0"
