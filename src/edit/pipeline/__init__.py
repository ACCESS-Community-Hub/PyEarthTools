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
# `edit.pipeline`

Create repeatable pipelines, transforming data and preparing for downstream applications.

Utilises `edit.data` to provide the data indexes, transforms to apply on data, and introduces 
operations, filters, samplers and iterators.

```python
import edit.data
import edit.pipeline

pipeline = edit.pipeline.Pipeline(
    edit.data.archive.ERA5.sample(), # Get ERA5

    edit.pipeline.operations.xarray.values.FillNan(), # FillNans
    edit.pipeline.operations.xarray.conversion.ToNumpy(), # Convert to Numpy
)

pipeline['2000-01-01T00']

```

"""

from edit.pipeline.save import save, load
from edit.pipeline.controller import Pipeline, PipelineIndex

from edit.pipeline.operation import Operation

from edit.pipeline import (
    branching,
    exceptions,
    filters,
    iterators,
    samplers,
    operations,
    modifications,
)

from edit.pipeline.marker import Marker

from edit.pipeline.modifications import Cache, SequenceRetrieval, TemporalRetrieval

from edit.pipeline.samplers import Sampler

from edit.pipeline.iterators import Iterator

from edit.pipeline.parallel import get_parallel

from edit.pipeline.exceptions import (
    PipelineException,
    PipelineFilterException,
    PipelineRuntimeError,
    PipelineTypeError,
)
from edit.pipeline.warnings import PipelineWarning

from edit.pipeline import config

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
__version__ = "1.0.1"
