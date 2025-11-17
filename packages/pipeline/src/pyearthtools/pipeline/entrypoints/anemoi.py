# (C) Copyright 2025- European Centre for Medium-Range Weather Forecasts (ECMWF)

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.


from functools import cached_property
from pathlib import Path

from pyearthtools.pipeline import load
from pyearthtools.pipeline import Pipeline

import earthkit.data as ekd
from anemoi.datasets.create.source import Source
from anemoi.datasets.create.typing import DateList


class pyearthtoolsSource(Source):
    emoji = "🌏"  # For tracing

    def __init__(self, context, pipeline: str | Path | Pipeline):
        """Initialise the source.

        Parameters
        ----------
        context : Any
            The context for the data source.
        pipeline: str
            The path to the pyearthtools pipeline file.
        """
        super().__init__(context)
        self._pyearthtools_pipeline = pipeline

    @cached_property
    def pipeline(self) -> Pipeline:
        pipeline = self._pyearthtools_pipeline
        if isinstance(pipeline, Pipeline):
            return pipeline
        return load(pipeline)

    def execute(self, dates: DateList) -> ekd.FieldList:
        """Execute the source.

        Parameters
        ----------
        dates : DateList
            The input dates.

        Returns
        -------
        ekd.FieldList
            The output data.
        """
        fields = []
        for date in dates:
            fields.extend(ekd.from_object(self.pipeline[date.isoformat()]))  # type: ignore
        return ekd.FieldList.from_fields(fields)
