# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

import pytest

from edit.pipeline_V2 import Pipeline, iterators, filters, exceptions


def replace_on_key(**replaces):
    def replace_function(arg):
        if str(arg) in replaces:
            return replaces[str(arg)]
        return arg

    return replace_function


def test_type_filter(monkeypatch):
    pipe = Pipeline(None, filters.TypeFilter(int), iterator=iterators.Range(0, 20))
    monkeypatch.setattr(
        pipe, "_get_initial_sample", replace_on_key(**{"10": []})
    )  # Replace initial index based retrieval to run all other steps after

    with pytest.raises(exceptions.PipelineFilterException):
        pipe[10]

    assert len(list(pipe)) == 19
