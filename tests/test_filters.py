# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

import pytest

from edit.pipeline_V2 import Pipeline, iterators, filters, exceptions, Operation, config
from tests.fake_pipeline_steps import FakeIndex

config.RUN_PARALLEL = False

class ReplaceOnKey(Operation):
    def __init__(self, **replaces):
        super().__init__(operation='apply')
        self.replaces = replaces
        
    def apply_func(self, sample):
        if str(sample) in self.replaces:
            return self.replaces[str(sample)]
        return sample
    

def test_type_filter():
    pipe = Pipeline(FakeIndex(), ReplaceOnKey(**{'10': 'break'}), filters.TypeFilter(int), iterator=iterators.Range(0, 20))

    with pytest.raises(exceptions.PipelineFilterException):
        pipe[10]

    assert len(list(pipe)) == 19
