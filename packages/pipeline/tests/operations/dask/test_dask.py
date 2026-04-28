# Copyright Commonwealth of Australia, Bureau of Meteorology 2026.
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

from unittest import mock
import pytest

from pyearthtools.pipeline.operations.dask.dask import DaskOperation
from pyearthtools.pipeline.operation import Operation
import numpy as np
import dask.array as da


class FakeDaskOperation(DaskOperation):
    _numpy_counterpart = "FakeNumpyOperation"

    def apply_func(self, sample):
        return "dask_apply"

    def undo_func(self, sample):
        return "dask_undo"


class FakeNumpyOperation(Operation):
    def apply_func(self, sample):
        return "numpy_apply"

    def undo_func(self, sample):
        return "numpy_undo"


def _augmented_dynamic_import(*args):
    return FakeNumpyOperation


@pytest.mark.parametrize(
    ("op", "arr_type", "expected", "dispatched"),
    [
        ("apply", np.array, "numpy_apply", True),
        ("undo", np.array, "numpy_undo", True),
        ("apply", da.array, "dask_apply", False),
        ("undo", da.array, "dask_undo", False),
    ],
)
def test_dask_operation_numpy_dispatch(op, arr_type, expected, dispatched):
    sample = arr_type(1)
    dask_op = FakeDaskOperation()

    # patch dynamic_import to ensure the fake numpy op is used
    with mock.patch("pyearthtools.utils.dynamic_import", side_effect=_augmented_dynamic_import) as mock_dynamic_import:

        # check correct dispatch depending on sample type
        #   when sample is np.ndarray, dask_op should dispatch to equivalent numpy op
        #   when sample is anything else, it should use the inbuilt op
        assert getattr(dask_op, op)(sample) == expected
        assert mock_dynamic_import.called == dispatched
        # when sample is np.ndarray, check that np.ndarray is added to recognised_types for the op
        if dispatched:
            assert np.ndarray in dask_op.recognised_types[op]
        else:
            assert dask_op.recognised_types == {}

        # run op again, to check that np.ndarry only appears once in recognised_types
        assert getattr(dask_op, op)(sample) == expected
        if dispatched:
            assert dask_op.recognised_types[op].count(np.ndarray) == 1
        else:
            assert dask_op.recognised_types == {}

        # turn off op in _operation to check that input sample is returned
        dask_op._operation[op] = False
        assert getattr(dask_op, op)(sample) == sample
