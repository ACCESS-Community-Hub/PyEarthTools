# Copyright Commonwealth of Australia, Bureau of Meteorology 2025.
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

from pyearthtools.pipeline.operations.dask import join

import dask.array as da
import numpy as np
import pytest


def test_stacks():
    """Tests that join.Stack reproduces da.stack behaviour."""

    numpy_arrays = (
        da.array(range(6)).reshape((2, 3)),
        da.array(range(6, 12)).reshape((2, 3)),
    )

    for stack_axis in range(3):
        stack = join.Stack(axis=stack_axis)
        result = stack.join(numpy_arrays)
        expected = da.stack(numpy_arrays, axis=stack_axis)
        assert np.array_equal(
            result.compute(), expected.compute()
        ), f"Stack(axis={stack_axis}).join() did not reproduce da.stack"
        unjoined_result = stack.unjoin(result)
        assert isinstance(
            unjoined_result, tuple
        ), f"Stack(axis={stack_axis}).unjoin() did not unjoin the input sample into tuples."
        for arr_undo, arr in zip(unjoined_result, numpy_arrays, strict=True):
            assert np.array_equal(
                arr_undo.compute(), arr.compute()
            ), f"Stack(axis={stack_axis}).unjoin() did not return original arrays"


@pytest.fixture
def concat_array_data():
    return (
        da.array(range(6)),
        da.array(range(6, 18)),
    )


@pytest.mark.parametrize(
    ("joiner", "equiv_np_op", "input_shapes"),
    (
        (join.VStack, da.vstack, ((1, 3, 2), (2, 3, 2))),
        (join.HStack, da.hstack, ((3, 1, 2), (3, 2, 2))),
    ),
)
def test_vstack(joiner, equiv_np_op, input_shapes, concat_array_data):
    """Tests that join.XStack reproduces da.xstack behaviour."""

    input_arrays = tuple(arr.reshape(shape) for arr, shape in zip(concat_array_data, input_shapes))

    stack = joiner()
    result = stack.join(input_arrays)
    expected = equiv_np_op(input_arrays)
    assert np.array_equal(
        result.compute(), expected.compute()
    ), f"{joiner.__name__}.join() did not reproduce {equiv_np_op.__name__} behaviour."
    unjoined_result = stack.unjoin(result)
    assert isinstance(
        unjoined_result, tuple
    ), f"{joiner.__name__}.unjoin() did not unjoin the input sample into tuples."
    for arr_undo, arr in zip(unjoined_result, input_arrays, strict=True):
        assert np.array_equal(
            arr_undo.compute(), arr.compute()
        ), f"{joiner.__name__}.unjoin() did not return original arrays."


@pytest.mark.parametrize(
    ("concat_axis", "input_shapes"),
    (
        (0, ((1, 3, 2), (2, 3, 2))),
        (1, ((3, 1, 2), (3, 2, 2))),
        (2, ((3, 2, 1), (3, 2, 2))),
    ),
)
def test_concatenate(concat_axis, input_shapes, concat_array_data):
    """Tests that join.Concatenate reproduces da.concatenate behaviour."""

    input_arrays = tuple(arr.reshape(shape) for arr, shape in zip(concat_array_data, input_shapes))

    stack = join.Concatenate(axis=concat_axis)
    result = stack.join(input_arrays)
    expected = da.concatenate(input_arrays, axis=concat_axis)
    assert np.array_equal(
        result.compute(), expected.compute()
    ), f"Concatenate(axis={concat_axis}) did not reproduce da.concatenate behaviour."
    unjoined_result = stack.unjoin(result)
    assert isinstance(
        unjoined_result, tuple
    ), f"Concatenate(axis={concat_axis}).unjoin() did not unjoin the input sample into tuples."
    for arr_undo, arr in zip(unjoined_result, input_arrays, strict=True):
        assert np.array_equal(
            arr_undo.compute(),
            arr.compute(),
            f"Concatenate(axis={concat_axis}).unjoin() did not return original arrays.",
        )
