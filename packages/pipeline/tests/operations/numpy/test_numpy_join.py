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

from pyearthtools.pipeline.operations.numpy import join

import numpy as np
import pytest


def test_stacks():
    """Tests that join.Stack reproduces np.stack behaviour."""

    numpy_arrays = (
        np.array(range(6)).reshape((2, 3)),
        np.array(range(6, 12)).reshape((2, 3)),
    )

    for stack_axis in range(3):
        stack = join.Stack(axis=stack_axis)
        result = stack.join(numpy_arrays)
        expected = np.stack(numpy_arrays, axis=stack_axis)
        assert np.array_equal(result, expected), f"Stack(axis={stack_axis}).join() did not reproduce np.stack"
        unjoined_result = stack.unjoin(result)
        assert isinstance(
            unjoined_result, tuple
        ), f"Stack(axis={stack_axis}).unjoin() did not unjoin the input sample into tuples."
        for arr_undo, arr in zip(unjoined_result, numpy_arrays, strict=True):
            assert np.array_equal(arr_undo, arr), f"Stack(axis={stack_axis}).unjoin() did not return original arrays"


@pytest.fixture
def concat_array_data():
    return (
        np.array(range(6)),
        np.array(range(6, 18)),
    )


@pytest.mark.parametrize(
    ("joiner", "equiv_np_op", "input_shapes"),
    (
        (join.VStack, np.vstack, ((1, 3, 2), (2, 3, 2))),
        (join.HStack, np.hstack, ((3, 1, 2), (3, 2, 2))),
    ),
)
def test_vstack(joiner, equiv_np_op, input_shapes, concat_array_data):
    """Tests that join.XStack reproduces np.xstack behaviour."""

    input_arrays = tuple(arr.reshape(shape) for arr, shape in zip(concat_array_data, input_shapes))

    stack = joiner()
    result = stack.join(input_arrays)
    expected = equiv_np_op(input_arrays)
    assert np.array_equal(
        result, expected
    ), f"{joiner.__name__}.join() did not reproduce {equiv_np_op.__name__} behaviour."
    unjoined_result = stack.unjoin(result)
    assert isinstance(
        unjoined_result, tuple
    ), f"{joiner.__name__}.unjoin() did not unjoin the input sample into tuples."
    for arr_undo, arr in zip(unjoined_result, input_arrays, strict=True):
        assert np.array_equal(arr_undo, arr), f"{joiner.__name__}.unjoin() did not return original arrays."


@pytest.mark.parametrize(
    ("concat_axis", "input_shapes"),
    (
        (0, ((1, 3, 2), (2, 3, 2))),
        (1, ((3, 1, 2), (3, 2, 2))),
        (2, ((3, 2, 1), (3, 2, 2))),
    ),
)
def test_concatenate(concat_axis, input_shapes, concat_array_data):
    """Tests that join.Concatenate reproduces np.concatenate behaviour."""

    input_arrays = tuple(arr.reshape(shape) for arr, shape in zip(concat_array_data, input_shapes))

    stack = join.Concatenate(axis=concat_axis)
    result = stack.join(input_arrays)
    expected = np.concatenate(input_arrays, axis=concat_axis)
    assert np.array_equal(
        result, expected
    ), f"Concatenate(axis={concat_axis}) did not reproduce np.concatenate behaviour."
    unjoined_result = stack.unjoin(result)
    assert isinstance(
        unjoined_result, tuple
    ), f"Concatenate(axis={concat_axis}).unjoin() did not unjoin the input sample into tuples."
    for arr_undo, arr in zip(unjoined_result, input_arrays, strict=True):
        assert np.array_equal(
            arr_undo, arr, f"Concatenate(axis={concat_axis}).unjoin() did not return original arrays."
        )
