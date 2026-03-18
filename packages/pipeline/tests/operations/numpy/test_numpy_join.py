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

from functools import partial

from pyearthtools.pipeline.operations.numpy.join import Stack, VStack, HStack, Concatenate

import numpy as np
import pytest


def _arrays(*shapes):
    """Create numpy arrays with given shapes whose elements are sequential integers."""
    offset = 0
    result = []
    for shape in shapes:
        size = int(np.prod(shape))
        result.append(np.arange(offset, offset + size).reshape(shape))
        offset += size
    return tuple(result)


# this parameterizations passes in the joiner class to test, with an appropriate axis as needed.
# It compares the joined result to an equivalent numpy function, partially initialised with axis as needed.
# The shape of the input array passed to the test is adjusted based on the joiner.
@pytest.mark.parametrize(
    ("joiner", "equiv_op", "input_arrays"),
    [
        pytest.param(Stack(axis=0), partial(np.stack, axis=0), _arrays((2, 3), (2, 3)), id="Stack-axis0"),
        pytest.param(Stack(axis=1), partial(np.stack, axis=1), _arrays((2, 3), (2, 3)), id="Stack-axis1"),
        pytest.param(Stack(axis=2), partial(np.stack, axis=2), _arrays((2, 3), (2, 3)), id="Stack-axis2"),
        pytest.param(VStack(), np.vstack, _arrays((1, 3, 2), (2, 3, 2)), id="VStack"),
        pytest.param(HStack(), np.hstack, _arrays((3, 1, 2), (3, 2, 2)), id="HStack"),
        pytest.param(
            Concatenate(axis=0), partial(np.concatenate, axis=0), _arrays((1, 3, 2), (2, 3, 2)), id="Concatenate-axis0"
        ),
        pytest.param(
            Concatenate(axis=1), partial(np.concatenate, axis=1), _arrays((3, 1, 2), (3, 2, 2)), id="Concatenate-axis1"
        ),
        pytest.param(
            Concatenate(axis=2), partial(np.concatenate, axis=2), _arrays((3, 2, 1), (3, 2, 2)), id="Concatenate-axis2"
        ),
    ],
)
def test_join(joiner, equiv_op, input_arrays):
    """Tests that joiners reproduce their numpy equivalents and are reversible."""
    name = type(joiner).__name__

    result = joiner.join(input_arrays)
    expected = equiv_op(input_arrays)
    assert np.array_equal(result, expected), f"{name}.join() did not reproduce expected behaviour."

    unjoined = joiner.unjoin(result)
    assert isinstance(unjoined, tuple), f"{name}.unjoin() did not return a tuple."
    for arr_undo, arr in zip(unjoined, input_arrays, strict=True):
        assert np.array_equal(arr_undo, arr), f"{name}.unjoin() did not return original arrays."
