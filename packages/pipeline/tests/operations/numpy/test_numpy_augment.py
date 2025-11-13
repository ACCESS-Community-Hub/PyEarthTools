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

from pyearthtools.pipeline.operations.numpy import augment

import numpy as np
import pytest


@pytest.mark.parametrize(
    "seed, rotations",
    [
        (42, 0),
        (1, 1),
        (4, 2),
        (2, 3),
    ]
)
def test_Rotate(seed, rotations):

    original = np.array([
        [1, 2],
        [4, 3]
    ])

    # The result depends on the random seed. This one has been manually checked
    # to produce a certain number of rotations the first time.
    match rotations:
        case 0:
            expected = np.array([
                [1, 2],
                [4, 3]
            ])
        case 1:
            expected = np.array([
                [4, 1],
                [3, 2]
            ])
        case 2:
            expected = np.array([
                [3, 4],
                [2, 1]
            ])
        case 3:
            expected = np.array([
                [2, 3],
                [1, 4]
            ])


    rotate = augment.Rotate(seed=seed, axis=(1, 0))

    result = rotate.apply_func(original)
    assert (result == expected).all()


@pytest.mark.parametrize(
    "seed, should_flip",
    [
        (0, True),
        (1, False),
    ]
)
def test_Flip(seed, should_flip):

    original = np.array([
        [1, 2],
        [3, 4]
    ])
    
    flipped = np.array([
        [4, 3],
        [2, 1]
    ])

    # The result depends on the random seed. This one has been manually checked
    # to produce a single rotation the first time.
    expected = flipped if should_flip else original
    flip = augment.Flip(seed=seed, axis=(1, 0))

    result = flip.apply_func(original)
    assert (result == expected).all()
