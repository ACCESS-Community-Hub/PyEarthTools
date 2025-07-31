# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
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

from typing import Any
import pytest

import pyearthtools.zoo


def list_to_category(
    arg: list[tuple[str, Any]],
) -> pyearthtools.zoo.utils.CategorisedObjects:
    cats = pyearthtools.zoo.utils.CategorisedObjects("Testing")
    arg.sort()

    for e in arg:
        if not isinstance(e, tuple):
            e = (e, True)  # Add true as value by default
        cats[e[0]] = e[1]
    return cats


@pytest.mark.parametrize(
    "category, result",
    [
        (["test"], ("test",)),
        (["test/model"], ("test/model",)),
        (["test/model", "test/model1"], ("test/model", "test/model1")),
        (["test/category1/model"], ("test/category1/model",)),
    ],
)
def test_available(category, result):
    categories = list_to_category(category)
    for r in result:
        assert r.split("/")[-1] in str(categories)
        assert r.split("/")[-1] in str(repr(categories))

    assert categories.available == result
    assert tuple(list(categories.keys())) == tuple(set(r.split("/")[0] for r in result))
    assert tuple(list(categories)) == tuple(set(r.split("/")[0] for r in result))
    # assert tuple(list(categories.values())) == tuple(r.split('/')[1] for r in result)


@pytest.mark.parametrize(
    "category, result",
    [
        (None, tuple()),
        ({"test": None}, ("test",)),
        ({"test": {"model": None}}, ("test/model",)),
        ({"test": {"model": None, "model_1": None}}, ("test/model", "test/model_1")),
        ({"test": {"category1": {"model": None}}}, ("test/category1/model",)),
    ],
)
def test_update(category, result):
    categories = pyearthtools.zoo.utils.CategorisedObjects("Testing")
    categories.update(category)
    assert categories.available == result


@pytest.mark.parametrize(
    "category, result",
    [
        (["test"], ("test",)),
        (["test/model"], ("test/model",)),
        (["test/model", "test/model1"], ("test/model", "test/model1")),
        (["test/category1/model"], ("test/category1/model",)),
    ],
)
def test_available__dir(category, result):
    categories = list_to_category(category)
    for r in result:
        assert r.split("/")[0] in categories.__dir__()
        assert r in categories.__dir__()


@pytest.mark.parametrize(
    "category, result",
    [
        ([("test", {"sub_cat": 1})], ("test/sub_cat",)),
    ],
)
def test_available1(category, result):
    categories = list_to_category(category)
    for r in result:
        assert r.split("/")[-1] in str(categories)
    assert categories.available == result


@pytest.mark.parametrize(
    "category, key, result",
    [
        ([("test", True)], "test", True),
        ([("test/wow", True)], "test/wow", True),
        ([("test/wow", True)], "wow", True),
    ],
)
def test_keys(category, key, result):
    categories = list_to_category(category)
    assert categories[key] == result


@pytest.mark.parametrize(
    "category, key, expectedError",
    [
        (["test"], "tes5t", IndexError),  # Wrong key
        (["test/model"], "test-model", IndexError),  # Invalid key
        (["test/model"], "test-model", IndexError),  # Invalid key
        (["test/model", "test1/model"], "model", ValueError),  # Not Unique
    ],
)
def test_getitem_fail(category, key, expectedError):
    with pytest.raises(expectedError):
        _obj = list_to_category(category)[key]


@pytest.mark.parametrize(
    "category, key, expectedError",
    [
        (["test"], "tes5t", AttributeError),  # Wrong key
        (["test/model"], "test-model", AttributeError),  # Invalid key
        (["test/model", "test1/model"], "model", ValueError),  # Not Unique
    ],
)
def test_getattr_fail(category, key, expectedError):
    with pytest.raises(expectedError):
        _obj = getattr(list_to_category(category), key)
