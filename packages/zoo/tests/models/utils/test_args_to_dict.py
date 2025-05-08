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

import pytest

tests = [
    (("--test", "value"), {"test": "value"}),  # Single argument
    (
        ("--test", "value", "--test_2", "value_1"),
        {"test": "value", "test_2": "value_1"},
    ),  # Two argument
    ([], {}),  # Empty
    ([""], {}),  # Empty
    (("--test",), {"test": True}),  # Boolean
    (("--test", "--test_1"), {"test": True, "test_1": True}),  # Boolean
    (
        ("--test", "--test_1", "value"),
        {"test": True, "test_1": "value"},
    ),  # Boolean with arg
    (
        ("--test", "--test_1", "value", ""),
        {"test": True, "test_1": "value"},
    ),  # Empty value
]


@pytest.mark.parametrize("args, result", tests)
def test_parse_args_to_dict(args, result):
    from pyearthtools.zoo.commands import utils

    assert utils.parse_args_to_dict(*args) == result


@pytest.mark.parametrize("args, result", tests)
def test_get_keyword_from_ctx(args, result):
    from pyearthtools.zoo.commands import utils
    import click

    class FakeContext(click.Context):
        def __init__(self, args):
            self.args = args

    assert utils.get_keyword_from_ctx(FakeContext(args)) == result


@pytest.mark.parametrize("args, result", tests)
def test_parse_str_to_dict(args, result):
    from pyearthtools.zoo.commands import utils

    assert utils.parse_str_to_dict(" ".join(args)) == result


@pytest.mark.parametrize(
    "args",
    [
        (("-test", "value")),  # Single argument
        (("--test", "value", "-test_2", "value_1")),  # Two argument
        (("test",)),  # Boolean
        (("-test",)),  # Boolean
        (("--test", "-_test_1", "value")),  # Boolean
    ],
)
def test_parse_args_to_dict_fail(args):
    from pyearthtools.zoo.commands import utils

    with pytest.raises(KeyError):
        utils.parse_args_to_dict(*args)
