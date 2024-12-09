# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

import pytest

from pyearthtools.data.indexes import decorators


@decorators.alias_arguments(value=["v", "val"])
@decorators.check_arguments(value=[1, 2])
def fake_function(value):
    return value


@pytest.mark.parametrize(
    "key, value, error",
    [
        ("value", 1, False),
        ("val", 1, False),
        ("why", 1, True),
        ("value", 1, False),
        ("value", 2, False),
        ("value", 0, True),
        ("value", "Test", True),
    ],
)
def test_fake_function(key, value, error: bool):
    if error:
        with pytest.raises(Exception):
            assert fake_function(**{key: value}) == value
    else:
        assert fake_function(**{key: value}) == value
