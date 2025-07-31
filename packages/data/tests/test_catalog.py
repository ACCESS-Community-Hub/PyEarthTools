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

import pyearthtools
from pyearthtools.data import catalog
from collections import namedtuple
import pytest
import io


def test_get_name():

    result = catalog.get_name("testname")
    assert result == "testname"

    result = catalog.get_name(pyearthtools.data)
    assert """pyearthtools.data' from """ in result

    result = catalog.get_name(type(pyearthtools.data))
    assert result == "type"

    mockObj = namedtuple("Mock", ["name"])("mockName")
    result = catalog.get_name(mockObj)
    assert result == "Mock(name='mockName')"

    mockObj = namedtuple("Mock2", ["noname"])("mockName2")
    result = catalog.get_name(mockObj)
    assert result == "Mock2(noname='mockName2')"


def test_CatalogEntry():
    def mockEntry():
        """
        Dummy function for catalog entry
        """
        return "foo"

    mockEntry()

    ce = catalog.CatalogEntry(mockEntry, args=[], name="MockEntry")

    with pytest.raises(NotImplementedError):
        ce()

    with pytest.raises(AttributeError):

        _error = ce.__getattr__("item_class")

    with pytest.raises(AttributeError):
        ce.nonexisting

    assert ce.name == "MockEntry"

    as_dict = ce.to_dict()
    assert as_dict["args"] == []
    assert as_dict["item_class"] == "test_catalog.mockEntry"

    therepr = repr(ce)
    assert "MockEntry - test_catalog.mockEntry" in therepr


def test_Catalog():
    def mockEntry():
        return "foo"

    mockEntry()

    ce = catalog.CatalogEntry(mockEntry, args=[], name="MockEntry")

    cat = catalog.Catalog(catalog_name="Test Catalog", entries={"TestEntryKey": ce})

    # Dictionary conversion
    as_dict = cat.to_dict()
    entrykey = as_dict["TestEntryKey"]
    assert entrykey["name"] == "TestEntryKey"

    # Saving to file
    output_io = io.StringIO()
    _save_dict = cat.save(output_io)  # Smoke test a save operation

    # Create and pop
    cat = catalog.Catalog(catalog_name="Test Catalog", entries={"TestEntryKey": ce})
    popped = cat.pop("TestEntryKey")
    assert popped == ce

    # Confirm can't pop the same thing twice
    with pytest.raises(KeyError):
        popped = cat.pop("TestEntryKey")

    # Create and remove
    cat = catalog.Catalog(catalog_name="Test Catalog", entries={"TestEntryKey": ce})
    cat.remove("TestEntryKey")
    with pytest.raises(KeyError):
        popped = cat.remove("TestEntryKey")
