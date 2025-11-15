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
from pyearthtools.utils.initialisation import imports
from collections import namedtuple
import pytest
import io
from types import ModuleType


def test_get_name():
    """
    Test the test_get_name functionality
    """
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


def test_CatalogEntry_Nones():
    class mockEntry:
        def __init__(self, x, kwarg_1):
            self.item_class = None
            self.x = x
            self.kwarg_1 = kwarg_1

    mockEntry("x", "kwarg1")

    # Test item_class with "String" instance
    ce = catalog.CatalogEntry("None", args=["None"], name=["None"], kwargs={"my_kwarg": "None"})
    assert ce.item_class is None  # Test item_class set to None

    ce = catalog.CatalogEntry("pytest", args=["None"], name=["None"], kwargs={"my_kwarg": "None"})
    assert isinstance(ce.item_class, ModuleType)  # Test item_class dynamic import work

    ce = catalog.CatalogEntry(mockEntry, args=["None"], name=["None"], kwargs={"my_kwarg": "None"})
    args = ce.to_dict()["args"]
    kwargs = ce.to_dict()["kwargs"]
    assert args[0] is None  # Test str to None conversion for args
    assert kwargs["my_kwarg"] is None  # Test str to None conversion for kwargs


def test_CatalogEntry_Save_and_Load(monkeypatch, tmpdir):
    class mock_callable:
        def __init__(self, _x):
            return None

        def mock_callable(self):
            return "Class has a method with the same name for dynamic importing save/load!"

    tmp_path = tmpdir.mkdir("sub").join("cat.tmp")
    tmp_path = tmp_path.strpath

    monkeypatch.setattr(imports, "dynamic_import", mock_callable)

    test_mock_callable = mock_callable("x")
    assert isinstance(test_mock_callable.mock_callable(), str)

    ce = catalog.CatalogEntry(mock_callable, args=["X"])
    cat = catalog.Catalog(catalog_name="smart_test_catalog", entries={"foo": ce})
    cat.save(tmp_path)
    loaded_cat = cat.load(tmp_path)

    cat_dict = cat.to_dict()
    loaded_cat_dict = loaded_cat.to_dict()

    first_cat_key = list(cat_dict.keys())[0]
    first_loaded_cat_key = list(loaded_cat_dict.keys())[0]

    # Assert that we were able to load in the same data as what was saved...
    assert loaded_cat_dict[first_loaded_cat_key]["args"] == cat_dict[first_cat_key]["args"]
