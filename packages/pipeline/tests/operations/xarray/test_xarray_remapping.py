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

import pytest
import builtins
import sys

realimport = builtins.__import__


def monkeypatch_healpy_import(name, globals=None, locals=None, fromlist=(), level=0):
    """A custom import function that raises ImportError if trying to import healpy."""
    if name == "healpy":
        raise ImportError()
    return realimport(name, globals, locals, fromlist, level)


def test_no_healpy(monkeypatch):
    """Tests that expected warning is raised when trying to use HEALPix without healpy installed."""
    monkeypatch.delitem(sys.modules, "healpy", raising=False)
    monkeypatch.delitem(sys.modules, "pyearthtools.pipeline.operations.xarray.remapping", raising=False)
    monkeypatch.delitem(sys.modules, "pyearthtools.pipeline.operations.xarray.remapping.healpix", raising=False)
    monkeypatch.setattr(builtins, "__import__", monkeypatch_healpy_import)
    from pyearthtools.pipeline.operations.xarray.remapping import HEALPix

    with pytest.warns(UserWarning):
        HEALPix()
