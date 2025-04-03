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

import json
import pytest
import tempfile
import os.path

from pyearthtools.data.save import jsonsave


def test_save(monkeypatch):
    def mock_dump(*args, **kwargs):
        return True

    monkeypatch.setattr(json, "dump", mock_dump)

    class MockThingError:
        """
        Used to return a non-string from search
        """

        def search(self, *args, **kwargs):
            return {}

    class MockThingWorks:
        """
        Used to return a valid temporary path which
        will get cleaned up
        """

        def __init__(self, usedir=None):
            self.usedir = usedir

        def search(self, *args, **kwargs):
            return os.path.join(self.usedir, "fakefile.txt")

    with pytest.raises(NotImplementedError):
        jsonsave.save("saveme", MockThingError())

    with tempfile.TemporaryDirectory() as tmpdir:

        mtw = MockThingWorks(tmpdir)
        path = jsonsave.save("saveme", mtw)
        assert path is not None
