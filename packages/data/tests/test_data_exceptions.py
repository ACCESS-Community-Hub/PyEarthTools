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

from pyearthtools.data import exceptions as e


def test_InvalidIndexError():

    # I tried setting samples up to parametrise the test but
    # it just didn't want to work that way

    samples = [
        (e.InvalidIndexError, KeyError),
        (e.InvalidDataError, KeyError),
        (e.DataNotFoundError, FileNotFoundError),
    ]

    for etype, ptype in samples:

        try:
            raise etype("testmessage")
        except etype as exc:
            assert exc.message == "testmessage"

        try:
            raise etype("testmessage", "a", "b")
        except Exception as exc:
            assert isinstance(exc, ptype)
            assert exc.message == "testmessageab"
            assert str(exc) == "testmessageab"


def test_run_and_catch():
    def makemistake():

        raise e.InvalidIndexError("Not here")

    e.run_and_catch_exception(makemistake)
