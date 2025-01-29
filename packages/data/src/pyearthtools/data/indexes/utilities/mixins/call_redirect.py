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


class CallRedirectMixin:
    """
    Provide overrides for `__func__`'s that redirect to `__call__`
    """

    def __call__(self, *args):
        raise NotImplementedError

    def __matmul__(self, key: Any):
        """@ accessor

        Expands tuple or lists passed
        """
        if isinstance(key, (list, tuple)):
            return self.__call__(*key)

        elif isinstance(key, dict):
            return self.__call__(**key)

        return self.__call__(key)

    def __getitem__(self, idx: Any):
        """[] accessor"""
        return self.__call__(idx)
