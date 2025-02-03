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


"""
Mixins to add functionality to `pyearthtools.data.indexes`

"""

from pyearthtools.data.indexes.utilities.mixins.index_repr import reprMixin
from pyearthtools.data.indexes.utilities.mixins.call_redirect import CallRedirectMixin
from pyearthtools.data.indexes.utilities.mixins.catalogs import CatalogMixin
