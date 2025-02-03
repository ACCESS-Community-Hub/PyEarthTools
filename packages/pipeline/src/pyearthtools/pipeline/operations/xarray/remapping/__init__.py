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
Re-mapping tools for processing data on different coordinate projections.
"""

import warnings

try:
    from .healpix import HEALPix  # noqa: F401
except (ImportError, ModuleNotFoundError) as e:
    warnings.warn(f"Could not import `healpix` remapping due to {e}.", ImportWarning)
