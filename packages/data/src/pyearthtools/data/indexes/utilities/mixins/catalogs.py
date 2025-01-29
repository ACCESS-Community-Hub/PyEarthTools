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


from __future__ import annotations

import warnings

from pathlib import Path

from pyearthtools.utils import initialisation
from pyearthtools.data.utils import parse_path


SUFFIX = "edi"


class CatalogMixin(initialisation.InitialisationRecordingMixin):
    def make_catalog(self, *args, **kwargs):
        warnings.warn("`make_catalog` is deprecated, please use `record_initialisation`.")
        return self.record_initialisation(*args, **kwargs)

    # @functools.wraps(initialisation.save)
    def save_index(self, path: Path | str | None = None, **kwargs):
        if path is not None:
            path = parse_path(path)
            if not path.suffix:
                path = path.with_suffix(SUFFIX)
            path = str(path)

        return initialisation.save(self, path, **kwargs)
