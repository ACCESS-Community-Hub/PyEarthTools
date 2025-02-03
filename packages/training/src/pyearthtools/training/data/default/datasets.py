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

from pyearthtools.pipeline import Pipeline


class BaseDefault:
    def __init__(self, pipeline: Pipeline, **kwargs) -> None:
        super().__init__(**kwargs)
        self._pipeline = pipeline

    def save(self, *args, **kwargs):
        return self._pipeline.save(*args, **kwargs)

    @property
    def iterator(self):
        return self._pipeline.iterator

    @iterator.setter
    def iterator(self, val):
        self._pipeline.iterator = val

    def __len__(self):
        return len(self._pipeline.iteration_order)


class IterableDataset(BaseDefault):
    """
    Iterate over pipeline
    """

    def __iter__(self):
        for sample in self._pipeline:
            yield sample


class IndexableDataset(BaseDefault):
    """Mapped Dataset of `Pipeline`"""

    def __getitem__(self, idx):
        return self._pipeline[self._pipeline.iteration_order[idx]]
