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


import xarray as xr

from pyearthtools.pipeline.graph import Graphed
from pyearthtools.pipeline.operation import PipelineStep


def find_shape(obj):
    if hasattr(obj, "shape"):
        return obj.shape

    if isinstance(obj, xr.Dataset):
        return tuple(obj[d].shape for d in obj.data_vars)


class Marker(PipelineStep):
    """
    Marker in a pipeline

    Useful for graph notes.
    """

    def __init__(self, text: str, shape: str = "note", print: bool = False, print_shape: bool = False):
        """
        Pipeline marker

        Args:
            text (str):
                Text to display in graph
            shape (str, optional):
                Shape for graph. Defaults to 'note'.
            print (bool, optional):
                Whether to print `sample` when running. Defaults to False.
        """
        super().__init__()
        self.record_initialisation()

        self.text = text
        self.shape = shape
        self._print = print
        self._print_shape = print_shape

    def run(self, sample):
        if self._print:
            to_print = sample if not self._print_shape else find_shape(sample)
            print(f"At marker {self.text!r} sample was:\n{to_print}")
        return sample

class Empty(PipelineStep, Graphed):
    """Empty Operation to do nothing to the data"""
    def run(self, sample):
        return sample
    
    def _get_tree(
        self, parent, graph = None
    ):
        """
        Get graphviz graph

        Args:
            parent (Optional[list[str]], optional):
                Parent elements of first layer in this `graph`
            graph (Optional[graphviz.Digraph]):
                Subgraph to build in. Defaults to None.

        Returns:
            (tuple[graphviz.Digraph, list[str]]):
                Generated graph, elements to be parent of next step
        """
        return graph, parent