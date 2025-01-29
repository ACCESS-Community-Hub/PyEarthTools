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



from abc import abstractmethod
from typing import Any, Optional

import graphviz

from pyearthtools.data import Index
import pyearthtools.pipeline


def format_graph_node(obj, parent: Optional[list[str]]) -> dict[str, Any]:
    """Parse step to useful name and attrs"""

    shape = "oval"
    if isinstance(obj, Index):
        shape = "rect"

    # elif parent is not None and len(parent) > 1:
    #     shape = 'triangle'

    last_module = str(obj.__module__).replace(f"{type(obj).__name__}", "").split(".")[-1]
    obj_name = f"{last_module}.{type(obj).__name__}".removeprefix(".")

    if isinstance(obj, pyearthtools.pipeline.Marker):
        obj_name = obj.text
        shape = obj.shape or "note"

    return {"label": obj_name, "shape": shape}


class Graphed:
    """
    Implement graph visualisation
    """

    @abstractmethod
    def _get_tree(
        self, parent: Optional[list[str]] = None, graph: Optional[graphviz.Digraph] = None
    ) -> tuple[graphviz.Digraph, list[str]]:  # pragma: no cover
        """
        Get  graphviz graph

        Args:
            parent (Optional[list[str]], optional):
                Parent elements of first layer in this `graph`
            graph (Optional[graphviz.Digraph]):
                Subgraph to build in. Defaults to None.

        Returns:
            (tuple[graphviz.Digraph, list[str]]):
                Generated graph, elements to be parent of next step
        """
        ...

    def graph(self) -> graphviz.Digraph:
        """Get graphical view of Pipeline"""
        return self._get_tree(parent=None, graph=graphviz.Digraph())[0]
