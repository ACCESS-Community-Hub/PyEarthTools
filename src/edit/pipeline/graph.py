# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.


from abc import abstractmethod
from typing import Any, Optional

import graphviz

from edit.data import Index
import edit.pipeline


def format_graph_node(obj, parent: Optional[list[str]]) -> dict[str, Any]:
    """Parse step to useful name and attrs"""

    shape = "oval"
    if isinstance(obj, Index):
        shape = "rect"


    # elif parent is not None and len(parent) > 1:
    #     shape = 'triangle'

    last_module = str(obj.__module__).replace(f"{type(obj).__name__}", "").split(".")[-1]
    obj_name = f"{last_module}.{type(obj).__name__}".removeprefix(".")

    if isinstance(obj, edit.pipeline.Marker):
        obj_name = obj.text
        shape = obj.shape or 'note'

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
