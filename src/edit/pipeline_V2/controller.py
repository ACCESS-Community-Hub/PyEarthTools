# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations
from abc import ABCMeta, abstractmethod

import functools
from typing import Any, ContextManager, Literal, Union, Optional
from pathlib import Path

import graphviz

from edit.data.indexes import Index

import edit.pipeline_V2
from edit.pipeline_V2.recording import PipelineRecordingMixin
from edit.pipeline_V2 import samplers, iterators, filters
from edit.pipeline_V2.step import PipelineStep
from edit.pipeline_V2.operation import Operation
from edit.pipeline_V2.exceptions import PipelineFilterException, ExceptionIgnoreContext


PIPELINE_TYPES = Union[Index, PipelineStep]
VALID_PIPELINE_TYPES = Union[PIPELINE_TYPES, tuple[PIPELINE_TYPES, ...], tuple[tuple, ...]]


__all___ = ["PipelineIndex", "Pipeline", "PipelineMod"]


def parse_to_graph_name(step: Union[Index, PipelineStep], parent: Optional[list[str]]) -> dict[str, Any]:
    """Parse step to useful name and attrs"""
    shape = "oval"
    if isinstance(step, Index):
        shape = "rect"
    # elif parent is not None and len(parent) > 1:
    #     shape = 'triangle'
    return {"label": step.__class__.__name__, "shape": shape}


class PipelineIndex(PipelineRecordingMixin, metaclass=ABCMeta):
    """Root PipelineIndex"""
    _edit_repr = {"ignore": ["args"], "expand_attr": ["Steps@flattened_steps"]}

    def __init__(
        self,
        *steps,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.steps = steps

    @property
    @abstractmethod
    def complete_steps(self) -> tuple:
        """Get steps in pipeline"""
        return self.steps

    @abstractmethod
    def __getitem__(self, idx):
        """Retrieve sample from pipeline"""
        pass

    def _get_tree(
        self, parent: Optional[list[str]] = None, graph: Optional[graphviz.Digraph] = None
    ) -> tuple[graphviz.Digraph, list[str]]:  # pragma: no cover
        """
        Get steps in a graphviz graph

        Args:
            parent (Optional[list[str]], optional):
                Parent elements of first layer in this `PipelineIndex`

        Returns:
            (tuple[graphviz.Digraph, list[str]]):
                Generated graph, elements to be parent of next step
        """

        import uuid

        graph = graph or graphviz.Digraph()
        prior_step = parent

        for step in self.steps:
            if isinstance(step, PipelineIndex):
                with graph.subgraph() as c:  # type: ignore
                    _, prior_step = step._get_tree(prior_step, graph=c)
            else:
                node_name = f"{step.__class__.__name__}_{uuid.uuid4()!s}"
                graph.node(node_name, **parse_to_graph_name(step, parent=prior_step))

                if prior_step is not None:
                    if isinstance(prior_step, list):
                        for p in prior_step:
                            graph.edge(p, node_name)
                    else:
                        graph.edge(prior_step, node_name)
                prior_step = [node_name]

        prior_step = prior_step or []
        return graph, prior_step


class Pipeline(PipelineIndex):
    """
    Core of `edit.pipeline_V2`, 

    Provides a way to set a sequence of operations to be applied to samples / data retrieved from `edit.data`.

    Examples
    >>> edit.pipeline_V2.Pipeline(
            edit.data.download.cds.ERA5('tcwv'),
            edit.pipeline_V2.operations.xarray.conversion.ToNumpy(),
    )
    """
    _sampler: samplers.Sampler
    _iterator: Optional[iterators.Iterator]
    _steps: tuple[Union[Index, PipelineStep, PipelineIndex, tuple[VALID_PIPELINE_TYPES, ...]], ...]
    _exceptions_to_ignore: Optional[tuple[Exception, ...]]

    def __init__(
        self,
        *steps: Union[
            VALID_PIPELINE_TYPES,
            PipelineIndex,
            PipelineMod,
            tuple[Union[VALID_PIPELINE_TYPES, Literal["map", "map_copy"]], ...],
        ],
        iterator: Optional[Union[iterators.Iterator, tuple[iterators.Iterator, ...]]] = None,
        sampler: Optional[Union[samplers.Sampler, tuple[samplers.Sampler, ...]]] = None,
        exceptions_to_ignore: Optional[tuple[Exception, ...]] = None,
        **kwargs,
    ):
        """_summary_

        Args:
            iterator (Optional[Union[iterators.Iterator, tuple[iterators.Iterator, ...]]], optional): 
                _description_. Defaults to None.
            sampler (Optional[Union[samplers.Sampler, tuple[samplers.Sampler, ...]]], optional): 
                _description_. Defaults to None.
            exceptions_to_ignore (Optional[tuple[Exception, ...]], optional): 
                _description_. Defaults to None.
        """        
        super().__init__(*steps, **kwargs)
        self.record_initialisation()

        self.iterator = iterator
        self.sampler = sampler
        self._exceptions_to_ignore = exceptions_to_ignore

    @property
    def flattened_steps(self) -> tuple:
        """Flat tuple of steps contained within this `PipelineIndex`"""

        def flatten(to_flatten):
            if isinstance(to_flatten, (tuple, list)):
                if len(to_flatten) == 0:
                    return []
                first, rest = to_flatten[0], to_flatten[1:]
                return flatten(first) + flatten(rest)
            else:
                return [to_flatten]

        return tuple(flatten(self.complete_steps))
    
    @property
    def complete_steps(self) -> tuple:
        """Get all steps"""
        return_steps = list(self.steps)
        expanded_steps = []

        for step in return_steps:
            if isinstance(step, Pipeline):
                expanded_steps.extend(step.complete_steps)
            elif isinstance(step, PipelineIndex):
                expanded_steps.append(step.complete_steps)
            else:
                expanded_steps.append(step)

        return tuple(expanded_steps)

    @property
    def steps(
        self,
    ) -> tuple[Union[VALID_PIPELINE_TYPES, PipelineIndex, tuple[VALID_PIPELINE_TYPES, ...]], ...]:
        """Steps of pipeline"""
        return self._steps

    @steps.setter
    def steps(
        self,
        val: tuple[
            Union[VALID_PIPELINE_TYPES, PipelineMod, tuple[Union[VALID_PIPELINE_TYPES, PipelineMod], ...]],
            ...,
        ],
    ):
        steps_list: list = []

        #TODO add ability to directly use transforms

        for v in val:
            if isinstance(v, (list, tuple)):
                steps_list.append(edit.pipeline_V2.branching.PipelineBranchPoint(*(i for i in v)))  # type: ignore
                continue
            elif isinstance(v, PipelineMod):
                v.set_steps(tuple(i for i in steps_list))
                steps_list = [v]
            else:
                steps_list.append(v)
        self._steps = tuple(steps_list)  # type: ignore

    @property
    def iterator(self):
        """Iterator of `Pipeline`"""
        return self._iterator

    @iterator.setter
    def iterator(self, val: Optional[Union[iterators.Iterator, tuple[iterators.Iterator, ...]]]):
        """
        Set iterator for `Pipeline`

        Args:
            val (Union[None, iterators.Iterator, tuple[iterators.Iterator, ...]]):
                Iterators, if is a tuple will create a `iterator.SuperIterator`
                which run one after each other.
        """

        if isinstance(val, tuple):
            val = iterators.SuperIterator(*val)
        self._iterator = val

    @property
    def sampler(self):
        """Sampler of `Pipeline`"""
        return self._sampler

    @sampler.setter
    def sampler(self, val: Optional[Union[samplers.Sampler, tuple[samplers.Sampler, ...]]]):
        """
        Set sampler for `Pipeline`

        Args:
            val (Optional[Union[samplers.Sampler, tuple[samplers.Sampler, ...]]]):
                Samplers, if is a tuple will create a `samplers.SuperSampler`
                which run one after each other.
        """
        if val is None:
            val = samplers.Default()
        elif isinstance(val, tuple):
            val = samplers.SuperSampler(*val)
        self._sampler = val

    def _get_initial_sample(self, idx: Any):
        """Get sample from first pipeline step"""
        if len(self.steps) == 0:
            raise ValueError("Cannot get data if no steps are given.")
        initial_step = self.steps[0]

        if not isinstance(initial_step, (PipelineIndex, Index)):
            raise TypeError(
                f"Cannot retrieve data from object if it is not an `edit.data.Index`. Found {type(initial_step).__qualname__}"
            )
        return initial_step[idx]

    def __getitem__(self, idx: Any):
        """Retrieve from pipeline at `idx`"""
        sample = self._get_initial_sample(idx)

        for step in self.steps[1:]:
            if not isinstance(step, PipelineStep):
                raise TypeError(f"When iterating through pipeline steps, found a {type(step)} which cannot be parsed.")
            sample = step(sample)
        return sample

    def undo(self, sample):
        """Undo `Pipeline` on `sample`"""
        for i, step in enumerate(self.steps[::-1]):
            if i == (len(self.steps) - 1) and (
                not isinstance(step, PipelineStep) and isinstance(step, (Index, PipelineIndex))
            ):
                # Remove last step on undo path if not PipelineStep, likely to be Index
                continue
            if not isinstance(step, PipelineStep):
                raise TypeError(f"When iterating through pipeline steps, found a {type(step)} which cannot be parsed.")
            elif isinstance(step, Operation):
                sample = step.undo(sample)
            else:
                sample = step(sample)
        return sample

    @property
    def iteration_order(self) -> tuple[Any, ...]:
        """Get ordering from `iterator`"""
        if self.iterator is None:
            raise ValueError("Cannot iterate over pipeline if iterator is not set.")
        return tuple(i for i in self.iterator)

    def __len__(self):
        """Length without any filtering applied"""
        return len(self.iteration_order)

    def __iter__(self):
        """Iterate over `Pipeline`, requires `iterator` to be set."""
        if self.iterator is None:
            raise ValueError("Cannot iterate over pipeline if iterator is not set.")
        sampler = self.sampler.generator()

        def check(obj):
            return obj is not None and not isinstance(obj, samplers.EmptyObject)

        next(sampler)
        filter_count: ContextManager[None] = filters.FilterWarningContext()
        exception_count: ContextManager[None] = ExceptionIgnoreContext(self._exceptions_to_ignore or tuple())

        for idx in self.iterator:
            sample = None
            with exception_count:
                try:
                    with filter_count:
                        sample = self[idx]
                except PipelineFilterException:
                    continue
            try:
                if isinstance(sample, iterators.IterateResults):
                    for sub_sample in sample.iterate_over_object():
                        sub_sample = sampler.send(sub_sample)
                        if check(sub_sample):
                            yield sub_sample
                else:
                    sample = sampler.send(sample)
                    if check(sample):
                        yield sample
            except StopIteration:
                break

        for remaining in sampler:
            if check(remaining):
                yield remaining


    def step(
        self, id: Union[str, int], limit: Optional[int] = -1
    ) -> Union[Index, Pipeline, Operation, tuple[Union[Index, Pipeline, Operation], ...]]:
        """Get step correspondant to `id`

        If `str` flattens steps and retrieves the first `limit` found,
        otherwise if `int`, gets step at the `idx`

        If `limit` is None, give back first found not in tuple, or if -1 return all.
        """
        if isinstance(id, str):
            matches = []
            for step in self.flattened_steps:
                if id == step.__class__.__name__:
                    if limit is None:
                        return step
                    matches.append(step)
                    if not limit == -1 and len(matches) >= limit:
                        return tuple(matches)

            if len(matches) > 0:
                return tuple(matches)

        elif isinstance(id, int):
            return self.complete_steps[id]

        raise TypeError(f"Cannot find step for {id!r}.")

    def __add__(self, other: Pipeline):
        """
        Combine pipelines

        Will set `self` steps first then `other`.

        But if other init kwargs were set, take from `other` if given.
        """
        if not isinstance(other, Pipeline):
            return NotImplemented

        init = dict(self.initialisation)
        other_init = dict(other.initialisation)

        args = (*init.pop("__args", []), *other_init.pop("__args", []))

        new_init = dict(init)
        new_init.update({key: val for key, val in other_init.items() if val is not None})

        return Pipeline(*args, **new_init)

    def graph(self) -> graphviz.Digraph:
        """Get graphical view of Pipeline"""
        return self._get_tree(parent=None, graph=graphviz.Digraph())[0]

    def save(self, path: Optional[Union[str, Path]] = None) -> Union[str, None]:
        """
        Save `Pipeline`

        Args:
            path (Optional[Union[str, Path]], optional):
                File to save to. If not given return save str. Defaults to None.

        Returns:
            (Union[str, None]):
                If `path` is None, `pipeline` in save form else None.
        """
        return edit.pipeline_V2.save(self, path)

    def _ipython_display_(self):
        """Override for repr of `Pipeline`, shows initialisation arguments and graph"""
        from IPython.core.display import display, HTML

        display(HTML(self._repr_html_()))
        if len(self.flattened_steps) > 1:
            display(HTML("<h2>Graph</h2>"))
            display(self.graph())


class PipelineMod(PipelineIndex):
    """Variant of Pipeline to be subclassed for modification"""

    _edit_repr = {"ignore": ["args"]}
    _steps: tuple[Union[Index, PipelineStep, PipelineIndex, tuple[VALID_PIPELINE_TYPES, ...]], ...]

    def __init__(self):
        super().__init__()

    def set_steps(self, steps: tuple[Union[Index, PipelineStep, PipelineIndex, tuple[VALID_PIPELINE_TYPES, ...]], ...]):
        """Set steps of this `PipelineMod`"""
        if len(steps) == 0:
            raise Exception()
        self._steps = steps

    def parent_pipeline(self) -> Pipeline:
        """Get parent pipeline of this `PipelineMod`, will not include self"""
        pipe = Pipeline()
        pipe._steps = self._steps
        return pipe

    def as_pipeline(self) -> Pipeline:
        """Get `PipelineMod` as full pipeline, will include self"""
        pipe = self.parent_pipeline()
        pipe._steps = (*pipe._steps, self)
        return pipe
    
    @functools.wraps(PipelineIndex._get_tree)
    def _get_tree(
        self, parent: Optional[list[str]] = None, graph: Optional[graphviz.Digraph] = None
    ) -> tuple[graphviz.Digraph, list[str]]:  # pragma: no cover
        """Override for `graph` creation"""
        import uuid

        graph = graph or graphviz.Digraph()
        graph, prior_step = self.parent_pipeline()._get_tree(parent, graph=graph)

        node_name = f"{self.__class__.__name__}_{uuid.uuid4()!s}"
        graph.node(node_name, self.__class__.__name__)

        if isinstance(prior_step, list):
            for p in prior_step:
                graph.edge(p, node_name)
        else:
            graph.edge(prior_step, node_name)
        return graph, [node_name]

    @property
    def complete_steps(self) -> tuple:
        """Get all steps"""
        return_steps = list(self.parent_pipeline().complete_steps)
        expanded_steps = []

        for step in return_steps:
            if isinstance(step, PipelineIndex):
                step = step.complete_steps
                expanded_steps.extend(step)
            expanded_steps.append(step)

        # Append self
        expanded_steps.append(self)

        return tuple(expanded_steps)
