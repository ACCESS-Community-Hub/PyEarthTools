# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.
"""
Pipeline Controller
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod

from typing import Any, ContextManager, Literal, Union, Optional, Type, overload
from pathlib import Path
import functools
import logging

import builtins
import graphviz

import edit.utils

from edit.data.indexes import Index
from edit.data.transforms import Transform, TransformCollection

import edit.pipeline
from edit.pipeline.recording import PipelineRecordingMixin
from edit.pipeline import samplers, iterators, filters
from edit.pipeline.step import PipelineStep
from edit.pipeline.operation import Operation
from edit.pipeline.exceptions import PipelineFilterException, ExceptionIgnoreContext
from edit.pipeline.validation import filter_steps
from edit.pipeline.graph import Graphed, format_graph_node


PIPELINE_TYPES = Union[Index, PipelineStep, Transform, TransformCollection]
VALID_PIPELINE_TYPES = Union[PIPELINE_TYPES, tuple[PIPELINE_TYPES, ...], tuple[tuple, ...]]

__all___ = ["Pipeline", "PipelineIndex"]

LOG = logging.getLogger("edit.pipeline")


class PipelineIndex(PipelineRecordingMixin, metaclass=ABCMeta):
    """
    Special Pipeline step which holds a copy of the pipeline above it.

    Must implement `__getitem__` to modify data retrieval flow,
    and if changes are needed on `undo`, override `undo_func`.
    """

    _edit_repr = {"ignore": ["args"]}
    _steps: tuple[
        Union[Index, PipelineStep, _Pipeline, PipelineIndex, VALID_PIPELINE_TYPES, tuple[VALID_PIPELINE_TYPES, ...]],
        ...,
    ]
    _partial_parent: functools.partial

    def set_parent_record(
        self,
        steps: tuple[
            Union[
                Index, PipelineStep, _Pipeline, PipelineIndex, VALID_PIPELINE_TYPES, tuple[VALID_PIPELINE_TYPES, ...]
            ],
            ...,
        ],
        iterator: Optional[iterators.Iterator] = None,
        sampler: Optional[samplers.Sampler] = None,
    ):
        """Set record of the parent of this `PipelineIndex`"""
        self._partial_parent = functools.partial(Pipeline, iterator=iterator, sampler=sampler)
        self._steps = steps

    def parent_pipeline(self) -> Pipeline:
        """Get parent pipeline of this `PipelineIndex`, will not include self"""
        return self._partial_parent(*self._steps)

    def as_pipeline(self) -> Pipeline:
        """Get `PipelineIndex` as full pipeline, will include self"""
        return self._partial_parent(*self._steps, self)

    @abstractmethod
    def __getitem__(self, idx):
        """Retrieve sample from PipelineIndex"""
        return self.parent_pipeline()[idx]

    def undo_func(self, sample: Any) -> Any:
        """Run `undo`. If the child class needs to make modifications."""
        return sample


class _Pipeline(PipelineRecordingMixin, Graphed, metaclass=ABCMeta):
    """Root Pipeline object"""

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
            if isinstance(step, Graphed):
                with graph.subgraph() as c:  # type: ignore
                    _, prior_step = step._get_tree(prior_step, graph=c)  # type: ignore
            else:
                node_name = f"{step.__class__.__name__}_{uuid.uuid4()!s}"
                graph.node(node_name, **format_graph_node(step, parent=prior_step))

                if prior_step is not None:
                    if isinstance(prior_step, list):
                        for p in prior_step:
                            graph.edge(p, node_name)
                    else:
                        graph.edge(prior_step, node_name)
                prior_step = [node_name]

        prior_step = prior_step or []
        return graph, prior_step


class Pipeline(_Pipeline, Index):
    """
    Core of `edit.pipeline`,

    Provides a way to set a sequence of operations to be applied to samples / data retrieved from `edit.data`.

    ## Example:
    ```python
    pipeline = edit.pipeline.Pipeline(
        edit.data.download.cds.ERA5('tcwv'),
        edit.pipeline.operations.xarray.conversion.ToNumpy()
    )
    pipeline['2000-01-01T00]
    ```

    ## Usage:
    A `Pipeline` can be used in three primary ways,

    | Type | Code|
    |----|----|
    | Direct Indexing | `pipeline[idx]` |
    | Iteration | `for i in pipeline` |
    | Applying | `pipeline.apply` |

    """

    _sampler: samplers.Sampler
    _iterator: Optional[iterators.Iterator]
    _steps: tuple[Union[Index, PipelineStep, _Pipeline, tuple[VALID_PIPELINE_TYPES, ...]], ...]
    _exceptions_to_ignore: Optional[tuple[Type[Exception], ...]]

    _edit_repr = {"ignore": ["args"], "expand_attr": ["Steps@flattened_steps"]}

    @property
    def _desc_(self) -> dict[str, Any]:
        return {"singleline": "`edit.pipeline` Data Pipeline"}

    def __init__(
        self,
        *steps: Union[
            VALID_PIPELINE_TYPES,
            _Pipeline,
            PipelineIndex,
            tuple[Union[VALID_PIPELINE_TYPES, Literal["map", "map_copy"]], ...],
        ],
        iterator: Optional[Union[iterators.Iterator, tuple[iterators.Iterator, ...]]] = None,
        sampler: Optional[Union[samplers.Sampler, tuple[samplers.Sampler, ...]]] = None,
        exceptions_to_ignore: Optional[tuple[Union[str, Type[Exception]], ...]] = None,
        **kwargs,
    ):
        """
        Create `Pipeline` of operations to run on samples of data.

        The `steps` will be run in order of inclusion.


        ## Branches

        If a tuple within the `steps` is encountered, it will be interpreted as a `BranchingPoint`,
        with each element in the `tuple` a seperate `Pipeline` of it's own right.
        Therefore to have a `BranchingPoint` with each branch containing multiple steps, a nested `tuple` is needed.

        E.g. # Pseudocode
        >>> Pipeline(
                Index,
                (Operation_1, Operation_2)
            )
        This will cause samples to be retrieved from `Index` and each of the operations run on the `sample`.
        The result will follow the form of:
            `(Operation_1 on Index, Operation_2 on Index)`

        If a branch consists of multiple operations, the nested tuples must be used.

        E.g. # Pseudocode
        >>> Pipeline(
                Index,
                ((Operation_1, Operation_1pt2), Operation_2)
            )
        This will cause samples to be retrieved from `Index` and each of the operations run on the `sample`.
        The result will follow the form of:
            `(Operation_1 + Operation_1pt2 on Index, Operation_2 on Index)`

        A `BranchingPoint` by default will cause each branch to be run seperately, and a tuple returned with the results of each branch.
        However, if 'map' is included in the `BranchingPoint` tuple, it will be mapped across elements in the incoming sample.

        ### Mapping
        E.g. # Pseudocode
        >>> Pipeline(
                Index,
                ((Operation_1, Operation_1pt2), Operation_2, 'map')
            )
        This will cause samples to be retrieved from `Index` and the operations to be mapped to the `sample`.
        The result will follow the form of:
            `(Operation_1 + Operation_1pt2 on Index[0], Operation_2 on Index[1])`

        'map_copy' can be used to copy the branch to the number of elements in the sample without having to
        manually specify each branch.

        ### Indexes in Branches
        Indexes can also be included in branches, which behaviour as expected, where the sample is retrieved rather than operations applied.
        E.g. #Pseudocode
        >>> Pipeline(
            Index,
            (Operation_1, Operation_2, Index)
        )

        ## Transforms
        Transforms from `edit.data` can be added directly inline in a pipeline, and will be applied on the forward pass.
        If they need to be applied on `undo`, or on both see, `edit.pipeline.operations.Transforms`


        Args:
            *steps (VALID_PIPELINE_TYPES, tuple[VALID_PIPELINE_TYPES]):
                Steps of the pipeline. Can include tuples to refer to branches.

            iterator (Optional[Union[iterators.Iterator, tuple[iterators.Iterator, ...]]], optional):
                `Iterator` to use to retrieve samples when the `Pipeline` is being iterated over. Defaults to None.
            sampler (Optional[Union[samplers.Sampler, tuple[samplers.Sampler, ...]]], optional):
                `Sampler` to use to sample the samples when iterating. If not given will yield all samples.
                Can be used to randomly sample, drop out and more
                Defaults to None.
            exceptions_to_ignore (Optional[tuple[Union[str, Type[Exception]], ...]], optional):
                Which exceptions to ignore when iterating. Defaults to None.
        """
        self.iterator = iterator
        self.sampler = sampler

        super().__init__(*steps, **kwargs)
        self.record_initialisation()

        self.exceptions_to_ignore = exceptions_to_ignore

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
        expanded_steps: list[Any] = []

        for step in return_steps:
            if isinstance(step, Pipeline):
                expanded_steps.extend(step.complete_steps)
            elif isinstance(step, _Pipeline):
                expanded_steps.append(step.complete_steps)
            else:
                expanded_steps.append(step)

        return tuple(expanded_steps)

    @property
    def steps(
        self,
    ) -> tuple[Union[VALID_PIPELINE_TYPES, _Pipeline, tuple[VALID_PIPELINE_TYPES, ...]], ...]:
        """Steps of pipeline"""
        return self._steps

    @steps.setter
    def steps(
        self,
        val: tuple[
            Union[VALID_PIPELINE_TYPES, PipelineIndex, tuple[Union[VALID_PIPELINE_TYPES, PipelineIndex], ...]],
            ...,
        ],
    ):
        steps_list: list = []

        filter_steps(
            val if isinstance(val, tuple) else (val,),
            (
                tuple,
                Index,
                _Pipeline,
                PipelineIndex,
                PipelineStep,
                Transform,
                TransformCollection,
                edit.pipeline.branching.PipelineBranchPoint,
            ),
            # invalid_types=(Filter,),
            responsible="Pipeline",
        )

        for v in val:
            if isinstance(v, (list, tuple)):
                steps_list.append(edit.pipeline.branching.PipelineBranchPoint(*(i for i in v)))  # type: ignore
                continue
            elif isinstance(v, PipelineIndex):
                v.set_parent_record(tuple(i for i in steps_list), iterator=self.iterator, sampler=self.sampler)
                steps_list.append(v)
                # steps_list = [v]
            elif isinstance(v, Pipeline):
                steps_list.extend(v.steps)
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
        if not isinstance(val, iterators.Iterator) and val is not None:
            raise TypeError(f"Iterator must be a `edit.pipeline.Iterator`, not {type(val)}.")
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
        if not isinstance(val, samplers.Sampler):
            raise TypeError(f"Sampler must be a `edit.pipeline.Sampler`, not {type(val)}.")
        self._sampler = val

    @property
    def exceptions_to_ignore(self):
        """Sampler of `Pipeline`"""
        return self._exceptions_to_ignore

    @exceptions_to_ignore.setter
    def exceptions_to_ignore(self, val: Optional[Union[str, Type[Exception], tuple[Union[str, Type[Exception]], ...]]]):
        """
        Set exceptions_to_ignore for `Pipeline`

        Args:
            val (Union[str, Exception, tuple[Union[str, Exception], ...]]):
                Exceptions to ignore.
        """

        def parse(v) -> Type[Exception]:
            if isinstance(v, str):
                return getattr(builtins, v)
            elif isinstance(v, Type):
                return v
            raise TypeError(f"Cannot use {v} as an exception class.")

        if val is None:
            pass
        else:
            val = (val,) if not isinstance(val, tuple) else val
            val = tuple(map(parse, val))

        self._exceptions_to_ignore = val  # type: ignore

    def has_source(self) -> bool:
        """Determine if this `Pipeline` contains a source of data, or is just a sequence of operations."""
        if isinstance(self._steps[0], (PipelineIndex, Index)):
            return True
        if isinstance(self._steps[0], edit.pipeline.branching.PipelineBranchPoint):
            return all(map(lambda x: x.has_source(), self._steps[0].sub_pipelines))
        return False

    def _get_initial_sample(self, idx: Any) -> tuple[Any, int]:
        """
        Get sample from first pipeline step or first working back index

        Returns:
            (tuple[Any, int]):
                Sample, index of step used to retrieve
        """
        if len(self.steps) == 0:
            raise ValueError("Cannot get data if no steps are given.")

        for index, step in enumerate(self.steps[::-1]):
            if isinstance(step, PipelineIndex):
                LOG.debug(f"Getting initial sample from {step} at {idx}")
                return step[idx], len(self.steps) - (index + 1)

        if isinstance(self.steps[0], (_Pipeline, Index)):
            LOG.debug(f"Getting initial sample from {self.steps[0]} at {idx}")
            return self.steps[0][idx], 0

        raise TypeError(f"Cannot find an `Index` to get data from. Found {type(self.steps[0]).__qualname__}")

    def __getitem__(self, idx: Any):
        """Retrieve from pipeline at `idx`"""
        sample, step_index = self._get_initial_sample(idx)
        LOG.debug(f"Call pipeline __getitem__ for {idx = }")

        for step in self.steps[step_index + 1 :]:
            if not isinstance(step, (Pipeline, PipelineStep, Transform, TransformCollection)):
                raise TypeError(f"When iterating through pipeline steps, found a {type(step)} which cannot be parsed.")
            LOG.debug(f"Apply step upon sample. {step}")

            if isinstance(step, Pipeline):
                sample = step.apply(sample)
            elif isinstance(step, edit.pipeline.branching.PipelineBranchPoint):
                with edit.utils.context.ChangeValue(step, "_current_idx", idx):
                    sample = step.apply(sample)
            else:
                sample = step(sample)  # type: ignore
        return sample

    def __call__(self, obj):
        if isinstance(obj, str):
            return self[obj]
        return self.apply(obj)

    def get(self, idx):
        """Get `idx` from `Pipeline`."""
        return self[idx]

    def apply(self, sample):
        """
        Apply pipeline to `sample`

        `Pipeline` should only consist of `PipelineStep`'s and `Transforms`, as `Indexes` cannot be applied,
        """
        for step in self.steps:
            if not isinstance(
                step, (PipelineStep, Transform, TransformCollection, edit.pipeline.branching.PipelineBranchPoint)
            ):
                raise TypeError(f"When iterating through pipeline steps, found a {type(step)} which cannot be parsed.")
            if isinstance(step, Pipeline):
                sample = step.apply(sample)  # type: ignore
            else:
                sample = step(sample)  # type: ignore
        return sample

    @property
    def get_and_catch(pipeline_self):
        """Get indexable object like pipeline which will ignore any expections known to be ignored."""
        if pipeline_self._exceptions_to_ignore is None:
            return pipeline_self

        class catch:
            def __getitem__(self, idx: Any):
                try:
                    return pipeline_self[idx]
                except pipeline_self._exceptions_to_ignore:  # type: ignore
                    return None

        return catch()

    def undo(self, sample):
        """
        Undo `Pipeline` on `sample`.

        Reverses the steps and operations applied to the `sample`.
        Ideally this should result in the `sample` looking identical to the initial data.

        Examples:
        ```python
            pipeline = (
                Index,
                Operation1,
            )
            pipeline[1]
            # Data
            pipeline.undo(pipeline[1])

        """
        for i, step in enumerate(reversed(self.steps)):
            if i == (len(self.steps) - 1) and (not isinstance(step, PipelineStep) and isinstance(step, Index)):
                # Remove last step on undo path if not PipelineStep, likely to be Index
                continue
            if not isinstance(step, (Pipeline, PipelineIndex, PipelineStep, Transform, TransformCollection)):
                raise TypeError(f"When iterating through pipeline steps, found a {type(step)} which cannot be parsed.")
            elif isinstance(step, Operation):
                sample = step.undo(sample)
            elif isinstance(step, PipelineIndex):
                sample = step.undo_func(sample)
                # sample = step.parent_pipeline().undo(sample)
            elif isinstance(step, Pipeline):
                sample = step.undo(sample)
            elif isinstance(step, edit.pipeline.branching.StopUndo):
                break
            elif isinstance(step, (Transform, TransformCollection)):
                pass
            else:
                sample = step(sample)  # type: ignore
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

    @overload
    def step(self, id: Union[str, int, Type[Any], Any], limit: None) -> Union[Index, Pipeline, Operation]: ...

    @overload
    def step(
        self, id: Union[str, int, Type[Any], Any], limit: int
    ) -> tuple[Union[Index, Pipeline, Operation], ...]: ...

    def step(
        self, id: Union[str, int, Type[Any], Any], limit: Optional[int] = -1
    ) -> Union[Union[Index, Pipeline, Operation], tuple[Union[Index, Pipeline, Operation], ...]]:
        """Get step correspondent to `id`

        If `str` flattens steps and retrieves the first `limit` found,
        otherwise if `int`, gets step at the `idx`

        If `limit` is None, give back first found not in tuple, or if -1 return all.

        Raises:
            ValueError:
                If cannot find `id` in self.
        """
        if isinstance(id, Type):
            id = id.__name__
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

        raise ValueError(f"Cannot find step for {id!r}.")

    @property
    def as_steps(pipeline_self):
        """
        Get an indexable object to recreate pipeline with a subset of steps.

        >>> pipeline.as_steps[:5]

        """
        steps = list(pipeline_self.complete_steps)

        class StepIndexer:
            def __getitem__(self, idx):
                if isinstance(idx, int):
                    return Pipeline(*steps[:idx])
                elif isinstance(idx, str):
                    return Pipeline(*steps[: pipeline_self.index(idx)])
                return Pipeline(*steps[idx])

        return StepIndexer()

    def index(self, id: Union[str, Type]) -> int:
        """
        Get index of `id` in Pipeline.
        """
        if isinstance(id, Type):
            id = id.__name__
        step_names = list(map(lambda x: str(x.__class__.__name__), self.complete_steps))

        if id in step_names:
            return step_names.index(id)
        raise ValueError(f"{id!r} is  not in Pipeline. {step_names}")

    def __contains__(self, id: Union[str, Type[Any]]) -> bool:
        try:
            self.step(id)  # type: ignore
            return True
        except ValueError:
            return False

    def __add__(self, other: Union[_Pipeline, PipelineIndex, PipelineStep]) -> Pipeline:
        """
        Combine pipelines

        Will set `self` steps first then `other`.

        But if other init kwargs were set, take from `other` if given.
        """

        if isinstance(other, Pipeline):
            init = dict(self.initialisation)
            other_init = dict(other.initialisation)

            args = (*init.pop("__args", []), *other_init.pop("__args", []))

            new_init = dict(init)
            new_init.update({key: val for key, val in other_init.items() if val is not None})

            return Pipeline(*args, **new_init)

        elif isinstance(other, (PipelineIndex, PipelineStep)):
            init = dict(self.initialisation)
            args = (*init.pop("__args", []), other)
            return Pipeline(*args, **init)

        return NotImplemented

    def save(self, path: Optional[Union[str, Path]] = None, only_steps: bool = False) -> Union[str, None]:
        """
        Save `Pipeline`

        Args:
            path (Optional[Union[str, Path]], optional):
                File to save to. If not given return save str. Defaults to None.
            only_steps (bool, optional):
                Save only steps of the pipeline, dropping iterator, sampler, and exceptions_to_ignore.

        Returns:
            (Union[str, None]):
                If `path` is None, `pipeline` in save form else None.
        """
        if only_steps:
            return edit.pipeline.save(Pipeline(*self.complete_steps), path)  # type: ignore
        return edit.pipeline.save(Pipeline(*self.complete_steps, iterator=self.iterator, sampler=self.sampler, exceptions_to_ignore=self._exceptions_to_ignore), path)  # type: ignore

    def _ipython_display_(self):
        """Override for repr of `Pipeline`, shows initialisation arguments and graph"""
        from IPython.core.display import display, HTML

        display(HTML(self._repr_html_()))

        if len(self.flattened_steps) > 1 and edit.utils.config.get("pipeline.repr.show_graph"):
            display(HTML("<h2>Graph</h2>"))
            display(self.graph())

    @classmethod
    def sample(
        cls,
        variables: Optional[list[str]] = None,
        iterator: Optional[Union[iterators.Iterator, tuple[iterators.Iterator, ...]]] = None,
        sampler: Optional[Union[samplers.Sampler, tuple[samplers.Sampler, ...]]] = None,
    ):
        """
        Simple sample Pipeline for testing.
        """
        import edit.data
        import edit.pipeline

        return edit.pipeline.Pipeline(
            edit.data.archive.ERA5.sample() if variables is None else edit.data.archive.ERA5(variables),  # type: ignore
            edit.pipeline.operations.xarray.conversion.ToNumpy(),
            iterator=iterator,
            sampler=sampler,
        )
