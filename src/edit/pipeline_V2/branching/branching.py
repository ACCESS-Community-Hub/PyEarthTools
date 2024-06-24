# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations

from typing import Any, Literal, Optional, Union
import warnings

import graphviz

from edit.data.indexes import Index
from edit.data.transforms import Transform, TransformCollection

from edit.pipeline_V2.step import PipelineStep
from edit.pipeline_V2.controller import PipelineIndex, Pipeline, _Pipeline
from edit.pipeline_V2.operation import Operation
from edit.pipeline_V2.filters import Filter

from edit.pipeline_V2 import parallel
from edit.pipeline_V2.warnings import PipelineWarning
from edit.pipeline_V2.validation import filter_steps
from edit.pipeline_V2.exceptions import PipelineRuntimeError


def get_key_from_steps(key: str, steps: tuple[Any, ...]):
    result = False
    if key in steps:
        result = True
        l_steps = list(steps)
        l_steps.pop(l_steps.index(key))  # type: ignore
        steps = tuple(l_steps)
    return result, steps


def expand_pipeline(original: PipelineBranchPoint, length: int) -> list[Pipeline]:
    """
    Expand `PipelineBranchPoint` to size of `length`.

    Will use % to retrieve original, so if `original` has two pipelines and a length of
    4 needed, would be pipeline index correspondant to
    >>> [0,1,0,1]
    """
    new_pipeline: list[Pipeline] = []
    original_pipelines = original.sub_pipelines

    for i in range(length):
        sub_pipe = original_pipelines[i % len(original_pipelines)]
        if isinstance(sub_pipe, Pipeline):
            new_pipeline.append(sub_pipe.copy())

    return new_pipeline


class PipelineBranchPoint(_Pipeline, Operation):
    """
    Branch Point in a `Pipeline`.

    Can be anywhere in the pipeline, including at the start.

    Special keys can be provided as part of the tuple to customise behaviour:
        `map`:
            Will not branch, but map elements of the `sample` which should be a tuple,
            to the associated ordered branch. Requires samples and branches to be of the same length.
        `map_copy`:
            Acts like `map` but will make copies of the pipelines to match incoming sample.
            Once one sample has been seen, and all copies made, becomes a `map`.
            If multiple branches, will use % to create more, so if it has has two pipelines and a length of
            4 needed, would be pipeline index correspondant to
            >>> [0,1,0,1]

    """

    _map: bool = False
    _map_copy: bool = False
    _current_idx: Optional[int] = None
    sub_pipelines: list[Pipeline]

    _override_interface = ["Serial"]

    def __init__(
        self,
        *steps: Union[
            tuple[
                Union[Index, Pipeline, PipelineStep, Transform, TransformCollection, Literal["map", "map_copy"]], ...
            ],
            Index,
            PipelineStep,
            Pipeline,
            Transform,
            TransformCollection,
        ],
    ):
        super().__init__()  # type: ignore
        self.record_initialisation()

        __incoming_steps = []

        self._map_copy, steps = get_key_from_steps("map_copy", steps)
        self._map, steps = get_key_from_steps("map", steps)

        for i, sub in enumerate(steps):
            # if isinstance(sub, PipelineIndex):
            #     sub = sub.steps

            filter_steps(
                sub if isinstance(sub, tuple) else (sub,),
                (tuple, Index, Pipeline, PipelineIndex, PipelineStep, Transform, TransformCollection),
                # invalid_types=(Filter,),
                responsible="PipelineBranchPoint",
            )
            if not isinstance(sub, (list, tuple)):
                __incoming_steps.append((sub,))  # type: ignore
            else:
                __incoming_steps.append(sub)  # type: ignore

        self.sub_pipelines = list(map(lambda x: Pipeline(*x), __incoming_steps))

    def __getitem__(self, idx: Any) -> tuple:
        """Get result from each branch"""
        results = []
        for pipe in self.sub_pipelines:
            results.append(self.parallel_interface.submit(pipe.__getitem__, idx))

        return tuple(self.parallel_interface.collect(results))

    # def _steps_function(self, sample, steps: tuple[Pipeline], func_name: Literal['apply', 'undo']):
    #     """
    #     Run `func_name` across all steps in `steps` for `sample`
    #     """

    #     for step in steps:
    #         if not isinstance(step, (Pipeline, PipelineStep, Transform, TransformCollection)):
    #             raise TypeError(f"When iterating through pipeline steps, found a {type(step)} which cannot be parsed.")
    #         elif isinstance(step, (Pipeline, Operation)):
    #             sample = getattr(step, func_name)(sample)
    #         elif isinstance(step, (Transform, TransformCollection)):
    #             if func_name == 'undo':
    #                 pass
    #             sample = step(sample)
    #         else:
    #             sample = step(sample)  # Run other `PipelineStep`'s.
    #     return sample

    def apply(self, sample):
        """Apply each branch on the sample"""

        sub_samples = []

        if self._map or self._map_copy:
            if not isinstance(sample, tuple):
                raise PipelineRuntimeError(f"Cannot map sample to branches as it is not a tuple. {type(sample)}.")

            if any(pipe.has_source() for pipe in self.sub_pipelines):
                raise ValueError(
                    "When attempting to map pipelines to sample, found a Pipeline with a source of data. Cannot continue.\n",
                    (pipe for pipe in self.sub_pipelines if pipe.has_source()),
                )

            if not len(sample) == len(self.sub_pipelines):
                if self._map:
                    raise PipelineRuntimeError(
                        f"Cannot map sample to branches as length differ. {len(sample)} != {len(self.sub_pipelines)}."
                    )
                elif self._map_copy:
                    self.sub_pipelines = expand_pipeline(self, len(sample))

            for s, pipe in zip(sample, self.sub_pipelines):
                sub_samples.append(self.parallel_interface.submit(pipe.apply, s))
                # sub_samples.append(
                #     self.parallel_interface.submit(self._steps_function, s, steps=pipe.steps, func_name="apply")
                # )
        else:
            for sub_pipe in self.sub_pipelines:
                if sub_pipe.has_source():
                    if self._current_idx is None:
                        raise ValueError(
                            "Applying branchs to `sample` found a pipeline with source, but the `current_idx` was not set."
                        )
                    sub_samples.append(self.parallel_interface.submit(sub_pipe.__getitem__, self._current_idx))
                else:
                    samp = type(sample)(sample)
                    sub_samples.append(self.parallel_interface.submit(sub_pipe.apply, samp))

                # steps = sub_pipe.steps
                # sub_samples.append(
                #     self.parallel_interface.submit(self._steps_function, samp, steps=steps, func_name="apply")
                # )
        return tuple(self.parallel_interface.collect(sub_samples))

    def undo(self, sample):
        """Undo the effects of the branches.

        This will still result in a tuple, so a unify may be needed after.

        If each branch executed a fully reversable operation, and originated from the same data source, each sample 'should' be identical.
        """
        sub_samples = []

        if not isinstance(sample, tuple):  # If not tuple, provide it to all pipelines
            sample = tuple([sample] * len(self.sub_pipelines))

        if self._map_copy and not len(sample) == len(self.sub_pipelines):
            self.sub_pipelines = expand_pipeline(self, len(sample))

        if not len(sample) == len(self.sub_pipelines):
            warnings.warn(
                "The length of samples, and number of branchpoints differed.",
                PipelineWarning,
            )
        with parallel.disable:
            for samp, sub_pipe in zip(sample, self.sub_pipelines):
                sub_samples.append(self.parallel_interface.submit(sub_pipe.undo, samp))

                # steps = sub_pipe.steps[::-1]
                # if not isinstance(steps[-1], PipelineStep) and isinstance(steps[-1], (Index,)):
                #     steps = steps[:-1]  # Remove last step on undo path if not PipelineStep

                # sub_samples.append(
                #     self.parallel_interface.submit(self._steps_function, samp, steps=steps, func_name="undo")
                # )
            result = tuple(self.parallel_interface.collect(sub_samples))

        if all(len(pipe.steps) == 1 and isinstance(pipe.steps[0], Index) for pipe in self.sub_pipelines):
            # if all(map(lambda x: result[0] == x, result[1:])):
            return result[0]
        return result

    def apply_func(self, sample):  # pragma: no cover
        ...

    def undo_func(self, sample):  # pragma: no cover
        ...

    # TODO Add steps and complete steps override
    @property
    def complete_steps(self):
        return tuple(x.complete_steps for x in self.sub_pipelines)

    def _get_tree(
        self, parent: list[str] | None = None, graph: Optional[graphviz.Digraph] = None
    ) -> tuple[graphviz.Digraph, list[str]]:  # type: ignore # pragma: no cover
        import uuid

        graph = graph or graphviz.Digraph()
        final_steps = []
        prior_step = parent

        if prior_step is not None and (isinstance(prior_step, list) and len(prior_step) > 1):
            # If not first or with multiple parents or mapping
            branch_name = f"{self.__class__.__name__}_{uuid.uuid4()!s}"
            graph.node(
                branch_name,
                "Branch Point" if not self._map else "Mapping",
                shape="diamond",
            )
            if isinstance(prior_step, list):
                for p in prior_step:
                    graph.edge(p, branch_name)
            else:
                graph.edge(prior_step, branch_name)
            prior_step = [branch_name]

        if self._map_copy and False:  # Disabling until a better visualisation is made
            sub_pipe = self.sub_pipelines[0]
            name = f"cluster_{uuid.uuid4()!s}" if len(sub_pipe.flattened_steps) > 1 and False else f"{uuid.uuid4()!s}"
            with graph.subgraph(name=name) as c:  # type: ignore
                _, prior_steps = sub_pipe._get_tree(prior_step, graph=c)
            final_steps.extend(prior_steps)

        else:
            for sub_pipes in self.sub_pipelines:
                name = (
                    f"cluster_{uuid.uuid4()!s}" if len(sub_pipes.flattened_steps) > 1 and False else f"{uuid.uuid4()!s}"
                )
                with graph.subgraph(name=name) as c:  # type: ignore
                    _, prior_steps = sub_pipes._get_tree(prior_step if not sub_pipes.has_source() else [], graph=c)  # type: ignore
                final_steps.extend(prior_steps)

        return graph, final_steps

    def __repr__(self) -> str:  # pragma: no cover
        repr_str = f"{self.__class__.__name__}:\n"
        for step in self.sub_pipelines:
            for s in step.steps:
                repr_s = str(repr(s)).replace("\n", "\n\t")
                repr_str += f"\t|-{repr_s}"
            if not repr_str.endswith("\n"):
                repr_str += "\n"
        return repr_str

    # def __str__(self):
    #     return repr(self)
