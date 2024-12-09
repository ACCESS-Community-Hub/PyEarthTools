# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations

from typing import Literal, Optional, Type, Union

import numpy as np

from pyearthtools.pipeline.step import PipelineStep

from pyearthtools.pipeline.decorators import potentialabstractmethod, PotentialABC
from pyearthtools.pipeline import parallel

__all__ = ["Operation"]


class Operation(PipelineStep, PotentialABC):
    """Pipeline Operation"""

    def __init__(
        self,
        *,
        split_tuples: Literal["apply", "undo", True, False] = False,
        recursively_split_tuples: bool = False,
        operation: Literal["apply", "undo", "both"] = "both",
        recognised_types: Optional[
            Union[
                tuple[Type, ...],
                Type,
                dict[Literal["apply", "undo"], Union[tuple[Type, ...], Type]],
            ]
        ] = None,
        response_on_type: Literal["warn", "exception", "ignore", "filter"] = "exception",
        **kwargs,
    ):
        """
        Base `Pipeline` Operation,

        Allows for tuple spliting, and type checking

        Args:
            split_tuples (Literal['apply', 'undo', True, False], optional):
                Split tuples on associated actions, if bool, apply to all functions. Defaults to False.
            recursively_split_tuples (bool, optional):
                Recursively split tuples. Defaults to False.
            operation (Literal['apply', 'undo', 'both'], optional):
                Which functions to apply operation to.
                If not 'apply' apply does nothing, same for `undo`. Defaults to "both".
            recognised_types (Optional[Union[tuple[Type, ...], Type, dict[str, Union[tuple[Type, ...], Type]]] ], optional):
                Types recognised, can be dictionary to reference different types per function Defaults to None.
            response_on_type (Literal['warn', 'exception', 'ignore', 'filter'], optional):
                Response when invalid type found. Defaults to "exception".
        """
        if isinstance(split_tuples, str):
            func_name = {"apply": "apply_func", "undo": "undo_func"}
            _split_tuples = {func_name[split_tuples]: True}
        else:
            _split_tuples = split_tuples

        if recognised_types is not None:
            if isinstance(recognised_types, dict):
                recognised_types = {
                    key: val if isinstance(val, tuple) else (val,) for key, val in recognised_types.items()
                }
            else:
                recognised_types = recognised_types if isinstance(recognised_types, tuple) else (recognised_types,)
                recognised_types = {"apply": recognised_types, "undo": recognised_types}

        super().__init__(
            split_tuples=_split_tuples,
            recursively_split_tuples=recursively_split_tuples,
            recognised_types=recognised_types,  # type: ignore
            response_on_type=response_on_type,
            **kwargs,
        )

        self._operation: dict[Literal["apply", "undo"], bool] = {
            "apply": operation in ["both", "apply"],
            "undo": operation in ["both", "undo"],
        }

        self.check_abstractions(
            [{"apply": "apply_func", "undo": "undo_func"}[key] for key, val in self._operation.items() if val]
        )

    def run(self, sample):
        return self.apply(sample)

    def apply(self, sample):
        """Run the `apply_func` on sample, splitting tuples if needed"""
        if not self._operation["apply"]:
            return sample
        self.check_type(sample, func_name="apply")
        if isinstance(sample, np.ndarray) or (
            isinstance(sample, tuple) and any(map(lambda x: isinstance(x, np.ndarray), sample))
        ):
            with parallel.disable:
                return self._split_tuples_call(sample, _function="apply_func")
        return self._split_tuples_call(sample, _function="apply_func")

    def undo(self, sample):
        """Run the `undo_func` on sample, splitting tuples if needed"""
        if not self._operation["undo"]:
            return sample
        self.check_type(sample, func_name="undo")
        if isinstance(sample, np.ndarray) or (
            isinstance(sample, tuple) and any(map(lambda x: isinstance(x, np.ndarray), sample))
        ):
            with parallel.disable:
                return self._split_tuples_call(sample, _function="undo_func")
        return self._split_tuples_call(sample, _function="undo_func")

    @property
    def T(self):
        """
        Transposed Operation.

        Swaps `apply` with `undo` so that this operation behaves inversely to normal.

        """
        new_operation = self.copy()
        new_operation.apply, new_operation.undo = new_operation.undo, new_operation.apply

        new_operation._operation["apply"] = self._operation["undo"]
        new_operation._operation["undo"] = self._operation["apply"]

        new_operation._property = "T"
        return new_operation

    @potentialabstractmethod
    def apply_func(self, sample):
        return sample

    @potentialabstractmethod
    def undo_func(self, sample):
        return sample

    def __str__(self):
        return str(self.__class__.__name__)
