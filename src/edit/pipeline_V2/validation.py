# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations

from typing import Type, Iterable, Optional, Union, Any

from edit.pipeline_V2.exceptions import PipelineTypeError


def as_tuple(obj) -> tuple[Any, ...]:
    if not isinstance(obj, tuple):
        return (obj,)
    return obj


def filter_steps(
    steps: Iterable,
    valid_types: Union[Type, tuple[Type, ...]],
    invalid_types: Optional[Union[Type, tuple[Type, ...]]] = None,
    *,
    responsible: Optional[str] = None,
):
    """Check if `steps` are of `valid_types`"""
    valid_types_str = tuple(map(lambda x: x, as_tuple(valid_types)))
    invalid_types_str = tuple(map(lambda x: x, as_tuple(invalid_types)))

    for s in steps:
        if not isinstance(s, valid_types):
            error_msg = f"found an invalid type.\n {type(s)} not in valid {valid_types_str}."
            if responsible:
                error_msg = f"Filtering pipeline steps for {responsible}{error_msg}."
            else:
                error_msg = f"Filtering pipeline steps {error_msg}."
            raise PipelineTypeError(error_msg)

        if invalid_types is not None and isinstance(s, invalid_types):
            error_msg = f"found an invalid type.\n {type(s)} in invalid {invalid_types_str}."
            if responsible:
                error_msg = f"Filtering pipeline steps for {responsible}{error_msg}."
            else:
                error_msg = f"Filtering pipeline steps {error_msg}."
            raise PipelineTypeError(error_msg)
