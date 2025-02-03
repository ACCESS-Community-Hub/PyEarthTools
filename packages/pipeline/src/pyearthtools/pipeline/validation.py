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

from typing import Type, Iterable, Optional, Union, Any

from pyearthtools.pipeline.exceptions import PipelineTypeError


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
