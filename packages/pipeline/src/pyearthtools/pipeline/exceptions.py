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



from typing import Any, Optional, Type
import warnings

import pyearthtools.utils
from pyearthtools.pipeline.warnings import PipelineWarning

__all__ = [
    "PipelineException",
    "PipelineFilterException",
    "PipelineUnificationException",
    "PipelineTypeError",
    "PipelineRuntimeError",
    "ExceptionIgnoreContext",
]


class PipelineException(Exception):
    """General Pipeline Exception"""


class PipelineFilterException(PipelineException):
    """Pipeline Filter Exception

    Indicates that the filter has detected an invalid sample.
    """

    def __init__(self, sample: Any, message: str = "", *args):
        """
        FilterException

        Args:
            sample (Any):
                Sample found to be invalid
            message (str, optional):
                Msg for the user. Defaults to "".
        """
        self.sample = sample
        self.message = message

        super(Exception, self).__init__(message, *args)

    def __str__(self):
        return f"Filtering the pipeline yielded an invalid sample, {self.message}\n{self.sample}."


class PipelineUnificationException(PipelineException):
    """Pipeline unification Exception

    Indicates a unification check failed

    """

    def __init__(self, sample: Any, message: str = "", *args):
        self.sample = sample
        self.message = message

        super(Exception, self).__init__(message, *args)

    def __str__(self):
        return f"Unifying the pipeline failed {self.message}\n{self.sample}."


class PipelineTypeError(PipelineException, TypeError):
    """Pipeline Type error"""


class PipelineRuntimeError(PipelineException, RuntimeError):
    """Pipeline Runtime Error"""


class ExceptionIgnoreContext:
    """
    Ignore Exceptions

    Will count how many `Exceptions` have been thrown, and warn if over `max_exceptions`.
    """

    def __init__(self, exceptions: tuple[Type[Exception], ...], max_exceptions: Optional[int] = None):

        self._max_exceptions = max_exceptions or pyearthtools.utils.config.get("pipeline.exceptions.max_filter")
        self._count = 0
        self._messages: list[str] = []
        self._exceptions = exceptions

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, traceback):
        if exc_type in self._exceptions:
            self._count += 1
            self._messages.append(str(exc_val))

        if self._count >= self._max_exceptions:
            str_msg = "\n".join(self._messages)

            warnings.warn(
                f"{self._count} exception's have occured.\nRaised the following messages:\n{str_msg}",
                PipelineWarning,
            )
            self._count = 0
            self._messages = []
