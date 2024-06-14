# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.


from abc import abstractmethod, ABCMeta
from typing import Optional, Type, Union

import warnings

from edit.pipeline_V2 import config
from edit.pipeline_V2.step import PipelineStep
from edit.pipeline_V2.exceptions import PipelineFilterException
from edit.pipeline_V2.warnings import PipelineWarning

__all__ = ["Filter", "FilterCheck", "FilterWarningContext", "TypeFilter"]


class Filter(PipelineStep, metaclass=ABCMeta):
    """
    Base Filter

    Allows for samples to be skipped if found to be invalid.

    `filter` should raise a `PipelineFilterException` if invalid.
    """

    def __init__(
        self,
        split_tuples: Union[dict[str, bool], bool] = True,
        recursively_split_tuples: bool = True,
        recognised_types: Optional[
            Union[
                tuple[Type, ...],
                Type,
                dict[str, Union[tuple[Type, ...], Type]],
            ]
        ] = None,
    ):
        super().__init__(
            split_tuples, recursively_split_tuples, recognised_types=recognised_types, response_on_type="filter"
        )

    def run(self, sample):
        """Run filtering"""
        self.filter(sample)
        return sample

    @abstractmethod
    def filter(self, sample) -> None:
        """To be implemented by child class, should raise a `PipelineFilterException` if `sample` is invalid."""
        raise PipelineFilterException(sample)


class FilterCheck(Filter):
    """
    Subclass of `Filter` to automate exception,

    Just needs `check` to return a `bool`.
    """

    @abstractmethod
    def check(self, sample) -> bool:
        return True

    def filter(self, sample) -> None:
        if not self.check(sample):
            raise PipelineFilterException(sample, f"{self.__class__.__name__} found the sample to be invalid.")


class FilterWarningContext:
    """
    Filter Warning context

    Will count how many `PipelineFilterException` have been thrown, and warn if over `config.MAX_FILTER_EXCEPTIONS`.
    """

    def __init__(self, max_exceptions: Optional[int] = None):

        self._max_exceptions = max_exceptions or config.MAX_FILTER_EXCEPTIONS
        self._count = 0
        self._messages = []

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, traceback):
        if exc_type == PipelineFilterException:
            self._count += 1
            self._messages.append(str(exc_val))

        if self._count >= self._max_exceptions:
            str_msg = "\n".join(self._messages)

            warnings.warn(
                f"{self._count} PipelineFilterException's have occured.\nRaised the following messages:\n{str_msg}",
                PipelineWarning,
            )
            self._count = 0
            self._messages = []

class TypeFilter(Filter):
    """
    Filter if type is wrong
    """

    def __init__(self, valid_types: Union[tuple[Type], Type], *, split_tuples: bool = False):
        super().__init__(split_tuples=split_tuples)
        self.record_initialisation()

        if not isinstance(valid_types, tuple):
            valid_types = (valid_types,)
        self._valid_types = valid_types

    def filter(self, sample) -> None:
        if not isinstance(sample, self._valid_types):
            raise PipelineFilterException(sample, f"Expecting type/s {self._valid_types}")
