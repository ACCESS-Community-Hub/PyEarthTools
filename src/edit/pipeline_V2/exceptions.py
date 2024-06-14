# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.


from typing import Any


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
