# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
`pyearthtools.data` Warnings

"""

import warnings


class pyearthtoolsDataWarning(Warning):
    """General warning for `pyearthtools.data` processes."""


class IndexWarning(pyearthtoolsDataWarning):
    """Data Index Warning."""

    pass


class AccessorRegistrationWarning(pyearthtoolsDataWarning):
    """Warning for conflicts in object registration."""

    pass


warnings.filterwarnings(action="default", category=pyearthtoolsDataWarning)
warnings.filterwarnings(action="module", category=IndexWarning)
