# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

"""
Decorators for `pyearthtools`
"""
import warnings

warnings.warn(
    "All decorators once here have been moved from `pyearthtools.data.decorators` to `pyearthtools.utils.decorators`, and will be removed here in the future",
    FutureWarning,
)

from pyearthtools.utils.decorators import alias_arguments
