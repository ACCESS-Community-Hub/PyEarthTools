# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty
# of any kind including, but not limited to, fitness for a particular
# purpose. The user assumes the entire risk as to the use and
# performance of the software. In no event shall the copyright holder
# be held liable for any claim, damages or other liability arising
# from the use of the software.

from __future__ import annotations
import functools

from typing import Callable
import math

from pyearthtools.data import TimeDelta, pyearthtoolsDatetime


def parse_recurrent(interval: int | TimeDelta = 1):
    """
    Parse kwargs given to a recurrent function.

    Converts to `steps`, number of model steps to run

    !!! Supported

        | Kwarg | Description |
        | ----- | ----------- |
        | steps | Default value, number of steps of model, all are converted to this |
        | time  | Time value given in same units as interval |
        | to_time  | Time to predict up to, uses `interval` to get steps from current. |

    !!! Examples
        ```python
        @parse_recurrent(interval = 6) # 6 hour interval
        def func(*args, steps, **kwargs):
            ....
        func(steps = 10) # Nothing, run 10 steps
        func(time = 48) # Time of 48 hours, becomes `steps = 8`
        ```

    Args:
        interval (int | TimeDelta, optional):
            Time interval to convert with. Defaults to 1.
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def parse(*args, **kwargs):
            if "steps" in kwargs:
                pass
            elif "time" in kwargs:
                time = kwargs.pop("time")
                kwargs["steps"] = math.ceil(time / interval)
            elif "to_time" in kwargs:
                to_time = kwargs.pop("to_time")
                kwargs["steps"] = math.ceil((pyearthtoolsDatetime("current") - to_time) / interval)
            return func(*args, **kwargs)

        return parse

    return decorator
