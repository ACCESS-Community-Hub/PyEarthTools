# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

"""
Function parameter searching


"""

import itertools
from typing import Callable
import tqdm.auto as tqdm

import concurrent.futures

_executor = concurrent.futures.ThreadPoolExecutor()


class SingleParameter:
    """
    Single parameter for searching

    Examples
    >>> list(SingleParameter((0,3)))
    [(0, 3)]
    """

    def __init__(self, item):
        self._item = item

    def __iter__(self):
        yield self._item


class ListParameter:
    """
    Parameter which is a list, and each element is with a range

    Examples
    >>> list(ListParameter.from_minmax(0, 3, 3))
    [(0, 0, 0),
     (0, 0, 1),
     ...
     (2, 2, 1),
     (2, 2, 2)]

    >>> list(ListParameter(['a','b','c'], 3))
    [('a', 'a', 'a'),
     ('a', 'a', 'b'),
     ...
     ('c', 'c', 'b'),
     ('c', 'c', 'c')]
    """

    def __init__(self, element_range: list, num_elements: int):
        self._element_range = element_range
        self._num_elements = num_elements

    def __iter__(self):
        for i in itertools.product(*[list(self._element_range)] * self._num_elements):
            yield i

    @staticmethod
    def from_minmax(min: int, max: int, *args, **kwargs):
        return ListParameter(range(min, max), *args, **kwargs)


class RangeParameter:
    """
    Integer range of parameters

    Examples
    >>> list(RangeParameter(0,3))
    [0, 1, 2]
    """

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __iter__(self):
        for i in range(*self._args, **self._kwargs):
            yield i


def search(function: Callable, verbose: bool = False, **kwargs) -> list[dict]:
    """
    Try calling the given function with every permuation of the given arguments

    Suggested to use SingleParameter, ListParameter, RangeParameter to create the arguments

    Args:
        function (Callable):
            Function to call
        verbose (bool, optional):
            Whether to print configs. Defaults to False.

    Returns:
        (list[dict]):
            List of valid configurations
    """
    keys = list(kwargs.keys())
    items = [val for _, val in kwargs.items()]

    valid_configs = []
    all_configs = list(itertools.product(*items))

    try:
        for config in tqdm.tqdm(all_configs, disable=not verbose):
            remapped = {keys[i]: val for i, val in enumerate(config)}
            try:
                function(**remapped)
                valid_configs.append(remapped)
                if verbose:
                    print(f"Valid config: {remapped}")
            except Exception as e:
                pass
    except KeyboardInterrupt:
        pass

    return valid_configs


def search_threaded(function: Callable, verbose: bool = False, **kwargs) -> list[dict]:
    """
    Threaded version of `search`

    Try calling the given function with every permuation of the given arguments

    Suggested to use SingleParameter, ListParameter, RangeParameter to create the arguments

    Args:
        function (Callable):
            Function to call
        verbose (bool, optional):
            Whether to print configs. Defaults to False.

    Returns:
        (list[dict]):
            List of valid configurations
    """
    keys = list(kwargs.keys())
    items = [val for _, val in kwargs.items()]

    valid_configs = []
    all_configs = list(itertools.product(*items))

    def run(function, **kwargs):
        try:
            function(**kwargs)
            return kwargs
        except Exception:
            return None

    futures = []
    for config in all_configs:
        remapped = {keys[i]: val for i, val in enumerate(config)}
        futures.append(_executor.submit(run, function, **remapped))

    try:
        for future in tqdm.tqdm(futures, disable=not verbose):
            result = future.result()
            if result:
                valid_configs.append(result)
                if verbose:
                    print(f"Valid config: {result}")

    except KeyboardInterrupt:
        pass

    return valid_configs
